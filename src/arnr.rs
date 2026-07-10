//! Motion-compensated temporal filtering for altref synthesis.
//!
//! Builds the *picture* that [`crate::encoder::encode_invisible_altref_update`]
//! installs into the §9.7 ALTREF slot: a noise-reduced blend of a small
//! window of source frames, aligned per 16×16 block by whole-pel motion
//! search. Prediction from a temporally-filtered anchor is stronger than
//! from any single source frame on noisy content, because the residual
//! against every nearby frame shrinks once the (uncorrelated) noise is
//! averaged out of the reference.
//!
//! RFC 6386 is silent on all of this by design — the bitstream only
//! defines how an altref picture is *transported* (§9.1 `show_frame`,
//! §9.7 `refresh_alternate_frame`), never how an encoder should build
//! one. The filter here is therefore a clean-room encoder-quality
//! decision, like the two-pass rate-control family:
//!
//! 1. For every 16×16 block of the **center** frame, each *other* frame
//!    in the window is aligned by a whole-pel three-round refinement
//!    search (±15 px, step 8 → 4 → 2 → 1) minimising SAD against the
//!    center block. Edge-clamped fetches make every candidate valid.
//! 2. A block whose best per-pixel SAD is still large is dropped from
//!    the blend for that block (occlusion / scene-change guard) — a bad
//!    match must not bleed foreign content into the anchor.
//! 3. Surviving pixels blend with a per-pixel weight that decays with
//!    the absolute difference from the center pixel
//!    (`w = W_MAX·S / (S + d²)`, `S` scaled by
//!    [`ArnrConfig::strength`]), so detail preserved in only one frame
//!    is not smeared. The center pixel always carries full weight —
//!    strength 0 degenerates to an exact copy of the center frame.
//!
//! Chroma planes reuse the luma motion (halved, as §18.2 does for
//! chroma MVs) with the same difference-driven weighting.

use crate::encoder::{EncodeError, I420Frame};

/// Full blend weight (the center pixel's constant weight).
const W_MAX: u32 = 16;

/// Per-pixel SAD ceiling (per luma pixel, ×256 per block) above which a
/// motion-compensated block is considered a bad match and excluded from
/// the blend entirely.
const BLOCK_SAD_PER_PIXEL_CUTOFF: u32 = 12;

/// Configuration for [`build_arnr_altref`].
///
/// `strength` is the only dial: `0` disables filtering (the output is a
/// copy of the center frame), higher values weight neighbouring frames'
/// pixels more aggressively at a given pixel difference. Values above
/// [`ArnrConfig::MAX_STRENGTH`] are clamped by [`ArnrConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArnrConfig {
    /// Filter aggressiveness, `0..=6`. `0` = pass-through (center frame
    /// copied verbatim); `6` = strongest blending. Default `3`.
    pub strength: u8,
}

impl ArnrConfig {
    /// Largest meaningful [`strength`](Self::strength).
    pub const MAX_STRENGTH: u8 = 6;

    /// Build a config, clamping `strength` into `0..=MAX_STRENGTH`.
    pub fn new(strength: u8) -> Self {
        ArnrConfig {
            strength: strength.min(Self::MAX_STRENGTH),
        }
    }
}

impl Default for ArnrConfig {
    fn default() -> Self {
        ArnrConfig { strength: 3 }
    }
}

/// An owned I420 picture produced by [`build_arnr_altref`] — the
/// synthetic altref anchor. Feed [`Self::as_i420`] to
/// [`crate::encoder::encode_invisible_altref_update`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArnrPicture {
    /// Visible width in luma pixels.
    pub width: u32,
    /// Visible height in luma pixels.
    pub height: u32,
    /// Luma plane, `width × height`, tightly packed.
    pub y: Vec<u8>,
    /// U plane, `((width+1)/2) × ((height+1)/2)`, tightly packed.
    pub u: Vec<u8>,
    /// V plane, same dimensions as `u`.
    pub v: Vec<u8>,
}

impl ArnrPicture {
    /// Borrow the picture as a tightly-packed [`I420Frame`].
    pub fn as_i420(&self) -> I420Frame<'_> {
        I420Frame::packed(self.width, self.height, &self.y, &self.u, &self.v)
    }
}

/// Edge-clamped pixel fetch from a strided plane.
#[inline]
fn pel(plane: &[u8], stride: usize, w: i32, h: i32, x: i32, y: i32) -> u8 {
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    plane[cy * stride + cx]
}

/// SAD of one `bw × bh` block of `src` (anchored at `(bx, by)`, fully
/// in-bounds) against the edge-clamped block of `refp` displaced by
/// `(mvx, mvy)`, accumulating each candidate row exactly as
/// [`block_sad_generic`] does.
///
/// Two cost levers over the straightforward per-pixel form, neither of
/// which changes any returned value the caller can observe:
///
/// * **in-bounds fast path** — when the whole displaced block lands
///   strictly inside the reference plane (the overwhelmingly common
///   case away from frame edges), the per-pixel `clamp` + row-base
///   multiply of [`pel`] collapses to two straight row slices. The
///   clamped generic path is kept verbatim for the genuine edge case
///   and doubles as the equivalence reference
///   (`block_sad_fast_matches_generic` pins them pixel-for-pixel).
/// * **monotone early exit** — the row-partial SAD only ever grows, so
///   once it reaches `early_exit_at` the final SAD is guaranteed to be
///   `>= early_exit_at` and the candidate can be abandoned. The caller
///   ([`refine_search`]) only accepts a candidate on a *strictly
///   smaller* SAD, so returning any partial `>= early_exit_at` selects
///   the identical winner with the identical best SAD: a candidate
///   that would win never triggers the exit (every partial of a
///   winning candidate is below the incumbent best), and a losing
///   candidate is rejected either way
///   (`refine_search_matches_no_early_exit_reference` pins the whole
///   descent).
fn block_sad(
    src: &I420Frame<'_>,
    refp: &I420Frame<'_>,
    (bx, by, bw, bh): (usize, usize, usize, usize),
    (mvx, mvy): (i32, i32),
    early_exit_at: u32,
) -> u32 {
    let w = src.width as i32;
    let h = src.height as i32;
    let x0 = bx as i32 + mvx;
    let y0 = by as i32 + mvy;
    let in_bounds = x0 >= 0 && y0 >= 0 && x0 + bw as i32 <= w && y0 + bh as i32 <= h;
    if !in_bounds {
        return block_sad_generic(src, refp, (bx, by, bw, bh), (mvx, mvy), early_exit_at);
    }
    let (rx0, ry0) = (x0 as usize, y0 as usize);
    let mut sad = 0u32;
    for r in 0..bh {
        let sy = by + r;
        let src_row = &src.y[sy * src.y_stride + bx..sy * src.y_stride + bx + bw];
        let ref_base = (ry0 + r) * refp.y_stride + rx0;
        let ref_row = &refp.y[ref_base..ref_base + bw];
        for (&s, &rp) in src_row.iter().zip(ref_row.iter()) {
            sad += (s as i32 - rp as i32).unsigned_abs();
        }
        if sad >= early_exit_at {
            return sad;
        }
    }
    sad
}

/// The straightforward per-pixel edge-clamped SAD — the reference form
/// [`block_sad`] must match, and the path it takes when the displaced
/// block genuinely crosses a frame edge. Carries the same per-row
/// monotone early exit (the partial sum is identical row-for-row, so
/// the exit fires at exactly the same row on both paths).
fn block_sad_generic(
    src: &I420Frame<'_>,
    refp: &I420Frame<'_>,
    (bx, by, bw, bh): (usize, usize, usize, usize),
    (mvx, mvy): (i32, i32),
    early_exit_at: u32,
) -> u32 {
    let w = src.width as i32;
    let h = src.height as i32;
    let mut sad = 0u32;
    for r in 0..bh {
        let sy = by + r;
        let src_row = &src.y[sy * src.y_stride + bx..sy * src.y_stride + bx + bw];
        for (c, &s) in src_row.iter().enumerate() {
            let rp = pel(
                refp.y,
                refp.y_stride,
                w,
                h,
                (bx + c) as i32 + mvx,
                sy as i32 + mvy,
            );
            sad += (s as i32 - rp as i32).unsigned_abs();
        }
        if sad >= early_exit_at {
            return sad;
        }
    }
    sad
}

/// Whole-pel three-round refinement search: start at the zero MV, probe
/// the 8 neighbours at step 8, recentre, halve the step (8 → 4 → 2 → 1).
/// Returns `(mvx, mvy, sad)` of the best whole-pel alignment within
/// ±15 px.
fn refine_search(
    src: &I420Frame<'_>,
    refp: &I420Frame<'_>,
    blk: (usize, usize, usize, usize),
) -> (i32, i32, u32) {
    let mut best = (0i32, 0i32);
    let mut best_sad = block_sad(src, refp, blk, (0, 0), u32::MAX);
    let mut step = 8i32;
    while step >= 1 {
        let center = best;
        for (dx, dy) in [
            (-step, -step),
            (0, -step),
            (step, -step),
            (-step, 0),
            (step, 0),
            (-step, step),
            (0, step),
            (step, step),
        ] {
            let cand = (center.0 + dx, center.1 + dy);
            if cand.0.abs() > 15 || cand.1.abs() > 15 {
                continue;
            }
            // `best_sad` as the exit bound: only a strictly smaller SAD
            // wins, so a candidate whose partial sum already reaches
            // `best_sad` is rejected with or without the exit.
            let sad = block_sad(src, refp, blk, cand, best_sad);
            if sad < best_sad {
                best_sad = sad;
                best = cand;
            }
        }
        step /= 2;
    }
    (best.0, best.1, best_sad)
}

/// Difference-driven blend weight: `W_MAX·S / (S + d²)` — full weight at
/// `d = 0`, decaying with the squared pixel difference; `S` grows with
/// the configured strength so stronger settings tolerate larger
/// differences before down-weighting.
#[inline]
fn pixel_weight(d: i32, s_scale: u32) -> u32 {
    (W_MAX * s_scale) / (s_scale + (d * d) as u32)
}

/// Precompute [`pixel_weight`] for every possible absolute pixel
/// difference (`|d| <= 255` for 8-bit planes). The weight depends on
/// `d` only through `d²`, so one 256-entry table replaces the per-pixel
/// integer division in the blend accumulation. Bit-identical by
/// construction (`weight_table_matches_pixel_weight` sweeps every
/// `(d, strength)` pair).
fn weight_table(s_scale: u32) -> [u32; 256] {
    let mut table = [0u32; 256];
    for (d, slot) in table.iter_mut().enumerate() {
        *slot = pixel_weight(d as i32, s_scale);
    }
    table
}

/// Accumulate one motion-aligned luma block of a window frame into the
/// blend accumulators.
///
/// `ctr_y` is the tightly-packed `w × h` center copy (`out.y`); the
/// reference plane arrives strided. The output positions `(bx..bx+bw,
/// by..by+bh)` are always in-bounds (the block grid never leaves the
/// frame); only the *displaced* reference fetch can cross an edge. When
/// it does not — the overwhelmingly common case — the per-pixel
/// edge-clamping [`pel`] collapses to a straight row slice; the clamped
/// generic form is kept verbatim below and pinned pixel-for-pixel by
/// `accumulate_luma_block_fast_matches_generic`.
#[allow(clippy::too_many_arguments)]
fn accumulate_luma_block(
    ctr_y: &[u8],
    (f_y, f_stride): (&[u8], usize),
    (w, h): (usize, usize),
    (bx, by, bw, bh): (usize, usize, usize, usize),
    (mvx, mvy): (i32, i32),
    wt_table: &[u32; 256],
    acc: &mut [u32],
    wsum: &mut [u32],
) {
    let x0 = bx as i32 + mvx;
    let y0 = by as i32 + mvy;
    if x0 >= 0 && y0 >= 0 && x0 + bw as i32 <= w as i32 && y0 + bh as i32 <= h as i32 {
        // In-bounds fast path: straight row slices, no per-pixel clamp.
        let (rx0, ry0) = (x0 as usize, y0 as usize);
        for r in 0..bh {
            let out_base = (by + r) * w + bx;
            let ref_base = (ry0 + r) * f_stride + rx0;
            let ctr_row = &ctr_y[out_base..out_base + bw];
            let ref_row = &f_y[ref_base..ref_base + bw];
            let acc_row = &mut acc[out_base..out_base + bw];
            let wsum_row = &mut wsum[out_base..out_base + bw];
            for c in 0..bw {
                let ref_px = ref_row[c] as i32;
                let d = ref_px - ctr_row[c] as i32;
                let wt = wt_table[d.unsigned_abs() as usize];
                acc_row[c] += wt * ref_px as u32;
                wsum_row[c] += wt;
            }
        }
        return;
    }
    // Edge-crossing generic path — the reference form.
    for r in 0..bh {
        let yy = by + r;
        for c in 0..bw {
            let xx = bx + c;
            let ctr_px = ctr_y[yy * w + xx] as i32;
            let ref_px = pel(
                f_y,
                f_stride,
                w as i32,
                h as i32,
                xx as i32 + mvx,
                yy as i32 + mvy,
            ) as i32;
            let wt = wt_table[(ref_px - ctr_px).unsigned_abs() as usize];
            acc[yy * w + xx] += wt * ref_px as u32;
            wsum[yy * w + xx] += wt;
        }
    }
}

/// Accumulate one motion-aligned chroma block (one plane — called once
/// for U and once for V) into the blend accumulators.
///
/// Unlike luma, the chroma *output* coordinates are themselves clamped
/// (`.min(cw - 1)` / `.min(ch - 1)`) because an odd-dimension plane's
/// last block row/column can overhang — and that clamping deliberately
/// accumulates twice into the edge pixel, which must be preserved. The
/// fast path therefore requires the output block to be overhang-free
/// *and* the displaced reference fetch to be in-bounds; anything else
/// takes the generic clamped form (pinned by
/// `accumulate_chroma_block_fast_matches_generic`).
#[allow(clippy::too_many_arguments)]
fn accumulate_chroma_block(
    ctr_plane: &[u8],
    (f_plane, f_stride): (&[u8], usize),
    (cw, ch): (usize, usize),
    (cbx, cby, cbw, cbh): (usize, usize, usize, usize),
    (cmvx, cmvy): (i32, i32),
    wt_table: &[u32; 256],
    acc: &mut [u32],
    wsum: &mut [u32],
) {
    let x0 = cbx as i32 + cmvx;
    let y0 = cby as i32 + cmvy;
    let output_in_bounds = cbx + cbw <= cw && cby + cbh <= ch;
    if output_in_bounds
        && x0 >= 0
        && y0 >= 0
        && x0 + cbw as i32 <= cw as i32
        && y0 + cbh as i32 <= ch as i32
    {
        let (rx0, ry0) = (x0 as usize, y0 as usize);
        for r in 0..cbh {
            let out_base = (cby + r) * cw + cbx;
            let ref_base = (ry0 + r) * f_stride + rx0;
            let ctr_row = &ctr_plane[out_base..out_base + cbw];
            let ref_row = &f_plane[ref_base..ref_base + cbw];
            let acc_row = &mut acc[out_base..out_base + cbw];
            let wsum_row = &mut wsum[out_base..out_base + cbw];
            for c in 0..cbw {
                let ref_px = ref_row[c] as i32;
                let d = ref_px - ctr_row[c] as i32;
                let wt = wt_table[d.unsigned_abs() as usize];
                acc_row[c] += wt * ref_px as u32;
                wsum_row[c] += wt;
            }
        }
        return;
    }
    // Overhanging / edge-crossing generic path — the reference form,
    // including the double-accumulation into clamped edge pixels.
    for r in 0..cbh {
        let yy = (cby + r).min(ch - 1);
        for c in 0..cbw {
            let xx = (cbx + c).min(cw - 1);
            let ctr_px = ctr_plane[yy * cw + xx] as i32;
            let ref_px = pel(
                f_plane,
                f_stride,
                cw as i32,
                ch as i32,
                xx as i32 + cmvx,
                yy as i32 + cmvy,
            ) as i32;
            let wt = wt_table[(ref_px - ctr_px).unsigned_abs() as usize];
            acc[yy * cw + xx] += wt * ref_px as u32;
            wsum[yy * cw + xx] += wt;
        }
    }
}

/// Build a temporally-filtered altref anchor from a window of source
/// frames.
///
/// * `frames` — the lookahead window, oldest first. All frames must
///   share the center frame's dimensions
///   ([`EncodeError::ReferenceDimensionsMismatch`] otherwise).
/// * `center` — index of the anchor frame the output is aligned to
///   (panics if out of range — a caller bug, not a data condition).
/// * `cfg` — see [`ArnrConfig`]; `strength = 0` (or a single-frame
///   window) returns an exact copy of the center frame.
///
/// The output picture is what the caller should feed to
/// [`crate::encoder::encode_invisible_altref_update`]; it is **not**
/// itself a displayed frame, so blending artifacts trade off only
/// against prediction quality, never against on-screen fidelity.
pub fn build_arnr_altref(
    frames: &[I420Frame<'_>],
    center: usize,
    cfg: &ArnrConfig,
) -> Result<ArnrPicture, EncodeError> {
    assert!(
        center < frames.len(),
        "build_arnr_altref: center index {center} out of range ({} frames)",
        frames.len()
    );
    let ctr = &frames[center];
    let w = ctr.width as usize;
    let h = ctr.height as usize;
    if ctr.width == 0 || ctr.height == 0 || ctr.width > 0x3FFF || ctr.height > 0x3FFF {
        return Err(EncodeError::InvalidDimensions {
            width: ctr.width,
            height: ctr.height,
        });
    }
    for f in frames {
        if f.width != ctr.width || f.height != ctr.height {
            return Err(EncodeError::ReferenceDimensionsMismatch {
                source: (ctr.width, ctr.height),
                reference: (f.width, f.height),
            });
        }
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    // Pass-through short-circuits: nothing to blend.
    let mut out = ArnrPicture {
        width: ctr.width,
        height: ctr.height,
        y: copy_plane(ctr.y, ctr.y_stride, w, h),
        u: copy_plane(ctr.u, ctr.uv_stride, cw, ch),
        v: copy_plane(ctr.v, ctr.uv_stride, cw, ch),
    };
    if cfg.strength == 0 || frames.len() < 2 {
        return Ok(out);
    }
    // Strength → weight-curve scale. Doubling per step: strength 1
    // down-weights a |d| = 4 pixel to ~1/3, strength 6 keeps it at
    // ~14/16.
    let s_scale = 8u32 << cfg.strength.min(ArnrConfig::MAX_STRENGTH);
    // One weight per possible |ref − center| difference — replaces the
    // per-pixel division in the accumulation loops below.
    let wt_table = weight_table(s_scale);

    // Per-16×16-block accumulation over the window.
    let mut acc_y = vec![0u32; w * h];
    let mut wsum_y = vec![0u32; w * h];
    let mut acc_u = vec![0u32; cw * ch];
    let mut wsum_u = vec![0u32; cw * ch];
    let mut acc_v = vec![0u32; cw * ch];
    let mut wsum_v = vec![0u32; cw * ch];

    // Seed with the center frame at full weight.
    accumulate_center(&out.y, &mut acc_y, &mut wsum_y);
    accumulate_center(&out.u, &mut acc_u, &mut wsum_u);
    accumulate_center(&out.v, &mut acc_v, &mut wsum_v);

    for (fi, f) in frames.iter().enumerate() {
        if fi == center {
            continue;
        }
        let mut by = 0usize;
        while by < h {
            let bh = (h - by).min(16);
            let mut bx = 0usize;
            while bx < w {
                let bw = (w - bx).min(16);
                let (mvx, mvy, sad) = refine_search(ctr, f, (bx, by, bw, bh));
                // Occlusion / scene-change guard: a block that still
                // mismatches after alignment is excluded outright.
                if sad / (bw * bh) as u32 > BLOCK_SAD_PER_PIXEL_CUTOFF {
                    bx += 16;
                    continue;
                }
                // Luma accumulation.
                accumulate_luma_block(
                    &out.y,
                    (f.y, f.y_stride),
                    (w, h),
                    (bx, by, bw, bh),
                    (mvx, mvy),
                    &wt_table,
                    &mut acc_y,
                    &mut wsum_y,
                );
                // Chroma accumulation — luma MV halved (round toward
                // zero), §18.2-style.
                let (cmvx, cmvy) = (mvx / 2, mvy / 2);
                let (cbx, cby) = (bx / 2, by / 2);
                let (cbw, cbh) = (bw.div_ceil(2), bh.div_ceil(2));
                for (ctr_plane, fp, acc, wsum) in [
                    (&out.u, f.u, &mut acc_u, &mut wsum_u),
                    (&out.v, f.v, &mut acc_v, &mut wsum_v),
                ] {
                    accumulate_chroma_block(
                        ctr_plane,
                        (fp, f.uv_stride),
                        (cw, ch),
                        (cbx, cby, cbw, cbh),
                        (cmvx, cmvy),
                        &wt_table,
                        acc,
                        wsum,
                    );
                }
                bx += 16;
            }
            by += 16;
        }
    }

    resolve_plane(&mut out.y, &acc_y, &wsum_y);
    resolve_plane(&mut out.u, &acc_u, &wsum_u);
    resolve_plane(&mut out.v, &acc_v, &wsum_v);
    Ok(out)
}

/// Copy a possibly-strided plane into a tightly-packed buffer.
fn copy_plane(plane: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h);
    for r in 0..h {
        out.extend_from_slice(&plane[r * stride..r * stride + w]);
    }
    out
}

/// Seed the accumulators with the center plane at full weight.
fn accumulate_center(center: &[u8], acc: &mut [u32], wsum: &mut [u32]) {
    for ((a, ws), &px) in acc.iter_mut().zip(wsum.iter_mut()).zip(center.iter()) {
        *a += W_MAX * px as u32;
        *ws += W_MAX;
    }
}

/// Rounded weighted mean, in place over the center-copy plane.
fn resolve_plane(plane: &mut [u8], acc: &[u32], wsum: &[u32]) {
    for ((px, &a), &ws) in plane.iter_mut().zip(acc.iter()).zip(wsum.iter()) {
        debug_assert!(ws >= W_MAX);
        *px = ((a + ws / 2) / ws) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 48;
    const H: usize = 48;

    /// Deterministic LCG noise in `-amp..=amp`.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 33
        }
        fn noise(&mut self, amp: i32) -> i32 {
            (self.next() % (2 * amp as u64 + 1)) as i32 - amp
        }
    }

    /// A clean textured scene.
    fn clean_scene() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; W * H];
        for r in 0..H {
            for c in 0..W {
                y[r * W + c] = (64 + ((r * 3 + c * 2) & 0x7f)) as u8;
            }
        }
        let (cw, ch) = (W / 2, H / 2);
        let u = vec![120u8; cw * ch];
        let v = vec![136u8; cw * ch];
        (y, u, v)
    }

    /// Add ±amp LCG noise to a plane (seeded per frame).
    fn noisy(plane: &[u8], seed: u64, amp: i32) -> Vec<u8> {
        let mut rng = Lcg(seed);
        plane
            .iter()
            .map(|&p| (p as i32 + rng.noise(amp)).clamp(0, 255) as u8)
            .collect()
    }

    fn mse(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sse: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as i64 - y as i64;
                (d * d) as u64
            })
            .sum();
        sse as f64 / a.len() as f64
    }

    #[test]
    fn strength_zero_and_single_frame_are_identity() {
        let (y, u, v) = clean_scene();
        let f = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
        let frames = [f];
        let out = build_arnr_altref(&frames, 0, &ArnrConfig::default()).unwrap();
        assert_eq!(out.y, y);
        assert_eq!(out.u, u);
        assert_eq!(out.v, v);

        let ny = noisy(&y, 7, 6);
        let f0 = I420Frame::packed(W as u32, H as u32, &ny, &u, &v);
        let f1 = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
        let frames = [f0, f1];
        let out = build_arnr_altref(&frames, 0, &ArnrConfig::new(0)).unwrap();
        assert_eq!(out.y, ny, "strength 0 must be a pass-through");
    }

    #[test]
    fn static_noisy_window_denoises_toward_the_clean_scene() {
        let (cy, cu, cv) = clean_scene();
        let noisy_planes: Vec<Vec<u8>> = (0..5).map(|i| noisy(&cy, 1000 + i, 6)).collect();
        let frames: Vec<I420Frame<'_>> = noisy_planes
            .iter()
            .map(|ny| I420Frame::packed(W as u32, H as u32, ny, &cu, &cv))
            .collect();
        let out = build_arnr_altref(&frames, 2, &ArnrConfig::default()).unwrap();
        let center_mse = mse(&noisy_planes[2], &cy);
        let filtered_mse = mse(&out.y, &cy);
        assert!(
            filtered_mse < center_mse / 2.0,
            "temporal blend must denoise: center {center_mse:.2} vs filtered {filtered_mse:.2}"
        );
    }

    #[test]
    fn motion_compensation_tracks_a_translating_scene() {
        // The window translates 4 px right per frame; each frame carries
        // independent noise. Without MC the blend of misaligned frames
        // would *add* error; with MC it must still denoise.
        let (cy, cu, cv) = clean_scene();
        let mut planes: Vec<Vec<u8>> = Vec::new();
        for i in 0..3i32 {
            let dx = (i - 1) * 4; // -4, 0, +4
            let mut y = vec![0u8; W * H];
            for r in 0..H {
                for c in 0..W {
                    let sc = (c as i32 - dx).clamp(0, W as i32 - 1) as usize;
                    y[r * W + c] = cy[r * W + sc];
                }
            }
            planes.push(noisy(&y, 2000 + i as u64, 6));
        }
        let frames: Vec<I420Frame<'_>> = planes
            .iter()
            .map(|p| I420Frame::packed(W as u32, H as u32, p, &cu, &cv))
            .collect();
        let out = build_arnr_altref(&frames, 1, &ArnrConfig::default()).unwrap();
        let center_mse = mse(&planes[1], &cy);
        let filtered_mse = mse(&out.y, &cy);
        assert!(
            filtered_mse < center_mse,
            "MC blend must denoise a translating scene: center {center_mse:.2} vs filtered {filtered_mse:.2}"
        );
    }

    #[test]
    fn scene_change_neighbour_is_excluded() {
        // Frame 1 is unrelated content: the SAD cutoff must exclude every
        // block, leaving the output exactly the center frame.
        let (cy, cu, cv) = clean_scene();
        let alien: Vec<u8> = cy.iter().map(|&p| 255 - p).collect();
        let f0 = I420Frame::packed(W as u32, H as u32, &cy, &cu, &cv);
        let f1 = I420Frame::packed(W as u32, H as u32, &alien, &cu, &cv);
        let frames = [f0, f1];
        let out = build_arnr_altref(&frames, 0, &ArnrConfig::new(6)).unwrap();
        assert_eq!(out.y, cy, "an unmatchable neighbour must not bleed in");
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let (cy, cu, cv) = clean_scene();
        let f0 = I420Frame::packed(W as u32, H as u32, &cy, &cu, &cv);
        let small_y = vec![0u8; 32 * 32];
        let small_c = vec![128u8; 16 * 16];
        let f1 = I420Frame::packed(32, 32, &small_y, &small_c, &small_c);
        let err = build_arnr_altref(&[f0, f1], 0, &ArnrConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            EncodeError::ReferenceDimensionsMismatch { .. }
        ));
    }

    #[test]
    fn config_clamps_strength() {
        assert_eq!(ArnrConfig::new(99).strength, ArnrConfig::MAX_STRENGTH);
        assert_eq!(ArnrConfig::new(2).strength, 2);
    }

    // =====================================================================
    // Fast-path equivalence pins (round 409). Each optimized path must be
    // pixel-for-pixel the straightforward per-pixel clamped form.
    // =====================================================================

    /// The weight table is a pure precomputation of `pixel_weight`.
    #[test]
    fn weight_table_matches_pixel_weight() {
        for strength in 0..=ArnrConfig::MAX_STRENGTH {
            let s_scale = 8u32 << strength;
            let table = weight_table(s_scale);
            for d in -255i32..=255 {
                assert_eq!(
                    table[d.unsigned_abs() as usize],
                    pixel_weight(d, s_scale),
                    "strength {strength} d {d}"
                );
            }
        }
    }

    /// A pair of textured, noisy frames for the block-level pins.
    fn stress_pair() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let (cy, cu, _cv) = clean_scene();
        let ry = noisy(&cy, 4242, 20);
        let ru = noisy(&cu, 2424, 12);
        (cy, ry, cu, ru)
    }

    /// `block_sad` (fast dispatch, no early exit) equals the clamped
    /// generic form on interior blocks, edge blocks, and every MV the
    /// refinement search can probe.
    #[test]
    fn block_sad_fast_matches_generic() {
        let (cy, ry, cu, cv_) = stress_pair();
        let src = I420Frame::packed(W as u32, H as u32, &cy, &cu, &cv_);
        let refp = I420Frame::packed(W as u32, H as u32, &ry, &cu, &cv_);
        // Interior, corner, and partial (right/bottom overhang-free)
        // blocks; W = H = 48 so blocks at 32 touch the frame edge.
        for &blk in &[
            (16usize, 16usize, 16usize, 16usize),
            (0, 0, 16, 16),
            (32, 32, 16, 16),
            (32, 0, 16, 16),
            (0, 32, 16, 16),
        ] {
            for mvy in -16i32..=16 {
                for mvx in -16i32..=16 {
                    let fast = block_sad(&src, &refp, blk, (mvx, mvy), u32::MAX);
                    let generic = block_sad_generic(&src, &refp, blk, (mvx, mvy), u32::MAX);
                    assert_eq!(fast, generic, "blk {blk:?} mv ({mvx},{mvy})");
                }
            }
        }
    }

    /// The whole refinement descent — with the monotone early exit and
    /// the in-bounds fast path — returns exactly the `(mv, sad)` the
    /// exhaustive no-early-exit generic descent returns.
    #[test]
    fn refine_search_matches_no_early_exit_reference() {
        /// The original descent shape: generic SAD, no early exit.
        fn refine_search_reference(
            src: &I420Frame<'_>,
            refp: &I420Frame<'_>,
            blk: (usize, usize, usize, usize),
        ) -> (i32, i32, u32) {
            let mut best = (0i32, 0i32);
            let mut best_sad = block_sad_generic(src, refp, blk, (0, 0), u32::MAX);
            let mut step = 8i32;
            while step >= 1 {
                let center = best;
                for (dx, dy) in [
                    (-step, -step),
                    (0, -step),
                    (step, -step),
                    (-step, 0),
                    (step, 0),
                    (-step, step),
                    (0, step),
                    (step, step),
                ] {
                    let cand = (center.0 + dx, center.1 + dy);
                    if cand.0.abs() > 15 || cand.1.abs() > 15 {
                        continue;
                    }
                    let sad = block_sad_generic(src, refp, blk, cand, u32::MAX);
                    if sad < best_sad {
                        best_sad = sad;
                        best = cand;
                    }
                }
                step /= 2;
            }
            (best.0, best.1, best_sad)
        }

        let (cy, cu, cv_) = clean_scene();
        // Reference scenes: pure noise, a genuine +4 px translation, and
        // an unrelated (inverted) frame — bracketing convergent,
        // walking, and hopeless descents.
        let noise_only = noisy(&cy, 99, 8);
        let mut translated = vec![0u8; W * H];
        for r in 0..H {
            for c in 0..W {
                let sc = (c as i32 - 4).clamp(0, W as i32 - 1) as usize;
                translated[r * W + c] = cy[r * W + sc];
            }
        }
        let translated = noisy(&translated, 7331, 5);
        let alien: Vec<u8> = cy.iter().map(|&p| 255 - p).collect();

        let src = I420Frame::packed(W as u32, H as u32, &cy, &cu, &cv_);
        for ref_y in [&noise_only, &translated, &alien] {
            let refp = I420Frame::packed(W as u32, H as u32, ref_y, &cu, &cv_);
            let mut by = 0usize;
            while by < H {
                let bh = (H - by).min(16);
                let mut bx = 0usize;
                while bx < W {
                    let bw = (W - bx).min(16);
                    let blk = (bx, by, bw, bh);
                    assert_eq!(
                        refine_search(&src, &refp, blk),
                        refine_search_reference(&src, &refp, blk),
                        "blk {blk:?}"
                    );
                    bx += 16;
                }
                by += 16;
            }
        }
    }

    /// `accumulate_luma_block` (in-bounds fast path + generic edge path)
    /// equals the original per-pixel `pel` + `pixel_weight` form on
    /// every probed MV, in-bounds and edge-crossing alike.
    #[test]
    fn accumulate_luma_block_fast_matches_generic() {
        /// The original accumulation loop, verbatim.
        #[allow(clippy::too_many_arguments)]
        fn reference(
            ctr_y: &[u8],
            f_y: &[u8],
            f_stride: usize,
            (w, h): (usize, usize),
            (bx, by, bw, bh): (usize, usize, usize, usize),
            (mvx, mvy): (i32, i32),
            s_scale: u32,
            acc: &mut [u32],
            wsum: &mut [u32],
        ) {
            for r in 0..bh {
                let yy = by + r;
                for c in 0..bw {
                    let xx = bx + c;
                    let ctr_px = ctr_y[yy * w + xx] as i32;
                    let ref_px = pel(
                        f_y,
                        f_stride,
                        w as i32,
                        h as i32,
                        xx as i32 + mvx,
                        yy as i32 + mvy,
                    ) as i32;
                    let wt = pixel_weight(ref_px - ctr_px, s_scale);
                    acc[yy * w + xx] += wt * ref_px as u32;
                    wsum[yy * w + xx] += wt;
                }
            }
        }

        let (cy, ry, _cu, _ru) = stress_pair();
        let s_scale = 8u32 << 3;
        let table = weight_table(s_scale);
        for &blk in &[
            (16usize, 16usize, 16usize, 16usize),
            (0, 0, 16, 16),
            (32, 32, 16, 16),
        ] {
            for mvy in [-15i32, -7, -1, 0, 1, 7, 15] {
                for mvx in [-15i32, -7, -1, 0, 1, 7, 15] {
                    let mut acc_a = vec![3u32; W * H];
                    let mut wsum_a = vec![5u32; W * H];
                    let mut acc_b = acc_a.clone();
                    let mut wsum_b = wsum_a.clone();
                    accumulate_luma_block(
                        &cy,
                        (&ry, W),
                        (W, H),
                        blk,
                        (mvx, mvy),
                        &table,
                        &mut acc_a,
                        &mut wsum_a,
                    );
                    reference(
                        &cy,
                        &ry,
                        W,
                        (W, H),
                        blk,
                        (mvx, mvy),
                        s_scale,
                        &mut acc_b,
                        &mut wsum_b,
                    );
                    assert_eq!(acc_a, acc_b, "acc blk {blk:?} mv ({mvx},{mvy})");
                    assert_eq!(wsum_a, wsum_b, "wsum blk {blk:?} mv ({mvx},{mvy})");
                }
            }
        }
    }

    /// `accumulate_chroma_block` equals the original clamped form —
    /// including the deliberate double-accumulation into clamped edge
    /// pixels on an odd-dimension plane's overhanging last block.
    #[test]
    fn accumulate_chroma_block_fast_matches_generic() {
        /// The original chroma accumulation loop, verbatim.
        #[allow(clippy::too_many_arguments)]
        fn reference(
            ctr_plane: &[u8],
            f_plane: &[u8],
            f_stride: usize,
            (cw, ch): (usize, usize),
            (cbx, cby, cbw, cbh): (usize, usize, usize, usize),
            (cmvx, cmvy): (i32, i32),
            s_scale: u32,
            acc: &mut [u32],
            wsum: &mut [u32],
        ) {
            for r in 0..cbh {
                let yy = (cby + r).min(ch - 1);
                for c in 0..cbw {
                    let xx = (cbx + c).min(cw - 1);
                    let ctr_px = ctr_plane[yy * cw + xx] as i32;
                    let ref_px = pel(
                        f_plane,
                        f_stride,
                        cw as i32,
                        ch as i32,
                        xx as i32 + cmvx,
                        yy as i32 + cmvy,
                    ) as i32;
                    let wt = pixel_weight(ref_px - ctr_px, s_scale);
                    acc[yy * cw + xx] += wt * ref_px as u32;
                    wsum[yy * cw + xx] += wt;
                }
            }
        }

        // A deliberately odd chroma geometry (25×23) so `div_ceil`
        // blocks overhang on the right and bottom.
        let (cw, ch) = (25usize, 23usize);
        let mut ctr = vec![0u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                ctr[r * cw + c] = (100 + ((r * 5 + c * 3) & 0x3f)) as u8;
            }
        }
        let refp = noisy(&ctr, 555, 10);
        let s_scale = 8u32 << 3;
        let table = weight_table(s_scale);
        // 8×8 chroma blocks over the odd plane, all halved MVs the luma
        // search can produce.
        for &blk in &[
            (0usize, 0usize, 8usize, 8usize),
            (8, 8, 8, 8),
            (24, 16, 8, 8), // right + bottom overhang (cbx = 24 on cw = 25)
            (16, 16, 8, 8),
        ] {
            for cmvy in [-7i32, -3, 0, 3, 7] {
                for cmvx in [-7i32, -3, 0, 3, 7] {
                    let mut acc_a = vec![1u32; cw * ch];
                    let mut wsum_a = vec![2u32; cw * ch];
                    let mut acc_b = acc_a.clone();
                    let mut wsum_b = wsum_a.clone();
                    accumulate_chroma_block(
                        &ctr,
                        (&refp, cw),
                        (cw, ch),
                        blk,
                        (cmvx, cmvy),
                        &table,
                        &mut acc_a,
                        &mut wsum_a,
                    );
                    reference(
                        &ctr,
                        &refp,
                        cw,
                        (cw, ch),
                        blk,
                        (cmvx, cmvy),
                        s_scale,
                        &mut acc_b,
                        &mut wsum_b,
                    );
                    assert_eq!(acc_a, acc_b, "acc blk {blk:?} cmv ({cmvx},{cmvy})");
                    assert_eq!(wsum_a, wsum_b, "wsum blk {blk:?} cmv ({cmvx},{cmvy})");
                }
            }
        }
    }
}
