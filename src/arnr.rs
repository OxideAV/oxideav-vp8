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
/// `(mvx, mvy)`.
fn block_sad(
    src: &I420Frame<'_>,
    refp: &I420Frame<'_>,
    (bx, by, bw, bh): (usize, usize, usize, usize),
    (mvx, mvy): (i32, i32),
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
    let mut best_sad = block_sad(src, refp, blk, (0, 0));
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
            let sad = block_sad(src, refp, blk, cand);
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
                for r in 0..bh {
                    let yy = by + r;
                    for c in 0..bw {
                        let xx = bx + c;
                        let ctr_px = out.y[yy * w + xx] as i32;
                        let ref_px = pel(
                            f.y,
                            f.y_stride,
                            w as i32,
                            h as i32,
                            xx as i32 + mvx,
                            yy as i32 + mvy,
                        ) as i32;
                        let wt = pixel_weight(ref_px - ctr_px, s_scale);
                        acc_y[yy * w + xx] += wt * ref_px as u32;
                        wsum_y[yy * w + xx] += wt;
                    }
                }
                // Chroma accumulation — luma MV halved (round toward
                // zero), §18.2-style.
                let (cmvx, cmvy) = (mvx / 2, mvy / 2);
                let (cbx, cby) = (bx / 2, by / 2);
                let (cbw, cbh) = (bw.div_ceil(2), bh.div_ceil(2));
                for r in 0..cbh {
                    let yy = (cby + r).min(ch - 1);
                    for c in 0..cbw {
                        let xx = (cbx + c).min(cw - 1);
                        for (ctr_plane, fp, acc, wsum) in [
                            (&out.u, f.u, &mut acc_u, &mut wsum_u),
                            (&out.v, f.v, &mut acc_v, &mut wsum_v),
                        ] {
                            let ctr_px = ctr_plane[yy * cw + xx] as i32;
                            let ref_px = pel(
                                fp,
                                f.uv_stride,
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
}
