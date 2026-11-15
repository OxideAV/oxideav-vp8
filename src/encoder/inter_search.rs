//! The reduced-effort macroblock search for inter frames.
//!
//! RFC 6386 specifies decoding only; how an encoder finds its motion
//! vectors and modes is its own business. The exhaustive search
//! ([`EncodeEffort::Thorough`]) runs a full motion search, every
//! `SPLITMV` partition and (optionally) sixteen intra mode pairs against
//! every reference, carries each reference's winner through transform,
//! trellis quantisation and reconstruction, and ranks on prediction
//! error. The reduced levels spend their time differently:
//!
//! * **Motion search.** Each reference is searched on the whole-pixel
//!   grid from the best of a few predicted vectors (zero, the §16.3
//!   census vectors, the neighbouring macroblocks' vectors) by pattern
//!   descent, with every sum of absolute differences abandoned as soon
//!   as it cannot beat the best so far. A search stops at once when the
//!   error is already below what the quantiser would keep.
//! * **Sub-pixel refinement** runs only around the best whole-pixel
//!   vector of the best reference (the best two at `Balanced`). The
//!   eight half-pixel neighbours are read out of three §18.3-filtered
//!   planes computed once, so they cost three filter passes rather than
//!   eight; quarter-pixel neighbours are probed around the best
//!   half-pixel position.
//! * **Mode decision** between the best inter prediction, `SPLITMV` and
//!   the intra modes is made on the transform-domain cost model of
//!   `effort.rs` (weighted squared rounding error plus tabled token
//!   bits), and the more expensive candidates are only built when the
//!   inter prediction error is large against the quantiser step. Only
//!   the winner is quantised for real.
//! * A macroblock whose coefficients cost more to code than the
//!   distortion they remove is coded as a §11.1 skip.
//!
//! Every decision is a function of the source, the references and the
//! neighbouring macroblocks' decisions, so the result does not depend on
//! the thread budget; the reconstruction is built with the decoder's own
//! inverse transforms on the decoder's own prediction.

use super::effort::{
    BlockModel, DCT_WEIGHTS, EncodeEffort, WEIGHT_SCALE, Y2_WEIGHTS, model_block_parts,
    residual_dct,
};
use super::inter::{InterSearch, MbDecision, MbPlace, RefPicture};
use super::{
    BlockType, IntraUvMode, IntraYMode, MbCoeffs, MbPixels, SplitMvCandidate, TrellisBlockSpec,
    TrellisRateTables, bool_bits, forward_wht_4x4, if_uv_mode_tree_bits, if_ymode_tree_bits,
    predict_uv8x8, predict_y16x16, ref_frame_tree_bits, score_split_partition,
    trellis_quantize_block,
};
use crate::dequant::MbDequantFactors;
use crate::intra_predict::DEFAULT_ABOVE_PIXEL;
use crate::inverse_transform::{inverse_dct_4x4_add_into, inverse_wht_4x4};
use crate::motion_comp::{RefFrame, predict_inter_mb, predict_split_mv, sixtap_run};
use crate::motion_search::{LumaRef, mb_luma_sad_at_mv};
use crate::motion_vector::{Mv, MvContexts, mv_component_bits};
use crate::near_mv::{InterMode, MvClampRect, MvPartition, clamp_mv, find_near_mvs, mv_ref_probs};
use crate::reconstruct::ReconstructedMb;

/// How the reduced levels spend their search.
#[derive(Clone, Copy)]
pub(super) struct SearchShape {
    /// Rounds of the one-pixel pattern descent per reference.
    descent_rounds: u32,
    /// Start with a two-pixel pattern when the seed error is large.
    wide_pattern: bool,
    /// Probe the four diagonal half-pixel neighbours (else only the
    /// horizontal and vertical ones).
    half_diagonals: bool,
    /// Probe quarter-pixel neighbours around the best half-pixel
    /// position.
    quarter: bool,
    /// Search the other references from their predicted vectors (else
    /// only test the predicted vectors themselves).
    search_other_refs: bool,
    /// Also refine the runner-up reference to sub-pixel precision when
    /// its whole-pixel cost is close to the leader's.
    refine_runner_up: bool,
    /// `SPLITMV` partitions to try on a poorly predicted macroblock.
    split_partitions: &'static [MvPartition],
    /// Compare all four §12.2 luma / chroma intra modes with the cost
    /// model (else only the mode with the smallest prediction error).
    intra_all_modes: bool,
    /// Inter prediction error, in squared pixel-domain quantiser steps
    /// per pixel, above which intra candidates are built.
    intra_gate: u64,
    /// Likewise for `SPLITMV`.
    split_gate: u64,
    /// Trellis-quantise the winner (else plain rounding).
    trellis: bool,
}

impl EncodeEffort {
    /// The reduced inter search of this level (`None` for
    /// [`EncodeEffort::Thorough`], the exhaustive search).
    pub(super) fn inter_shape(self) -> Option<SearchShape> {
        match self {
            EncodeEffort::Fastest => Some(SearchShape {
                descent_rounds: 4,
                wide_pattern: false,
                half_diagonals: true,
                quarter: false,
                search_other_refs: false,
                refine_runner_up: false,
                split_partitions: &[],
                intra_all_modes: false,
                intra_gate: 16,
                split_gate: u64::MAX,
                trellis: false,
            }),
            EncodeEffort::Fast => Some(SearchShape {
                descent_rounds: 8,
                wide_pattern: true,
                half_diagonals: true,
                quarter: true,
                search_other_refs: true,
                refine_runner_up: false,
                split_partitions: &[],
                intra_all_modes: false,
                intra_gate: 8,
                split_gate: u64::MAX,
                trellis: true,
            }),
            EncodeEffort::Balanced => Some(SearchShape {
                descent_rounds: 16,
                wide_pattern: true,
                half_diagonals: true,
                quarter: true,
                search_other_refs: true,
                refine_runner_up: true,
                split_partitions: &[
                    MvPartition::Quarters,
                    MvPartition::TopBottom,
                    MvPartition::LeftRight,
                ],
                intra_all_modes: true,
                intra_gate: 4,
                split_gate: 4,
                trellis: true,
            }),
            EncodeEffort::Thorough => None,
        }
    }
}

/// §17.2 bit cost of every motion-vector differential component under
/// one frame's MV contexts, tabled once per frame.
pub(super) struct MvCostTable {
    /// `[component][diff + 1023]`, component 0 = row, 1 = column.
    bits: [Vec<f32>; 2],
}

impl MvCostTable {
    pub(super) fn new(contexts: &MvContexts) -> Self {
        let table = |ctx: usize| -> Vec<f32> {
            (-1023..=1023)
                .map(|d| mv_component_bits(&contexts[ctx], d) as f32)
                .collect()
        };
        MvCostTable {
            bits: [table(0), table(1)],
        }
    }

    /// Bits of coding `mv` as a differential against `base`; `None` when
    /// the differential is outside the §17.1 range.
    #[inline]
    fn bits(&self, mv: Mv, base: Mv) -> Option<f64> {
        let dr = mv.row as i32 - base.row as i32;
        let dc = mv.col as i32 - base.col as i32;
        if !(-1023..=1023).contains(&dr) || !(-1023..=1023).contains(&dc) {
            return None;
        }
        Some((self.bits[0][(dr + 1023) as usize] + self.bits[1][(dc + 1023) as usize]) as f64)
    }
}

/// Sum of absolute differences of two 16-pixel rows.
#[inline(always)]
fn row_sad16(a: &[u8], b: &[u8]) -> u32 {
    let a: &[u8; 16] = a[..16].try_into().expect("16 pixels");
    let b: &[u8; 16] = b[..16].try_into().expect("16 pixels");
    let mut acc = 0u32;
    for i in 0..16 {
        acc += a[i].abs_diff(b[i]) as u32;
    }
    acc
}

/// SAD of the source macroblock against the 16×16 window of `plane`
/// (a `stride`-wide buffer) whose top-left sample is at `offset`.
#[inline]
fn window_sad(src: &[u8; 256], plane: &[u8], stride: usize, offset: usize) -> u32 {
    let mut acc = 0u32;
    for r in 0..16 {
        acc += row_sad16(&src[r * 16..], &plane[offset + r * stride..]);
    }
    acc
}

/// SAD of the source macroblock against the reference at whole-pixel
/// position `(x, y)` (top-left sample, may lie outside the picture —
/// coordinates clamp to the edge as §20.14 border extension does).
/// Gives up and returns a value `>= limit` once the sum reaches `limit`.
#[inline]
fn whole_sad(reference: &LumaRef, x: i32, y: i32, src: &[u8; 256], limit: u32) -> u32 {
    let (w, h) = (reference.width as i32, reference.height as i32);
    let mut acc = 0u32;
    if x >= 0 && y >= 0 && x + 16 <= w && y + 16 <= h {
        let base = y as usize * reference.stride + x as usize;
        for band in 0..4 {
            for r in band * 4..band * 4 + 4 {
                acc += row_sad16(
                    &src[r * 16..],
                    &reference.plane[base + r * reference.stride..],
                );
            }
            if acc >= limit {
                return acc;
            }
        }
        return acc;
    }
    for r in 0..16 {
        let sy = (y + r as i32).clamp(0, h - 1) as usize;
        let row = &reference.plane[sy * reference.stride..];
        for c in 0..16 {
            let sx = (x + c as i32).clamp(0, w - 1) as usize;
            acc += src[r * 16 + c].abs_diff(row[sx]) as u32;
        }
        if acc >= limit {
            return acc;
        }
    }
    acc
}

/// Side of the square window the half-pixel planes are computed from:
/// the macroblock plus one more position on each side, plus the
/// six-tap support (two samples before, three after).
const WIN: usize = 22;
/// Half-pixel planes hold 17 positions per axis (from half a pixel
/// before the macroblock origin to half a pixel after its last sample).
const HP: usize = 17;

/// The reference samples around whole-pixel position `(x, y)`: rows
/// `y − 3 ..= y + 18`, columns `x − 3 ..= x + 18`, edge-clamped.
fn fetch_window(reference: &LumaRef, x: i32, y: i32) -> [u8; WIN * WIN] {
    let (w, h) = (reference.width as i32, reference.height as i32);
    let mut out = [0u8; WIN * WIN];
    let (x0, y0) = (x - 3, y - 3);
    if x0 >= 0 && y0 >= 0 && x0 + WIN as i32 <= w && y0 + WIN as i32 <= h {
        let base = y0 as usize * reference.stride + x0 as usize;
        for r in 0..WIN {
            let s = base + r * reference.stride;
            out[r * WIN..r * WIN + WIN].copy_from_slice(&reference.plane[s..s + WIN]);
        }
        return out;
    }
    for r in 0..WIN {
        let sy = (y0 + r as i32).clamp(0, h - 1) as usize;
        let row = &reference.plane[sy * reference.stride..];
        for c in 0..WIN {
            let sx = (x0 + c as i32).clamp(0, w - 1) as usize;
            out[r * WIN + c] = row[sx];
        }
    }
    out
}

/// The three half-pixel planes around a whole-pixel position `P`:
/// `h[r][c]` is the §18.3 prediction sample half a pixel right of
/// `P + (c − 1, r − 3)` (22 rows, so the diagonal plane can be filtered
/// from it), `v[r][c]` the one half a pixel below `P + (c, r − 1)`, and
/// `d[r][c]` the one half a pixel right of and below `P + (c − 1, r − 1)`.
/// They are exactly the samples the two-pass §20.14 filter produces for
/// the corresponding half-pixel vectors.
struct HalfPlanes {
    h: [u8; WIN * HP],
    v: [u8; HP * 16],
    d: [u8; HP * HP],
}

impl HalfPlanes {
    fn new(window: &[u8; WIN * WIN], taps: &[i32; 6], diagonals: bool) -> Self {
        let mut planes = HalfPlanes {
            h: [0u8; WIN * HP],
            v: [0u8; HP * 16],
            d: [0u8; HP * HP],
        };
        // Horizontal: output column c sits between window columns c + 2
        // and c + 3, so its support starts at window column c.
        let h_rows = if diagonals { 0..WIN } else { 3..19 };
        for r in h_rows {
            let row: [u8; HP] = sixtap_run(&window[r * WIN..], 1, taps);
            planes.h[r * HP..r * HP + HP].copy_from_slice(&row);
        }
        // Vertical: output row r sits between window rows r + 2 and
        // r + 3; columns are window columns 3..19.
        for r in 0..HP {
            let row: [u8; 16] = sixtap_run(&window[r * WIN + 3..], WIN, taps);
            planes.v[r * 16..r * 16 + 16].copy_from_slice(&row);
        }
        if diagonals {
            for r in 0..HP {
                let row: [u8; HP] = sixtap_run(&planes.h[r * HP..], HP, taps);
                planes.d[r * HP..r * HP + HP].copy_from_slice(&row);
            }
        }
        planes
    }

    /// SAD of the source against the prediction for the half-pixel
    /// offset `(dx, dy)` (each −1, 0 or +1 half pixels, not both 0).
    fn sad(&self, src: &[u8; 256], dx: i32, dy: i32) -> u32 {
        // An offset of −½ reads plane position 0, +½ position 1.
        let cx = (dx > 0) as usize;
        let cy = (dy > 0) as usize;
        if dy == 0 {
            window_sad(src, &self.h, HP, 3 * HP + cx)
        } else if dx == 0 {
            window_sad(src, &self.v, 16, cy * 16)
        } else {
            window_sad(src, &self.d, HP, cy * HP + cx)
        }
    }
}

/// A searched vector on one reference.
#[derive(Clone, Copy)]
struct Found {
    mv: Mv,
    sad: u32,
    /// `sad + lambda_sad · (differential bits)`.
    cost: f64,
}

/// What the §16.3 census gives one reference at this macroblock.
struct Census {
    /// The predictor a `NEWMV` differential is coded against.
    best: Mv,
    nearest: Mv,
    near: Mv,
    /// §16.2 mode-tree bits of ZERO / NEAREST / NEAR / NEW / SPLIT.
    mode_bits: [f64; 5],
}

/// Per-macroblock search context for one reference.
struct RefSearch<'a> {
    picture: &'a RefPicture<'a>,
    luma: LumaRef<'a>,
    census: Census,
    /// Whole-pixel search range (macroblock displacement, pixels).
    range_x: (i32, i32),
    range_y: (i32, i32),
}

/// Whole-pixel search of one reference: the best of the seed vectors,
/// then pattern descent.
#[allow(clippy::too_many_arguments)]
fn whole_pixel_search(
    r: &RefSearch,
    src: &[u8; 256],
    origin: (i32, i32),
    seeds: &[Mv],
    costs: &MvCostTable,
    lambda_sad: f64,
    good_enough: u32,
    shape: &SearchShape,
    descend: bool,
) -> Found {
    let eval = |px: i32, py: i32, best_cost: f64| -> Option<Found> {
        let mv = Mv {
            row: (py * 4) as i16,
            col: (px * 4) as i16,
        };
        let rate = lambda_sad * costs.bits(mv, r.census.best)?;
        if rate >= best_cost {
            return None;
        }
        let limit = (best_cost - rate).min(u32::MAX as f64) as u32;
        let sad = whole_sad(
            &r.luma,
            origin.0 + px,
            origin.1 + py,
            src,
            limit.saturating_add(1),
        );
        let cost = sad as f64 + rate;
        (cost < best_cost).then_some(Found { mv, sad, cost })
    };
    let clamp_px = |mv: Mv| -> (i32, i32) {
        // Round to the nearest whole pixel, inside the search range.
        let px = ((mv.col as i32 + 2) >> 2).clamp(r.range_x.0, r.range_x.1);
        let py = ((mv.row as i32 + 2) >> 2).clamp(r.range_y.0, r.range_y.1);
        (px, py)
    };

    let mut best = Found {
        mv: Mv::default(),
        sad: u32::MAX,
        cost: f64::INFINITY,
    };
    let mut tried = [(i32::MIN, i32::MIN); 8];
    let mut tried_len = 0usize;
    for &seed in seeds {
        let p = clamp_px(seed);
        if tried[..tried_len].contains(&p) {
            continue;
        }
        if tried_len < tried.len() {
            tried[tried_len] = p;
            tried_len += 1;
        }
        if let Some(found) = eval(p.0, p.1, best.cost) {
            best = found;
        }
    }
    if !descend || best.sad <= good_enough {
        return best;
    }

    let in_range = |px: i32, py: i32| {
        px >= r.range_x.0 && px <= r.range_x.1 && py >= r.range_y.0 && py <= r.range_y.1
    };
    // Two-pixel pattern first when the seed is far from good.
    const WIDE: [(i32, i32); 8] = [
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];
    const NARROW: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    if shape.wide_pattern && best.sad > 4 * good_enough {
        for _ in 0..shape.descent_rounds {
            let (cx, cy) = (best.mv.col as i32 / 4, best.mv.row as i32 / 4);
            let mut improved = false;
            for (dx, dy) in WIDE {
                let (px, py) = (cx + dx, cy + dy);
                if in_range(px, py)
                    && let Some(found) = eval(px, py, best.cost)
                {
                    best = found;
                    improved = true;
                }
            }
            if !improved || best.sad <= good_enough {
                break;
            }
        }
    }
    for _ in 0..shape.descent_rounds {
        if best.sad <= good_enough {
            break;
        }
        let (cx, cy) = (best.mv.col as i32 / 4, best.mv.row as i32 / 4);
        let mut improved = false;
        for (dx, dy) in NARROW {
            let (px, py) = (cx + dx, cy + dy);
            if in_range(px, py)
                && let Some(found) = eval(px, py, best.cost)
            {
                best = found;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    best
}

/// Refine a whole-pixel result to half- and quarter-pixel precision.
#[allow(clippy::too_many_arguments)]
fn sub_pixel_refine(
    r: &RefSearch,
    src: &[u8; 256],
    origin: (i32, i32),
    place: &MbPlace,
    start: Found,
    costs: &MvCostTable,
    lambda_sad: f64,
    filters: &[[i32; 6]; 8],
    shape: &SearchShape,
) -> Found {
    let (px, py) = (start.mv.col as i32 / 4, start.mv.row as i32 / 4);
    let window = fetch_window(&r.luma, origin.0 + px, origin.1 + py);
    let planes = HalfPlanes::new(&window, &filters[4], shape.half_diagonals);
    let mut best = start;
    let consider = |mv: Mv, sad: u32, best: &mut Found| {
        if let Some(bits) = costs.bits(mv, r.census.best) {
            let cost = sad as f64 + lambda_sad * bits;
            if cost < best.cost {
                *best = Found { mv, sad, cost };
            }
        }
    };
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if (dx == 0 && dy == 0) || (!shape.half_diagonals && dx != 0 && dy != 0) {
                continue;
            }
            let mv = Mv {
                row: (start.mv.row as i32 + 2 * dy) as i16,
                col: (start.mv.col as i32 + 2 * dx) as i16,
            };
            if mv.row.unsigned_abs() > 1020 || mv.col.unsigned_abs() > 1020 {
                continue;
            }
            consider(mv, planes.sad(src, dx, dy), &mut best);
        }
    }
    if shape.quarter {
        let centre = best.mv;
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let mv = Mv {
                row: (centre.row as i32 + dy) as i16,
                col: (centre.col as i32 + dx) as i16,
            };
            if mv.row.unsigned_abs() > 1022 || mv.col.unsigned_abs() > 1022 {
                continue;
            }
            let sad = mb_luma_sad_at_mv(r.luma, place.mb_col, place.mb_row, src, mv);
            consider(mv, sad, &mut best);
        }
    }
    best
}

/// Transform-domain model of coding one macroblock against a prediction.
struct MbModel {
    /// Transform coefficients, raster order within each block. With a
    /// Y2 block the luma DCs are zeroed (they live in `y2`).
    dct: MbCoeffs,
    /// Plain-rounded levels of `dct`.
    levels: MbCoeffs,
    /// Blocks with a non-zero rounded level: bits 0..=15 luma,
    /// 16..=19 U, 20..=23 V, 24 Y2.
    nonzero: u32,
    /// Weighted squared rounding error (`1/4096` units).
    dist: u64,
    /// The same with every level zero.
    dist_zero: u64,
    /// Token bits of the rounded levels.
    rate: f64,
}

impl MbModel {
    /// `D + lambda·R` of the cheaper of coding the rounded levels and
    /// skipping the residual, and whether skipping is the cheaper.
    fn cost(&self, lambda: f64) -> (f64, bool) {
        let skip = self.dist_zero as f64 / WEIGHT_SCALE;
        if self.nonzero == 0 {
            return (skip, true);
        }
        let coded = self.dist as f64 / WEIGHT_SCALE + lambda * self.rate;
        if skip <= coded {
            (skip, true)
        } else {
            (coded, false)
        }
    }
}

fn add_block(model: &mut MbModel, bit: u32, m: &BlockModel) {
    model.dist += m.dist;
    model.dist_zero += m.dist_zero;
    model.rate += m.rate;
    model.nonzero |= (m.nonzero as u32) << bit;
}

/// Model the macroblock `src` predicted by `pred`; `with_y2` selects the
/// whole-block layout (luma DCs through the Y2 block) or the `SPLITMV`
/// one (every luma block carries its own DC).
fn model_mb(
    src: &MbPixels,
    pred: &ReconstructedMb,
    factors: &MbDequantFactors,
    tables: &TrellisRateTables,
    with_y2: bool,
) -> MbModel {
    let dct_weights = &*DCT_WEIGHTS;
    let mut model = MbModel {
        dct: MbCoeffs::default(),
        levels: MbCoeffs::default(),
        nonzero: 0,
        dist: 0,
        dist_zero: 0,
        rate: 0.0,
    };
    let luma_type = if with_y2 {
        BlockType::YAfterY2
    } else {
        BlockType::YNoY2
    };
    let mut dcs = [0i16; 16];
    let mut above = [false; 4];
    let mut left = [false; 4];
    for (i, left_nonzero) in left.iter_mut().enumerate() {
        for (j, above_nonzero) in above.iter_mut().enumerate() {
            let idx = i * 4 + j;
            let mut coeffs = residual_dct(&src.y, &pred.y, 16, i * 4, j * 4);
            if with_y2 {
                dcs[idx] = coeffs[0];
                coeffs[0] = 0;
            }
            model.dct.y[idx] = coeffs;
            let m = model_block_parts(
                &mut coeffs,
                TrellisBlockSpec {
                    dc: factors.y1_dc,
                    ac: factors.y1_ac,
                    first_coeff: luma_type.first_coeff(),
                    entry_ctx3: *above_nonzero as usize + *left_nonzero as usize,
                    plane: luma_type.plane_index(),
                },
                dct_weights,
                tables,
            );
            model.levels.y[idx] = coeffs;
            *above_nonzero = m.nonzero;
            *left_nonzero = m.nonzero;
            add_block(&mut model, idx as u32, &m);
        }
    }
    if with_y2 {
        let mut y2 = [0i16; 16];
        forward_wht_4x4(&dcs, &mut y2);
        model.dct.y2 = y2;
        let m = model_block_parts(
            &mut y2,
            TrellisBlockSpec {
                dc: factors.y2_dc,
                ac: factors.y2_ac,
                first_coeff: BlockType::Y2.first_coeff(),
                entry_ctx3: 0,
                plane: BlockType::Y2.plane_index(),
            },
            &Y2_WEIGHTS,
            tables,
        );
        model.levels.y2 = y2;
        add_block(&mut model, 24, &m);
    }
    for (plane, (src_c, pred_c)) in [(&src.u, &pred.u), (&src.v, &pred.v)]
        .into_iter()
        .enumerate()
    {
        let mut above = [false; 2];
        let mut left = [false; 2];
        for (i, left_nonzero) in left.iter_mut().enumerate() {
            for (j, above_nonzero) in above.iter_mut().enumerate() {
                let idx = i * 2 + j;
                let mut coeffs = residual_dct(src_c, pred_c, 8, i * 4, j * 4);
                let dct = coeffs;
                let m = model_block_parts(
                    &mut coeffs,
                    TrellisBlockSpec {
                        dc: factors.uv_dc,
                        ac: factors.uv_ac,
                        first_coeff: BlockType::UV.first_coeff(),
                        entry_ctx3: *above_nonzero as usize + *left_nonzero as usize,
                        plane: BlockType::UV.plane_index(),
                    },
                    dct_weights,
                    tables,
                );
                *above_nonzero = m.nonzero;
                *left_nonzero = m.nonzero;
                if plane == 0 {
                    model.dct.u[idx] = dct;
                    model.levels.u[idx] = coeffs;
                } else {
                    model.dct.v[idx] = dct;
                    model.levels.v[idx] = coeffs;
                }
                add_block(&mut model, 16 + 4 * plane as u32 + idx as u32, &m);
            }
        }
    }
    model
}

/// Quantise a modelled macroblock for real: the rounded levels, or the
/// §13 trellis on the blocks that keep a level. Returns the coefficients
/// and whether the macroblock is a §11.1 skip.
fn quantise(
    model: &MbModel,
    force_skip: bool,
    search: &InterSearch,
    place: &MbPlace,
    with_y2: bool,
    trellis: bool,
) -> (MbCoeffs, bool) {
    if force_skip || model.nonzero == 0 {
        return (MbCoeffs::default(), true);
    }
    let trellis_lambda = place.lambda * search.trellis_strength.lambda_scale();
    if !trellis || trellis_lambda <= 0.0 {
        return (model.levels, false);
    }
    let factors = place.factors;
    let mut out = MbCoeffs::default();
    let luma_type = if with_y2 {
        BlockType::YAfterY2
    } else {
        BlockType::YNoY2
    };
    let run = |block: &[i16; 16], dc: i16, ac: i16, block_type: BlockType| -> [i16; 16] {
        let mut b = *block;
        trellis_quantize_block(
            &mut b,
            TrellisBlockSpec {
                dc,
                ac,
                first_coeff: block_type.first_coeff(),
                entry_ctx3: 0,
                plane: block_type.plane_index(),
            },
            search.coeff_probs,
            search.rate_tables,
            trellis_lambda,
        );
        b
    };
    for idx in 0..16 {
        if model.nonzero >> idx & 1 != 0 {
            out.y[idx] = run(&model.dct.y[idx], factors.y1_dc, factors.y1_ac, luma_type);
        }
    }
    for idx in 0..4 {
        if model.nonzero >> (16 + idx) & 1 != 0 {
            out.u[idx] = run(
                &model.dct.u[idx],
                factors.uv_dc,
                factors.uv_ac,
                BlockType::UV,
            );
        }
        if model.nonzero >> (20 + idx) & 1 != 0 {
            out.v[idx] = run(
                &model.dct.v[idx],
                factors.uv_dc,
                factors.uv_ac,
                BlockType::UV,
            );
        }
    }
    if with_y2 && model.nonzero >> 24 & 1 != 0 {
        out.y2 = run(&model.dct.y2, factors.y2_dc, factors.y2_ac, BlockType::Y2);
    }
    let skip = !crate::loop_filter::mb_has_coeffs(&out);
    (out, skip)
}

/// The decoder's reconstruction of quantised `coeffs` on top of `pred`
/// (§14.1 dequantisation, §14.3 inverse WHT, §14.4 inverse DCT, §14.5
/// add) — the tail of `reconstruct_inter_mb` / `reconstruct_split_mv_mb`
/// applied to a prediction already in hand.
fn reconstruct(
    pred: &ReconstructedMb,
    coeffs: &MbCoeffs,
    factors: &MbDequantFactors,
    with_y2: bool,
    skip: bool,
) -> ReconstructedMb {
    let mut out = *pred;
    if skip {
        return out;
    }
    let mut dq = *coeffs;
    factors.dequantize(&mut dq);
    if with_y2 {
        let mut y2_residue = [0i16; 16];
        inverse_wht_4x4(&dq.y2, &mut y2_residue);
        for (block, dc) in dq.y.iter_mut().zip(y2_residue) {
            block[0] = dc;
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            inverse_dct_4x4_add_into(&dq.y[i * 4 + j], &mut out.y, 16, i, j);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            inverse_dct_4x4_add_into(&dq.u[i * 2 + j], &mut out.u, 8, i, j);
            inverse_dct_4x4_add_into(&dq.v[i * 2 + j], &mut out.v, 8, i, j);
        }
    }
    out
}

/// `coeffs` after §14.1 dequantisation (what the decoder's
/// reconstruction entry points take).
#[cfg(debug_assertions)]
fn dequantised(coeffs: &MbCoeffs, factors: &MbDequantFactors) -> MbCoeffs {
    let mut dq = *coeffs;
    factors.dequantize(&mut dq);
    dq
}

fn luma_ssd(a: &[u8; 256], b: &[u8; 256]) -> u64 {
    let mut acc = 0u32;
    for i in 0..256 {
        let d = a[i].abs_diff(b[i]) as u32;
        acc += d * d;
    }
    acc as u64
}

/// An inter candidate ready for the cost model.
struct InterCandidate {
    ref_frame: RefFrame,
    mode: InterMode,
    mv: Mv,
    /// Signalling bits: reference, mode tree, vector differential.
    bits: f64,
}

/// The reduced-effort search for one macroblock.
pub(super) fn decide_mb_fast(
    search: &InterSearch,
    pixels: &MbPixels,
    place: &MbPlace,
    shape: &SearchShape,
) -> MbDecision {
    // The §16.2 reference selector and the intra flag are priced at even
    // odds during the search; the probabilities that ship are fitted to
    // the decisions afterwards.
    const PROB_PICK: u8 = 128;
    let costs = search
        .mv_costs
        .as_ref()
        .expect("the frame driver tables the vector costs for the reduced search");
    let tables = search
        .rate_tables
        .expect("the frame driver attaches the rate tables for the reduced search");
    let src = &pixels.y;
    let lambda = place.lambda;
    let lambda_sad = lambda.sqrt();
    // The quantiser step seen from the pixel domain (the inverse DCT
    // divides by 8), and the error levels the search steers by.
    let qpix = (place.factors.y1_ac as u32 / 8).max(1);
    let good_enough = 256 * qpix / 2;
    let origin = ((place.mb_col * 16) as i32, (place.mb_row * 16) as i32);
    let bounds = MvClampRect::for_mb(place.mb_col, place.mb_row, search.mb_cols, search.mb_rows);
    let range_x = (
        (bounds.to_left / 4).max(-255),
        (bounds.to_right / 4).min(255),
    );
    let range_y = (
        (bounds.to_top / 4).max(-255),
        (bounds.to_bottom / 4).min(255),
    );

    // ---- Whole-pixel search of every reference -----------------------
    let mut refs: [Option<(RefSearch, Found, f64)>; 3] = [None, None, None];
    let mut leader = 0usize;
    let mut leader_cost = f64::INFINITY;
    for (n, picture) in search.refs.iter().enumerate() {
        let near = find_near_mvs(
            place.above,
            place.left,
            place.aboveleft,
            picture.frame,
            crate::near_mv::SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);
        let tree = |ones: usize, last: Option<bool>| -> f64 {
            let mut bits = 0.0;
            for p in probs.iter().take(ones) {
                bits += bool_bits(*p, true);
            }
            if let Some(bit) = last {
                bits += bool_bits(probs[ones], bit);
            }
            bits
        };
        let census = Census {
            best: clamp_mv(near.mvs[0], &bounds),
            nearest: clamp_mv(near.mvs[1], &bounds),
            near: clamp_mv(near.mvs[2], &bounds),
            mode_bits: [
                tree(0, Some(false)),
                tree(1, Some(false)),
                tree(2, Some(false)),
                tree(3, Some(false)),
                tree(4, None),
            ],
        };
        let r = RefSearch {
            picture,
            luma: LumaRef {
                plane: picture.planes.y,
                stride: picture.planes.y_stride,
                width: picture.planes.mb_cols * 16,
                height: picture.planes.mb_rows * 16,
            },
            census,
            range_x,
            range_y,
        };
        // Seeds: zero, the census vectors, and what the neighbours used
        // on this reference.
        let mut seeds = [Mv::default(); 6];
        let mut len = 1usize;
        for mv in [r.census.nearest, r.census.near, r.census.best] {
            seeds[len] = mv;
            len += 1;
        }
        for info in [place.left, place.above] {
            if info.ref_frame == Some(picture.frame) {
                seeds[len] = info.mv;
                len += 1;
            }
        }
        let descend = n == 0 || shape.search_other_refs;
        let found = whole_pixel_search(
            &r,
            src,
            origin,
            &seeds[..len],
            costs,
            lambda_sad,
            good_enough,
            shape,
            descend,
        );
        let ref_bits = ref_frame_tree_bits(picture.frame, PROB_PICK, PROB_PICK);
        let total = found.cost + lambda_sad * ref_bits;
        if total < leader_cost {
            leader_cost = total;
            leader = n;
        }
        let stop = found.sad <= good_enough;
        refs[n] = Some((r, found, total));
        if stop {
            // Good enough to stop looking at further references.
            break;
        }
    }

    // ---- Sub-pixel refinement of the leader (and a close runner-up) --
    let mut best_inter: Option<(InterCandidate, f64)> = None;
    let consider = |cand: InterCandidate, sad: u32, best: &mut Option<(InterCandidate, f64)>| {
        let cost = sad as f64 + lambda_sad * cand.bits;
        if best.as_ref().is_none_or(|(_, c)| cost < *c) {
            *best = Some((cand, cost));
        }
    };
    for (n, slot) in refs.iter().enumerate() {
        let Some((r, found, total)) = slot else {
            continue;
        };
        let refine = n == leader || (shape.refine_runner_up && *total < leader_cost * 1.125);
        if !refine {
            continue;
        }
        let ref_bits = ref_frame_tree_bits(r.picture.frame, PROB_PICK, PROB_PICK);
        let found = if found.sad > good_enough / 2 {
            sub_pixel_refine(
                r,
                src,
                origin,
                place,
                *found,
                costs,
                lambda_sad,
                search.filters,
                shape,
            )
        } else {
            *found
        };
        // The searched vector under the cheapest mode that expresses it.
        let zero = Mv::default();
        let census = &r.census;
        let (mode, mode_bits) = if found.mv == zero {
            (InterMode::Zero, census.mode_bits[0])
        } else if found.mv == census.nearest {
            (InterMode::Nearest, census.mode_bits[1])
        } else if found.mv == census.near {
            (InterMode::Near, census.mode_bits[2])
        } else {
            let diff = costs
                .bits(found.mv, census.best)
                .expect("searched vectors are inside the differential range");
            (InterMode::New, census.mode_bits[3] + diff)
        };
        consider(
            InterCandidate {
                ref_frame: r.picture.frame,
                mode,
                mv: found.mv,
                bits: ref_bits + mode_bits,
            },
            found.sad,
            &mut best_inter,
        );
        // The vectors that cost no differential.
        let mut free = |mode: InterMode, mv: Mv, mode_bits: f64| {
            if mv == found.mv {
                return;
            }
            let sad = if mv.row % 4 == 0 && mv.col % 4 == 0 {
                whole_sad(
                    &r.luma,
                    origin.0 + mv.col as i32 / 4,
                    origin.1 + mv.row as i32 / 4,
                    src,
                    u32::MAX,
                )
            } else {
                mb_luma_sad_at_mv(r.luma, place.mb_col, place.mb_row, src, mv)
            };
            consider(
                InterCandidate {
                    ref_frame: r.picture.frame,
                    mode,
                    mv,
                    bits: ref_bits + mode_bits,
                },
                sad,
                &mut best_inter,
            );
        };
        free(InterMode::Zero, zero, census.mode_bits[0]);
        if census.nearest != zero {
            free(InterMode::Nearest, census.nearest, census.mode_bits[1]);
        }
        if census.near != zero && census.near != census.nearest {
            free(InterMode::Near, census.near, census.mode_bits[2]);
        }
    }
    let (inter, inter_sad_cost) = best_inter.expect("the leader reference was refined");
    let inter_ref = search
        .refs
        .iter()
        .find(|p| p.frame == inter.ref_frame)
        .expect("the candidate's reference is one of the available ones");

    // ---- Cost model of the inter winner -------------------------------
    let inter_bit = bool_bits(PROB_PICK, true);
    let pred = predict_inter_mb(
        &inter_ref.planes,
        place.mb_col,
        place.mb_row,
        inter.mv,
        false,
        search.filters,
    );
    let model = model_mb(pixels, &pred, place.factors, tables, true);
    let (model_cost, skip) = model.cost(lambda);
    let mut best_cost = model_cost + lambda * (inter.bits + inter_bit);
    // How badly the prediction misses, in squared pixel-domain quantiser
    // steps per pixel: the gate for building costlier candidates.
    let miss = luma_ssd(src, &pred.y) / (256 * (qpix * qpix) as u64);

    enum Winner {
        Inter,
        Split(SplitMvCandidate, ReconstructedMb, MbModel, bool),
        Intra(IntraYMode, IntraUvMode, ReconstructedMb, MbModel, bool),
    }
    let mut winner = Winner::Inter;

    // ---- SPLITMV ------------------------------------------------------
    if !skip && miss >= shape.split_gate && !shape.split_partitions.is_empty() {
        let (r, _, _) = refs
            .iter()
            .flatten()
            .find(|(r, _, _)| r.picture.frame == inter.ref_frame)
            .expect("the winner's reference was searched");
        let ref_bits = ref_frame_tree_bits(inter.ref_frame, PROB_PICK, PROB_PICK);
        let mut best_split: Option<SplitMvCandidate> = None;
        for &partition in shape.split_partitions {
            if let Some(cand) = score_split_partition(
                partition,
                r.luma,
                place.mb_col,
                place.mb_row,
                src,
                r.census.best,
                place.above,
                place.left,
                search.mv_contexts,
                lambda_sad,
            ) && best_split.as_ref().is_none_or(|b| cand.j < b.j)
            {
                best_split = Some(cand);
            }
        }
        if let Some(cand) = best_split
            && cand.j + lambda_sad * (ref_bits + r.census.mode_bits[4]) < inter_sad_cost
        {
            // Signalling bits: `j` is prediction error plus
            // `lambda_sad ·` (partition + sub-vector bits).
            let mut sad = 0u32;
            let split_pred = predict_split_mv(
                &inter_ref.planes,
                place.mb_col,
                place.mb_row,
                &cand.split_mvs,
                false,
                search.filters,
            );
            for (a, b) in src.iter().zip(split_pred.y.iter()) {
                sad += a.abs_diff(*b) as u32;
            }
            let split_bits =
                ((cand.j - sad as f64) / lambda_sad).max(0.0) + ref_bits + r.census.mode_bits[4];
            let split_model = model_mb(pixels, &split_pred, place.factors, tables, false);
            let (cost, split_skip) = split_model.cost(lambda);
            let total = cost + lambda * (split_bits + inter_bit);
            if total < best_cost {
                best_cost = total;
                winner = Winner::Split(cand, split_pred, split_model, split_skip);
            }
        }
    }

    // ---- Intra ----------------------------------------------------------
    if search.pick_intra && !skip && miss >= shape.intra_gate {
        const Y_MODES: [IntraYMode; 4] =
            [IntraYMode::Dc, IntraYMode::V, IntraYMode::H, IntraYMode::Tm];
        const UV_MODES: [IntraUvMode; 4] = [
            IntraUvMode::Dc,
            IntraUvMode::V,
            IntraUvMode::H,
            IntraUvMode::Tm,
        ];
        let n = place.neighbors;
        let mut y_preds = [[0u8; 256]; 4];
        let mut y_err = [0u64; 4];
        for (k, &mode) in Y_MODES.iter().enumerate() {
            predict_y16x16(
                &mut y_preds[k],
                mode,
                n.y_above.as_ref(),
                n.y_left.as_ref(),
                n.y_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL),
            )
            .expect("whole-block modes only");
            y_err[k] = luma_ssd(src, &y_preds[k]);
        }
        let mut uv_preds = [([0u8; 64], [0u8; 64]); 4];
        let mut uv_err = [0u64; 4];
        for (k, &mode) in UV_MODES.iter().enumerate() {
            predict_uv8x8(
                &mut uv_preds[k].0,
                mode,
                n.u_above.as_ref(),
                n.u_left.as_ref(),
                n.u_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL),
            );
            predict_uv8x8(
                &mut uv_preds[k].1,
                mode,
                n.v_above.as_ref(),
                n.v_left.as_ref(),
                n.v_topleft.unwrap_or(DEFAULT_ABOVE_PIXEL),
            );
            uv_err[k] = super::block_ssd(&uv_preds[k].0, &pixels.u)
                + super::block_ssd(&uv_preds[k].1, &pixels.v);
        }
        let argmin = |errs: &[u64; 4]| -> usize {
            let mut best = 0;
            for k in 1..4 {
                if errs[k] < errs[best] {
                    best = k;
                }
            }
            best
        };
        let uv_k = argmin(&uv_err);
        let y_first = argmin(&y_err);
        // Worth modelling only if some intra prediction is at least in
        // the neighbourhood of the inter one.
        let inter_err = luma_ssd(src, &pred.y);
        if y_err[y_first] < inter_err.saturating_mul(2) {
            let intra_bit = bool_bits(PROB_PICK, false);
            let candidates: &[usize] = if shape.intra_all_modes {
                &[0, 1, 2, 3]
            } else {
                core::slice::from_ref(&y_first)
            };
            for &k in candidates {
                let intra_pred = ReconstructedMb {
                    y: y_preds[k],
                    u: uv_preds[uv_k].0,
                    v: uv_preds[uv_k].1,
                };
                let intra_model = model_mb(pixels, &intra_pred, place.factors, tables, true);
                let (cost, intra_skip) = intra_model.cost(lambda);
                let bits = if_ymode_tree_bits(Y_MODES[k])
                    + if_uv_mode_tree_bits(UV_MODES[uv_k])
                    + intra_bit;
                let total = cost + lambda * bits;
                if total < best_cost {
                    best_cost = total;
                    winner = Winner::Intra(
                        Y_MODES[k],
                        UV_MODES[uv_k],
                        intra_pred,
                        intra_model,
                        intra_skip,
                    );
                }
            }
        }
    }

    // ---- Quantise and reconstruct the winner ------------------------
    match winner {
        Winner::Inter => {
            let (coeffs, skip) = quantise(&model, skip, search, place, true, shape.trellis);
            let recon = reconstruct(&pred, &coeffs, place.factors, true, skip);
            #[cfg(debug_assertions)]
            {
                let dq = dequantised(&coeffs, place.factors);
                debug_assert_eq!(
                    recon,
                    crate::motion_comp::reconstruct_inter_mb(
                        &inter_ref.planes,
                        place.mb_col,
                        place.mb_row,
                        inter.mv,
                        false,
                        search.filters,
                        skip,
                        &dq.y2,
                        &dq.y,
                        &dq.u,
                        &dq.v,
                    )
                );
            }
            MbDecision {
                is_intra: false,
                y_mode: IntraYMode::Dc,
                uv_mode: IntraUvMode::Dc,
                ref_frame: inter.ref_frame,
                mode: inter.mode,
                mv: inter.mv,
                split: None,
                coeffs,
                skip,
                no_y2: false,
                recon,
            }
        }
        Winner::Split(cand, split_pred, split_model, split_skip) => {
            let (coeffs, skip) = quantise(
                &split_model,
                split_skip,
                search,
                place,
                false,
                shape.trellis,
            );
            let recon = reconstruct(&split_pred, &coeffs, place.factors, false, skip);
            #[cfg(debug_assertions)]
            {
                let dq = dequantised(&coeffs, place.factors);
                debug_assert_eq!(
                    recon,
                    crate::motion_comp::reconstruct_split_mv_mb(
                        &inter_ref.planes,
                        place.mb_col,
                        place.mb_row,
                        &cand.split_mvs,
                        false,
                        search.filters,
                        skip,
                        &dq.y,
                        &dq.u,
                        &dq.v,
                    )
                );
            }
            MbDecision {
                is_intra: false,
                y_mode: IntraYMode::Dc,
                uv_mode: IntraUvMode::Dc,
                ref_frame: inter.ref_frame,
                mode: InterMode::Split,
                mv: cand.split_mvs[15],
                split: Some(cand),
                coeffs,
                skip,
                no_y2: true,
                recon,
            }
        }
        Winner::Intra(y_mode, uv_mode, intra_pred, intra_model, intra_skip) => {
            let (coeffs, skip) =
                quantise(&intra_model, intra_skip, search, place, true, shape.trellis);
            let recon = reconstruct(&intra_pred, &coeffs, place.factors, true, skip);
            #[cfg(debug_assertions)]
            {
                let dq = dequantised(&coeffs, place.factors);
                debug_assert_eq!(
                    Ok(recon),
                    crate::reconstruct::decode_keyframe_mb_non_bpred(
                        y_mode,
                        uv_mode,
                        skip,
                        place.neighbors,
                        &dq.y2,
                        &dq.y,
                        &dq.u,
                        &dq.v,
                    )
                );
            }
            MbDecision {
                is_intra: true,
                y_mode,
                uv_mode,
                ref_frame: RefFrame::Last,
                mode: InterMode::Zero,
                mv: Mv::default(),
                split: None,
                coeffs,
                skip,
                no_y2: false,
                recon,
            }
        }
    }
}
