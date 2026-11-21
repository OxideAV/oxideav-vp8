//! A quick look at a key frame's token statistics before the search.
//!
//! The mode search prices every candidate's tokens against one
//! probability table. The table a frame finally ships is fitted to the
//! frame itself (§13.4), but that fit only exists once the search is
//! over — and a search that priced against the §13.5 defaults has
//! optimised for costs the frame does not end up paying. A second
//! search against the fitted table closes that gap at twice the time.
//!
//! The pilot closes most of it for a few percent: it runs the cheapest
//! model search over a sparse set of macroblock rows, predicting from
//! the *source* picture instead of a reconstruction (so the rows are
//! independent of each other and of the real search, and spread over
//! the caller's thread budget without changing the result), tallies the
//! tokens those rows would code, and blends the tallies into the
//! defaults. The real search then prices against that table. Nothing of
//! the pilot reaches the stream: the shipped probabilities are fitted to
//! the real search's own decisions afterwards.

use super::effort::{BmodeContext, EncodeEffort, analyze_mb_model};
use super::tape::TokenTape;
use super::{
    BranchCounts, CoeffProbs, I420Frame, IntraBmode, IntraYMode, MbEntropyCtx, MbRdCtx,
    clear_skip_ctx, empty_branch_counts, encode_mb_tokens,
};
use crate::reconstruct::MbNeighbors;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Weight of the §13.5 default, in observed decisions, when a tally is
/// blended into it: a slot the pilot saw this many times moves half way
/// to its observed frequency.
const PRIOR_WEIGHT: u64 = 24;

/// About how many macroblocks the pilot looks at, however large the
/// picture.
const PILOT_MACROBLOCKS: usize = 768;

/// Length of a sampled run of macroblocks.
const RUN: usize = 8;

/// The intra-prediction neighbourhood of a macroblock taken from the
/// source picture (clamped at the picture edge the way the encoder pads
/// it). Frame-edge macroblocks see the same "no neighbour" as in a real
/// encode.
fn source_neighbors(frame: &I420Frame, mb_row: usize, mb_col: usize) -> MbNeighbors {
    let (x0, y0) = (mb_col * 16, mb_row * 16);
    let (cx0, cy0) = (mb_col * 8, mb_row * 8);
    let mut n = MbNeighbors::default();
    if mb_row > 0 {
        n.y_above = Some(core::array::from_fn(|c| frame.y_at(x0 + c, y0 - 1)));
        n.y_above_right = Some(core::array::from_fn(|c| frame.y_at(x0 + 16 + c, y0 - 1)));
        n.u_above = Some(core::array::from_fn(|c| {
            frame.uv_at(cx0 + c, cy0 - 1, false)
        }));
        n.v_above = Some(core::array::from_fn(|c| {
            frame.uv_at(cx0 + c, cy0 - 1, true)
        }));
    }
    if mb_col > 0 {
        n.y_left = Some(core::array::from_fn(|r| frame.y_at(x0 - 1, y0 + r)));
        n.u_left = Some(core::array::from_fn(|r| {
            frame.uv_at(cx0 - 1, cy0 + r, false)
        }));
        n.v_left = Some(core::array::from_fn(|r| {
            frame.uv_at(cx0 - 1, cy0 + r, true)
        }));
    }
    if mb_row > 0 && mb_col > 0 {
        n.y_topleft = Some(frame.y_at(x0 - 1, y0 - 1));
        n.u_topleft = Some(frame.uv_at(cx0 - 1, cy0 - 1, false));
        n.v_topleft = Some(frame.uv_at(cx0 - 1, cy0 - 1, true));
    } else if mb_row > 0 {
        // §20.14: left of the frame, below the top row.
        let left = crate::intra_predict::DEFAULT_LEFT_PIXEL;
        n.y_topleft = Some(left);
        n.u_topleft = Some(left);
        n.v_topleft = Some(left);
    }
    n
}

/// Tally the §13.3 token decisions of every `stride`-th macroblock row
/// under the cheapest model search, predicting from the source.
///
/// A sampled row has no coded row above it, so the "above" token context
/// of a macroblock is stood in for by what its left neighbour left
/// behind — drawn from the same picture, which is all a tally needs.
/// That makes the rows independent: they run on up to `threads` scoped
/// threads, and the tallies add up to the same counts in any order.
pub(super) fn pilot_branch_counts(
    frame: &I420Frame,
    rd: &MbRdCtx,
    (mb_cols, mb_rows): (usize, usize),
    stride: usize,
    threads: usize,
) -> BranchCounts {
    let stride = stride.max(1);
    let rows: Vec<usize> = (stride / 2..mb_rows).step_by(stride).collect();
    // The statistics settle after a few hundred macroblocks, so a large
    // picture is sampled in runs of `RUN` macroblocks, one run out of
    // every `period`, staggered from row to row.
    let period = (rows.len() * mb_cols / PILOT_MACROBLOCKS).max(1);
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut counts = empty_branch_counts();
        let mut tape = TokenTape::with_capacity(mb_cols * 256);
        loop {
            let n = next.fetch_add(1, Ordering::Relaxed);
            let Some(&mb_row) = rows.get(n) else {
                break;
            };
            let first_run = (n * 3) % period;
            for run in (first_run..mb_cols.div_ceil(RUN)).step_by(period) {
                let cols = run * RUN..((run + 1) * RUN).min(mb_cols);
                pilot_run(frame, rd, mb_row, cols, &mut tape);
            }
            tape.tally(&mut counts);
            tape.clear();
        }
        counts
    };
    let workers = threads.clamp(1, rows.len().max(1));
    if workers == 1 {
        return worker();
    }
    let mut total = empty_branch_counts();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (1..workers).map(|_| scope.spawn(worker)).collect();
        let mut parts = vec![worker()];
        for handle in handles {
            match handle.join() {
                Ok(part) => parts.push(part),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        let flat_total = total
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut();
        for part in &parts {
            let flat = part.as_flattened().as_flattened().as_flattened();
            for (sum, add) in flat_total.iter_mut().zip(flat) {
                sum.0 += add.0;
                sum.1 += add.1;
            }
        }
    });
    total
}

/// Record the token decisions of one run of macroblocks on `tape`.
fn pilot_run(
    frame: &I420Frame,
    rd: &MbRdCtx,
    mb_row: usize,
    cols: core::ops::Range<usize>,
    tape: &mut TokenTape,
) {
    let mut left_ctx = MbEntropyCtx::default();
    let mut above_proxy = MbEntropyCtx::default();
    let mut bmode_ctx = BmodeContext {
        above: [IntraBmode::Dc; 4],
        left: [IntraBmode::Dc; 4],
    };
    for mb_col in cols {
        let pixels = frame.extract_mb(mb_row, mb_col);
        let neighbors = source_neighbors(frame, mb_row, mb_col);
        let analysis = analyze_mb_model(&pixels, &neighbors, &bmode_ctx, rd, EncodeEffort::Fastest);
        let projected = analysis.y_mode.project_to_subblock_context();
        bmode_ctx.left = match &analysis.b_subblock_modes {
            Some(sub) => [sub[3], sub[7], sub[11], sub[15]],
            None => [projected; 4],
        };
        let use_bpred = analysis.y_mode == IntraYMode::B;
        if analysis.nonzero_block_count == 0 {
            clear_skip_ctx(use_bpred, &mut above_proxy, &mut left_ctx);
        } else {
            // A magnitude past the §13.2 token range ends the block's
            // walk early; such a block just contributes less to the tally.
            let _ = encode_mb_tokens(
                tape,
                &analysis.coeffs,
                use_bpred,
                rd.coeff_probs,
                &mut above_proxy,
                &mut left_ctx,
            );
        }
    }
}

/// The §13.5 defaults pulled towards the observed branch frequencies:
/// each probability becomes the frequency of its `false` branch over the
/// observed decisions plus [`PRIOR_WEIGHT`] decisions at the default.
pub(super) fn blend_into_defaults(counts: &BranchCounts) -> CoeffProbs {
    let mut probs = crate::dct_tokens::DEFAULT_COEFF_PROBS;
    let flat_probs = probs
        .as_flattened_mut()
        .as_flattened_mut()
        .as_flattened_mut();
    let flat_counts = counts.as_flattened().as_flattened().as_flattened();
    for (prob, &(zeros, ones)) in flat_probs.iter_mut().zip(flat_counts) {
        let total = zeros as u64 + ones as u64;
        if total == 0 {
            continue;
        }
        let weighted = zeros as u64 * 256 + PRIOR_WEIGHT * *prob as u64;
        let denom = total + PRIOR_WEIGHT;
        *prob = ((weighted + denom / 2) / denom).clamp(1, 255) as u8;
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_keeps_unseen_slots_and_moves_seen_ones() {
        let mut counts = empty_branch_counts();
        let defaults = crate::dct_tokens::DEFAULT_COEFF_PROBS;
        assert_eq!(blend_into_defaults(&counts), defaults);
        // A slot that only ever took the `true` branch moves down, one
        // that only took `false` moves up, in proportion to the evidence.
        counts[1][2][0][0] = (0, PRIOR_WEIGHT as u32);
        counts[1][2][0][1] = (1_000_000, 0);
        let blended = blend_into_defaults(&counts);
        let before = defaults[1][2][0][0] as u32;
        assert_eq!(blended[1][2][0][0] as u32, before.div_ceil(2));
        assert_eq!(blended[1][2][0][1], 255);
        assert_eq!(blended[0], defaults[0]);
    }
}
