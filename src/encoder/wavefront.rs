//! Key-frame macroblock analysis as a macroblock-row wavefront, with the
//! §13 token pass riding behind it.
//!
//! Intra prediction of macroblock `(r, c)` reads reconstructed pixels of
//! `(r, c − 1)` (left column), `(r − 1, c − 1)` (corner), `(r − 1, c)`
//! (above row) and `(r − 1, c + 1)` (the §12.3 above-right extension).
//! Nothing else couples macroblocks during analysis — the rate estimates
//! use contexts local to the macroblock — so row `r` may run as soon as
//! row `r − 1` is two macroblocks ahead of it. Each row is one unit of
//! work; a worker claims the next unclaimed row and, per macroblock,
//! waits until the row above has published far enough.
//!
//! What a row shares with the row below is only its bottom pixel line
//! (luma and both chroma planes), published macroblock by macroblock.
//! The reconstruction itself is written into that row's own disjoint
//! strip of the frame planes.
//!
//! The §13.3 tokens split the same way. *Which* decisions a row codes
//! depends on the row above only through each column's "above" non-zero
//! context, a pure function of the coefficients that is published with
//! the bottom line — so each worker also tokenises its row into a list
//! of `(probability, bit)` decisions. What must happen in row order is
//! only the §7.3 arithmetic coding of those lists: whichever worker
//! finishes a row replays every list that is ready into its partition's
//! boolean encoder, so with more than one thread the entropy coding
//! overlaps the remaining analysis and its serial part is the bare
//! coder loop.
//!
//! The serial path is the same code on the calling thread, rows in
//! order. Because every macroblock sees the same neighbour pixels and
//! the token pass visits rows in the same order, the output does not
//! depend on the number of threads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};

use super::effort::{BmodeContext, EncodeEffort, analyze_mb_model};
use super::tape::TokenTape;
use super::{
    BMODE_CANDIDATES, BoolEncoder, EncodeError, I420Frame, IntraBmode, IntraYMode, MacroblockModes,
    MbEntropyCtx, MbRdCtx, analyze_mb_exhaustive, clear_skip_ctx, encode_mb_tokens,
};
use crate::frame::KeyframePlanes;
use crate::reconstruct::{MbNeighbors, ReconstructedMb};
use crate::rowsync::{AbortOnUnwind, Progress, SharedEntropy, SharedPixelLine, Wave};

/// Frame-constant inputs of the per-macroblock search.
pub(super) struct KeyframeSearch<'a> {
    pub(super) frame: &'a I420Frame<'a>,
    pub(super) rd: MbRdCtx<'a>,
    pub(super) effort: EncodeEffort,
    pub(super) mb_cols: usize,
    pub(super) mb_rows: usize,
    /// §9.5 token partition count (1 / 2 / 4 / 8).
    pub(super) num_partitions: usize,
    /// Leave the arithmetic coding of the token tapes to the caller
    /// (who fits the §13.4 probabilities to them first) instead of
    /// coding them against `rd.coeff_probs` as the rows complete.
    pub(super) defer_token_coding: bool,
    /// Length of the longest row tape recorded so far — what the next
    /// row reserves, so that a tape rarely has to grow (and be copied)
    /// while it is recorded. Never affects what is recorded.
    pub(super) tape_hint: AtomicUsize,
}

/// Result of the analysis + token pass over a whole frame.
pub(super) struct AnalysedFrame {
    /// Per-macroblock modes, raster order.
    pub(super) modes: Vec<MacroblockModes>,
    /// Per-macroblock "has any non-zero coefficient" flag, raster order
    /// — all the §15 loop filter needs of the residual.
    pub(super) has_coeffs: Vec<bool>,
    /// The §13.3 token decisions of each macroblock row.
    pub(super) tapes: Vec<TokenTape>,
    /// The finished §9.5 token partitions, coded against the search's
    /// probabilities — `None` when the search deferred the coding.
    pub(super) partitions: Option<Vec<Vec<u8>>>,
}

impl AnalysedFrame {
    /// The §13.4 branch counts of the frame's token decisions, on up to
    /// `threads` threads (a sum, so the same in any order).
    pub(super) fn tally(&self, counts: &mut super::BranchCounts, threads: usize) {
        let workers = threads.clamp(1, self.tapes.len().max(1));
        if workers == 1 {
            for tape in &self.tapes {
                tape.tally(counts);
            }
            return;
        }
        let share = self.tapes.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .tapes
                .chunks(share)
                .map(|tapes| {
                    scope.spawn(move || {
                        let mut part = super::empty_branch_counts();
                        for tape in tapes {
                            tape.tally(&mut part);
                        }
                        part
                    })
                })
                .collect();
            let flat = counts
                .as_flattened_mut()
                .as_flattened_mut()
                .as_flattened_mut();
            for handle in handles {
                let part = match handle.join() {
                    Ok(part) => part,
                    Err(panic) => std::panic::resume_unwind(panic),
                };
                let part = part.as_flattened().as_flattened().as_flattened();
                for (sum, add) in flat.iter_mut().zip(part) {
                    sum.0 += add.0;
                    sum.1 += add.1;
                }
            }
        });
    }

    /// Arithmetic-code the tapes against `probs`: macroblock row `r`
    /// into partition `r % partitions` (§20.4), each partition through
    /// its own boolean encoder.
    ///
    /// The partitions are independent bitstreams, so with a thread
    /// budget they are coded side by side.
    pub(super) fn code_partitions(
        &self,
        probs: &super::CoeffProbs,
        partitions: usize,
        threads: usize,
    ) -> Vec<Vec<u8>> {
        let code_one = |partition: usize| {
            let mut encoder = BoolEncoder::new();
            for tape in self.tapes.iter().skip(partition).step_by(partitions) {
                tape.replay(&mut encoder, probs);
            }
            encoder.finish()
        };
        if threads <= 1 || partitions == 1 {
            return (0..partitions).map(code_one).collect();
        }
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Vec<u8>>> = (0..partitions).map(|_| Mutex::new(Vec::new())).collect();
        let worker = || {
            loop {
                let partition = next.fetch_add(1, Ordering::Relaxed);
                if partition >= partitions {
                    break;
                }
                *lock(&slots[partition]) = code_one(partition);
            }
        };
        std::thread::scope(|scope| {
            for _ in 1..threads.min(partitions) {
                scope.spawn(worker);
            }
            worker();
        });
        slots
            .into_iter()
            .map(|slot| slot.into_inner().unwrap_or_else(|p| p.into_inner()))
            .collect()
    }
}

/// The bottom pixel line of one macroblock row's reconstruction, and
/// the §11.3 sub-block-mode context its bottom sub-block row presents to
/// the row below (four entries per macroblock).
struct RowBottom {
    pixels: SharedPixelLine,
    /// The four bottom sub-block modes of each macroblock, as positions
    /// in [`BMODE_CANDIDATES`], four bits each.
    bmodes: Vec<AtomicU16>,
    /// §13.3 "above" non-zero context each macroblock leaves for the one
    /// below it (per column, frame-lived: a skipped `B_PRED` macroblock
    /// passes the inherited Y2 slot through).
    entropy: SharedEntropy,
}

impl RowBottom {
    fn store_bmodes(&self, mb_col: usize, modes: &[IntraBmode; 4]) {
        let mut bits = 0u16;
        for (n, mode) in modes.iter().enumerate() {
            let position = BMODE_CANDIDATES
                .iter()
                .position(|candidate| candidate == mode)
                .expect("every sub-block mode is a candidate");
            bits |= (position as u16) << (4 * n);
        }
        self.bmodes[mb_col].store(bits, Ordering::Relaxed);
    }

    fn load_bmodes(&self, mb_col: usize) -> [IntraBmode; 4] {
        let bits = self.bmodes[mb_col].load(Ordering::Relaxed);
        core::array::from_fn(|n| BMODE_CANDIDATES[(bits >> (4 * n) & 15) as usize])
    }
}

/// One row's decisions, handed from the analysis to the token pass.
struct RowOutput {
    modes: Vec<MacroblockModes>,
    has_coeffs: Vec<bool>,
    /// The row's §13.3 token decisions, in coding order.
    tokens: TokenTape,
}

/// What a row shares with other workers.
struct RowShared {
    /// Bottom pixel line and contexts, valid for macroblock columns
    /// `< done`; lock-free, as both rows touch it once per macroblock
    /// (see `rowsync.rs`).
    bottom: RowBottom,
    /// Number of macroblocks of this row whose bottom line is published.
    done: Progress,
    /// The row's decisions, set once the whole row is analysed.
    output: Mutex<Option<RowOutput>>,
}

/// One row of work: its index and its strips of the frame planes.
struct RowJob<'p> {
    mb_row: usize,
    y: &'p mut [u8],
    u: &'p mut [u8],
    v: &'p mut [u8],
}

/// The in-order §13.3 token pass.
struct TokenPass {
    /// Next row to code.
    next_row: usize,
    partitions: Vec<BoolEncoder>,
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Analyse every macroblock (writing the unfiltered reconstruction into
/// `planes`) and code the token partitions, on up to `threads` threads.
pub(super) fn analyse_and_tokenize(
    search: &KeyframeSearch,
    planes: &mut KeyframePlanes,
    threads: usize,
) -> Result<AnalysedFrame, EncodeError> {
    let (mb_cols, mb_rows) = (search.mb_cols, search.mb_rows);
    let shared: Vec<RowShared> = (0..mb_rows)
        .map(|_| RowShared {
            bottom: RowBottom {
                pixels: SharedPixelLine::new(mb_cols),
                bmodes: (0..mb_cols).map(|_| AtomicU16::new(0)).collect(),
                entropy: SharedEntropy::new(mb_cols),
            },
            done: Progress::new(),
            output: Mutex::new(None),
        })
        .collect();
    let jobs: Vec<RowJob> = planes
        .y
        .chunks_mut(16 * planes.y_stride)
        .zip(planes.u.chunks_mut(8 * planes.uv_stride))
        .zip(planes.v.chunks_mut(8 * planes.uv_stride))
        .enumerate()
        .map(|(mb_row, ((y, u), v))| RowJob { mb_row, y, u, v })
        .collect();
    let jobs = Mutex::new(jobs.into_iter());
    let abort = Wave::new(mb_cols);
    let first_error: Mutex<Option<(usize, EncodeError)>> = Mutex::new(None);
    let tokens = Mutex::new(TokenPass {
        next_row: 0,
        partitions: (0..search.num_partitions)
            .map(|_| BoolEncoder::new())
            .collect(),
    });

    let worker = || {
        let _guard = AbortOnUnwind(&abort);
        loop {
            if abort.aborted() {
                break;
            }
            let Some(job) = lock(&jobs).next() else {
                break;
            };
            let mb_row = job.mb_row;
            if let Err(e) = analyse_row(search, job, &shared, &abort) {
                let mut slot = lock(&first_error);
                if slot.as_ref().is_none_or(|(row, _)| mb_row < *row) {
                    *slot = Some((mb_row, e));
                }
                abort.abort();
                break;
            }
            // Advance the token pass if nobody else is on it.
            if !search.defer_token_coding
                && let Ok(mut pass) = tokens.try_lock()
            {
                advance_tokens(search, &shared, &mut pass);
            }
        }
    };
    let workers = threads.clamp(1, mb_rows.max(1));
    if workers == 1 {
        worker();
    } else {
        std::thread::scope(|scope| {
            for _ in 1..workers {
                scope.spawn(worker);
            }
            worker();
        });
    }

    if let Some((_, e)) = lock(&first_error).take() {
        return Err(e);
    }
    let mut pass = tokens.into_inner().unwrap_or_else(|p| p.into_inner());
    if !search.defer_token_coding {
        advance_tokens(search, &shared, &mut pass);
        debug_assert_eq!(pass.next_row, mb_rows);
    }

    let mut modes = Vec::with_capacity(mb_rows * mb_cols);
    let mut has_coeffs = Vec::with_capacity(mb_rows * mb_cols);
    let mut tapes = Vec::with_capacity(mb_rows);
    for row in shared {
        let output = row
            .output
            .into_inner()
            .unwrap_or_else(|p| p.into_inner())
            .expect("every row was analysed");
        modes.extend(output.modes);
        has_coeffs.extend(output.has_coeffs);
        tapes.push(output.tokens);
    }
    let partitions = (!search.defer_token_coding)
        .then(|| pass.partitions.into_iter().map(|p| p.finish()).collect());
    Ok(AnalysedFrame {
        modes,
        has_coeffs,
        tapes,
        partitions,
    })
}

/// Arithmetic-code the recorded token decisions of every analysed row
/// from `pass.next_row` on, in order, stopping at the first row that is
/// not ready yet. Per the §20.4 row loop, macroblock row `r` goes to
/// partition `r % N`.
fn advance_tokens(search: &KeyframeSearch, shared: &[RowShared], pass: &mut TokenPass) {
    while pass.next_row < search.mb_rows {
        let guard = lock(&shared[pass.next_row].output);
        let Some(output) = guard.as_ref() else {
            break;
        };
        output.tokens.replay(
            &mut pass.partitions[pass.next_row % search.num_partitions],
            search.rd.coeff_probs,
        );
        pass.next_row += 1;
    }
}

/// Analyse one macroblock row, left to right.
fn analyse_row(
    search: &KeyframeSearch,
    job: RowJob,
    shared: &[RowShared],
    abort: &Wave,
) -> Result<(), EncodeError> {
    let mb_cols = search.mb_cols;
    let mb_row = job.mb_row;
    let above_row = mb_row.checked_sub(1).map(|r| &shared[r]);
    let this_row = &shared[mb_row];
    let y_stride = mb_cols * 16;
    let uv_stride = mb_cols * 8;
    let mut modes = Vec::with_capacity(mb_cols);
    let mut has_coeffs = Vec::with_capacity(mb_cols);
    // §13.3 token decisions of the row. The "left" non-zero context
    // resets at each row start; the "above" context of a macroblock is
    // what the macroblock over it left behind.
    let hint = search.tape_hint.load(Ordering::Relaxed);
    let mut tokens = TokenTape::with_capacity((hint + hint / 4).max(mb_cols * 64));
    let mut left_ctx = MbEntropyCtx::default();
    let mut left: Option<ReconstructedMb> = None;
    // §11.3 sub-block-mode contexts from the neighbouring macroblocks:
    // `B_DC_PRED` outside the frame, as the mode layer initialises them.
    let mut bmode_ctx = BmodeContext {
        above: [IntraBmode::Dc; 4],
        left: [IntraBmode::Dc; 4],
    };

    for mb_col in 0..mb_cols {
        let mut neighbors = MbNeighbors::default();
        bmode_ctx.above = [IntraBmode::Dc; 4];
        let mut above_ctx = MbEntropyCtx::default();
        if let Some(above) = above_row {
            // The above-right macroblock must be reconstructed too.
            let need = (mb_col + 2).min(mb_cols);
            if !abort.wait(&above.done, need) {
                return Ok(());
            }
            let bottom = &above.bottom;
            bottom.pixels.load_above(mb_col, mb_cols, &mut neighbors);
            bmode_ctx.above = bottom.load_bmodes(mb_col);
            above_ctx = bottom.entropy.load(mb_col);
        }
        if let Some(prev) = &left {
            let mut y_left = [0u8; 16];
            for (r, slot) in y_left.iter_mut().enumerate() {
                *slot = prev.y[r * 16 + 15];
            }
            neighbors.y_left = Some(y_left);
            let mut u_left = [0u8; 8];
            let mut v_left = [0u8; 8];
            for r in 0..8 {
                u_left[r] = prev.u[r * 8 + 7];
                v_left[r] = prev.v[r * 8 + 7];
            }
            neighbors.u_left = Some(u_left);
            neighbors.v_left = Some(v_left);
        }

        let pixels = search.frame.extract_mb(mb_row, mb_col);
        let analysis = match search.effort {
            EncodeEffort::Thorough => analyze_mb_exhaustive(&pixels, &neighbors, &search.rd),
            effort => analyze_mb_model(&pixels, &neighbors, &bmode_ctx, &search.rd, effort),
        };

        // A macroblock with no non-zero residual is a §11.1 skip: the
        // decoder reconstructs it as pure prediction.
        let mb_skip_coeff = analysis.nonzero_block_count == 0;
        // Reconstruct through the decoder's own §14.2 / §12.3
        // orchestrators so the next macroblock predicts from exactly the
        // pixels a decoder will hold.
        let mut dq = analysis.coeffs;
        search.rd.factors.dequantize(&mut dq);
        let recon = if analysis.y_mode == IntraYMode::B {
            crate::reconstruct::decode_keyframe_mb_bpred(
                analysis.b_subblock_modes.as_ref(),
                analysis.uv_mode,
                mb_skip_coeff,
                &neighbors,
                &dq.y,
                &dq.u,
                &dq.v,
            )
        } else {
            crate::reconstruct::decode_keyframe_mb_non_bpred(
                analysis.y_mode,
                analysis.uv_mode,
                mb_skip_coeff,
                &neighbors,
                &dq.y2,
                &dq.y,
                &dq.u,
                &dq.v,
            )
        }
        .map_err(EncodeError::Reconstruct)?;

        let (yx, cx) = (mb_col * 16, mb_col * 8);
        for r in 0..16 {
            let dst = r * y_stride + yx;
            job.y[dst..dst + 16].copy_from_slice(&recon.y[r * 16..r * 16 + 16]);
        }
        for r in 0..8 {
            let dst = r * uv_stride + cx;
            job.u[dst..dst + 8].copy_from_slice(&recon.u[r * 8..r * 8 + 8]);
            job.v[dst..dst + 8].copy_from_slice(&recon.v[r * 8..r * 8 + 8]);
        }
        // What this macroblock presents as sub-block-mode context to
        // its right and lower neighbours: its own edge sub-modes, or the
        // §11.3 projection of its whole-block mode.
        let projected = analysis.y_mode.project_to_subblock_context();
        let (edge_bottom, edge_right) = match &analysis.b_subblock_modes {
            Some(sub) => (
                [sub[12], sub[13], sub[14], sub[15]],
                [sub[3], sub[7], sub[11], sub[15]],
            ),
            None => ([projected; 4], [projected; 4]),
        };
        bmode_ctx.left = edge_right;

        // §13.3 token decisions of this macroblock. A skip macroblock
        // (§13.1) emits none, but clears its predictor slots.
        let use_bpred = analysis.y_mode == IntraYMode::B;
        if mb_skip_coeff {
            clear_skip_ctx(use_bpred, &mut above_ctx, &mut left_ctx);
        } else {
            encode_mb_tokens(
                &mut tokens,
                &analysis.coeffs,
                use_bpred,
                search.rd.coeff_probs,
                &mut above_ctx,
                &mut left_ctx,
            )
            .map_err(EncodeError::Token)?;
        }
        this_row.bottom.entropy.store(mb_col, &above_ctx);
        this_row.bottom.store_bmodes(mb_col, &edge_bottom);
        this_row.bottom.pixels.store(mb_col, &recon);
        abort.advance(&this_row.done, mb_col + 1);

        modes.push(MacroblockModes {
            segment_id: None,
            mb_skip_coeff,
            y_mode: analysis.y_mode,
            subblock_modes: analysis.b_subblock_modes,
            uv_mode: analysis.uv_mode,
        });
        has_coeffs.push(crate::loop_filter::mb_has_coeffs(&analysis.coeffs));
        left = Some(recon);
    }

    search.tape_hint.fetch_max(tokens.len(), Ordering::Relaxed);
    *lock(&this_row.output) = Some(RowOutput {
        modes,
        has_coeffs,
        tokens,
    });
    Ok(())
}
