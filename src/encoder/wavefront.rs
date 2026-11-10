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
//! The §13.3 token pass needs rows in order (each row's "above" non-zero
//! context is the previous row's outcome), but it does not need the
//! analysis of later rows: whichever worker finishes a row advances the
//! token pass over every row that is ready, so with more than one thread
//! the entropy coding overlaps the remaining analysis.
//!
//! The serial path is the same code on the calling thread, rows in
//! order. Because every macroblock sees the same neighbour pixels and
//! the token pass visits rows in the same order, the output does not
//! depend on the number of threads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::effort::{BmodeContext, EncodeEffort, analyze_mb_model};
use super::{
    BoolEncoder, EncodeError, I420Frame, IntraBmode, IntraYMode, MacroblockModes, MbCoeffs,
    MbEntropyCtx, MbRdCtx, TokenEncodeError, analyze_mb_exhaustive, clear_skip_ctx,
    encode_mb_tokens,
};
use crate::frame::KeyframePlanes;
use crate::intra_predict::DEFAULT_LEFT_PIXEL;
use crate::reconstruct::{MbNeighbors, ReconstructedMb};

/// Frame-constant inputs of the per-macroblock search.
pub(super) struct KeyframeSearch<'a> {
    pub(super) frame: &'a I420Frame<'a>,
    pub(super) rd: MbRdCtx<'a>,
    pub(super) effort: EncodeEffort,
    pub(super) mb_cols: usize,
    pub(super) mb_rows: usize,
    /// §9.5 token partition count (1 / 2 / 4 / 8).
    pub(super) num_partitions: usize,
}

/// Result of the analysis + token pass over a whole frame.
pub(super) struct AnalysedFrame {
    /// Per-macroblock modes, raster order.
    pub(super) modes: Vec<MacroblockModes>,
    /// Per-macroblock raw-quantised coefficients, raster order.
    pub(super) coeffs: Vec<MbCoeffs>,
    /// The finished §9.5 token partitions.
    pub(super) partitions: Vec<Vec<u8>>,
}

/// The bottom pixel line of one macroblock row's reconstruction, and
/// the §11.3 sub-block-mode context its bottom sub-block row presents to
/// the row below (four entries per macroblock).
struct RowBottom {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    bmodes: Vec<IntraBmode>,
}

/// One row's decisions, handed from the analysis to the token pass.
struct RowOutput {
    modes: Vec<MacroblockModes>,
    coeffs: Vec<MbCoeffs>,
}

/// What a row shares with other workers.
struct RowShared {
    /// Bottom pixel line, valid for macroblock columns `< done`.
    bottom: Mutex<RowBottom>,
    /// Number of macroblocks of this row whose bottom line is published.
    done: AtomicUsize,
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
    /// One above-context per macroblock column, frame-lived (§13.3).
    above_ctx: Vec<MbEntropyCtx>,
    error: Option<TokenEncodeError>,
}

/// Raises the abort flag if the thread unwinds, so workers waiting on
/// this thread's row stop instead of spinning forever.
struct AbortOnUnwind<'a>(&'a AtomicBool);

impl Drop for AbortOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
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
            bottom: Mutex::new(RowBottom {
                y: vec![0u8; mb_cols * 16],
                u: vec![0u8; mb_cols * 8],
                v: vec![0u8; mb_cols * 8],
                bmodes: vec![IntraBmode::Dc; mb_cols * 4],
            }),
            done: AtomicUsize::new(0),
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
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<(usize, EncodeError)>> = Mutex::new(None);
    let tokens = Mutex::new(TokenPass {
        next_row: 0,
        partitions: (0..search.num_partitions)
            .map(|_| BoolEncoder::new())
            .collect(),
        above_ctx: vec![MbEntropyCtx::default(); mb_cols],
        error: None,
    });

    let worker = || {
        let _guard = AbortOnUnwind(&abort);
        loop {
            if abort.load(Ordering::Acquire) {
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
                abort.store(true, Ordering::Release);
                break;
            }
            // Advance the token pass if nobody else is on it.
            if let Ok(mut pass) = tokens.try_lock() {
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
    advance_tokens(search, &shared, &mut pass);
    if let Some(e) = pass.error {
        return Err(EncodeError::Token(e));
    }
    debug_assert_eq!(pass.next_row, mb_rows);

    let mut modes = Vec::with_capacity(mb_rows * mb_cols);
    let mut coeffs = Vec::with_capacity(mb_rows * mb_cols);
    for row in shared {
        let output = row
            .output
            .into_inner()
            .unwrap_or_else(|p| p.into_inner())
            .expect("every row was analysed");
        modes.extend(output.modes);
        coeffs.extend(output.coeffs);
    }
    Ok(AnalysedFrame {
        modes,
        coeffs,
        partitions: pass.partitions.into_iter().map(|p| p.finish()).collect(),
    })
}

/// Code the tokens of every analysed row from `pass.next_row` on, in
/// order, stopping at the first row that is not ready yet.
///
/// Per the §20.4 row loop, macroblock row `r` goes to partition
/// `r % N`. The "left" context resets at each row start; the "above"
/// context is per column and frame-lived, shared across partitions.
fn advance_tokens(search: &KeyframeSearch, shared: &[RowShared], pass: &mut TokenPass) {
    while pass.next_row < search.mb_rows && pass.error.is_none() {
        let guard = lock(&shared[pass.next_row].output);
        let Some(output) = guard.as_ref() else {
            break;
        };
        let partition = &mut pass.partitions[pass.next_row % search.num_partitions];
        let mut left_ctx = MbEntropyCtx::default();
        for (mb_col, above_col) in pass.above_ctx.iter_mut().enumerate() {
            let mb = &output.modes[mb_col];
            let use_bpred = mb.y_mode == IntraYMode::B;
            if mb.mb_skip_coeff {
                // §13.1: a skip macroblock emits no tokens, but its
                // predictor slots are cleared.
                clear_skip_ctx(use_bpred, above_col, &mut left_ctx);
                continue;
            }
            if let Err(e) = encode_mb_tokens(
                partition,
                &output.coeffs[mb_col],
                use_bpred,
                search.rd.coeff_probs,
                above_col,
                &mut left_ctx,
            ) {
                pass.error = Some(e);
                return;
            }
        }
        pass.next_row += 1;
    }
}

/// Analyse one macroblock row, left to right.
fn analyse_row(
    search: &KeyframeSearch,
    job: RowJob,
    shared: &[RowShared],
    abort: &AtomicBool,
) -> Result<(), EncodeError> {
    let mb_cols = search.mb_cols;
    let mb_row = job.mb_row;
    let above_row = mb_row.checked_sub(1).map(|r| &shared[r]);
    let this_row = &shared[mb_row];
    let y_stride = mb_cols * 16;
    let uv_stride = mb_cols * 8;
    let mut modes = Vec::with_capacity(mb_cols);
    let mut coeffs = Vec::with_capacity(mb_cols);
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
        if let Some(above) = above_row {
            // The above-right macroblock must be reconstructed too.
            let need = (mb_col + 2).min(mb_cols);
            let mut spins = 0u32;
            while above.done.load(Ordering::Acquire) < need {
                if abort.load(Ordering::Acquire) {
                    return Ok(());
                }
                spins += 1;
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            let bottom = lock(&above.bottom);
            let (yx, cx) = (mb_col * 16, mb_col * 8);
            let mut y_above = [0u8; 16];
            y_above.copy_from_slice(&bottom.y[yx..yx + 16]);
            neighbors.y_above = Some(y_above);
            let mut u_above = [0u8; 8];
            u_above.copy_from_slice(&bottom.u[cx..cx + 8]);
            neighbors.u_above = Some(u_above);
            let mut v_above = [0u8; 8];
            v_above.copy_from_slice(&bottom.v[cx..cx + 8]);
            neighbors.v_above = Some(v_above);
            // §12.3: the rightmost macroblock repeats its own (-1, 15).
            let mut above_right = [bottom.y[yx + 15]; 4];
            if mb_col + 1 < mb_cols {
                above_right.copy_from_slice(&bottom.y[yx + 16..yx + 20]);
            }
            neighbors.y_above_right = Some(above_right);
            bmode_ctx
                .above
                .copy_from_slice(&bottom.bmodes[mb_col * 4..mb_col * 4 + 4]);
            if mb_col == 0 {
                // §20.14: left of the frame, below the top row → 129.
                neighbors.y_topleft = Some(DEFAULT_LEFT_PIXEL);
                neighbors.u_topleft = Some(DEFAULT_LEFT_PIXEL);
                neighbors.v_topleft = Some(DEFAULT_LEFT_PIXEL);
            } else {
                neighbors.y_topleft = Some(bottom.y[yx - 1]);
                neighbors.u_topleft = Some(bottom.u[cx - 1]);
                neighbors.v_topleft = Some(bottom.v[cx - 1]);
            }
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
        {
            let mut bottom = lock(&this_row.bottom);
            bottom.bmodes[mb_col * 4..mb_col * 4 + 4].copy_from_slice(&edge_bottom);
            bottom.y[yx..yx + 16].copy_from_slice(&recon.y[240..256]);
            bottom.u[cx..cx + 8].copy_from_slice(&recon.u[56..64]);
            bottom.v[cx..cx + 8].copy_from_slice(&recon.v[56..64]);
        }
        this_row.done.store(mb_col + 1, Ordering::Release);

        modes.push(MacroblockModes {
            segment_id: None,
            mb_skip_coeff,
            y_mode: analysis.y_mode,
            subblock_modes: analysis.b_subblock_modes,
            uv_mode: analysis.uv_mode,
        });
        coeffs.push(analysis.coeffs);
        left = Some(recon);
    }

    *lock(&this_row.output) = Some(RowOutput { modes, coeffs });
    Ok(())
}
