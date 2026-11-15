//! Residual decoding and reconstruction of one frame on several threads.
//!
//! The mode layer of a frame lives in the first partition and is parsed
//! up front, on the calling thread (it is one arithmetic-coded stream).
//! What remains per macroblock row is
//!
//! * **token decoding** — §9.5 gives macroblock row `r` to DCT partition
//!   `r mod N`, each partition its own boolean decoder; within a
//!   partition rows must be read in order, and macroblock `(r, c)` needs
//!   the "above" non-zero context `(r − 1, c)` left behind;
//! * **reconstruction** — motion compensation reads only the reference
//!   frames; an intra macroblock predicts from the reconstructed pixels
//!   of `(r, c − 1)`, `(r − 1, c − 1)`, `(r − 1, c)` and `(r − 1, c + 1)`.
//!   (The loop filter runs afterwards, on the whole frame — intra
//!   prediction reads unfiltered pixels.)
//!
//! Each row therefore becomes two units of work, each kind claimed in
//! row order. A token row waits for the previous row of its partition
//! to finish and, per macroblock, for the row above to be one macroblock
//! ahead; at most as many workers take token rows as there are
//! partitions (with a single partition one worker reads them all, never
//! waiting). A reconstruction row waits for its row's tokens and, only
//! at an intra macroblock, for the row above to be two macroblocks
//! ahead — motion-compensated rows do not wait on each other at all.
//! Every wait is on a row that was claimed earlier, so it ends.
//!
//! Every macroblock sees exactly the contexts and neighbour pixels the
//! raster-order decoder shows it, so the pictures are identical at every
//! thread count.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::backoff::{SharedEntropy, SharedPixelLine};
use crate::bool_decoder::BoolDecoder;
use crate::dct_tokens::{CoeffProbs, MbEntropyCtx};
use crate::decoder::DecodeError;
use crate::dequant::{MbDequantFactors, decode_and_dequantize_mb};
use crate::frame::{FrameError, KeyframePlanes, MbCoeffs};
use crate::loop_filter::MAX_MB_SEGMENTS;
use crate::macroblock::{IntraYMode, MacroblockModes};
use crate::motion_comp::{ReferencePlanes, reconstruct_inter_mb, reconstruct_split_mv_mb};
use crate::motion_vector::Mv;
use crate::reconstruct::{MbNeighbors, ReconstructedMb};

/// How one macroblock is predicted.
#[derive(Clone, Copy)]
pub(super) enum Prediction {
    /// Intra, with the modes of the frame's `modes` entry.
    Intra,
    /// One motion vector for the whole macroblock; `reference` indexes
    /// the frame's reference list.
    Whole { reference: usize, mv: Mv },
    /// `SPLITMV`: one vector per luma sub-block.
    Split { reference: usize, mvs: [Mv; 16] },
}

/// Frame-constant inputs.
pub(super) struct RowDecode<'a> {
    pub(super) mb_cols: usize,
    pub(super) mb_rows: usize,
    /// Per-macroblock modes (segment, skip flag, intra modes), raster
    /// order.
    pub(super) modes: &'a [MacroblockModes],
    /// Per-macroblock prediction, raster order.
    pub(super) predictions: &'a [Prediction],
    /// The reference pictures [`Prediction`] indexes.
    pub(super) references: &'a [ReferencePlanes<'a>],
    /// The §9.5 DCT partitions.
    pub(super) partitions: &'a [&'a [u8]],
    pub(super) coeff_probs: &'a CoeffProbs,
    pub(super) dequant_factors: &'a [MbDequantFactors; MAX_MB_SEGMENTS],
    /// §18.1 full-pixel flag and §18.3 tap set of the frame's version.
    pub(super) full_pixel: bool,
    pub(super) filters: &'a [[i32; 6]; 8],
}

/// What a row publishes while it is worked on.
struct RowShared {
    /// Macroblocks of this row whose tokens are decoded.
    tokens_done: AtomicUsize,
    /// "Above" non-zero context each macroblock leaves for the one below
    /// (valid for columns `< tokens_done`).
    entropy: SharedEntropy,
    /// The row's dequantised coefficients, once its tokens are complete.
    coeffs: Mutex<Option<Vec<MbCoeffs>>>,
    /// Macroblocks of this row that are reconstructed.
    pixels_done: AtomicUsize,
    /// Bottom pixel line of the reconstruction (valid for columns
    /// `< pixels_done`).
    bottom: SharedPixelLine,
    /// Per-macroblock "any non-zero coefficient", for the loop filter.
    has_coeffs: Mutex<Vec<bool>>,
}

/// One row's strips of the frame planes.
struct RowStrips<'p> {
    mb_row: usize,
    y: &'p mut [u8],
    u: &'p mut [u8],
    v: &'p mut [u8],
}

/// Raises the abort flag if the thread unwinds, so nobody waits on this
/// thread's row forever.
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

/// Spin, then yield, until `ready()`; `false` if the decode was aborted
/// meanwhile.
fn wait_until(abort: &AtomicBool, ready: impl Fn() -> bool) -> bool {
    let mut backoff = crate::backoff::Backoff::new();
    while !ready() {
        if abort.load(Ordering::Acquire) {
            return false;
        }
        backoff.pause();
    }
    true
}

/// Decode the residuals of every macroblock and reconstruct the frame
/// (unfiltered) on up to `threads` threads. Returns the planes and the
/// per-macroblock "has coefficients" flags.
pub(super) fn decode_rows(
    frame: &RowDecode,
    threads: usize,
) -> Result<(KeyframePlanes, Vec<bool>), DecodeError> {
    let (mb_cols, mb_rows) = (frame.mb_cols, frame.mb_rows);
    let (y_stride, uv_stride) = (mb_cols * 16, mb_cols * 8);
    let mut planes = KeyframePlanes {
        y: vec![0u8; y_stride * mb_rows * 16],
        u: vec![0u8; uv_stride * mb_rows * 8],
        v: vec![0u8; uv_stride * mb_rows * 8],
        y_stride,
        uv_stride,
        mb_cols,
        mb_rows,
    };
    let shared: Vec<RowShared> = (0..mb_rows)
        .map(|_| RowShared {
            tokens_done: AtomicUsize::new(0),
            entropy: SharedEntropy::new(mb_cols),
            coeffs: Mutex::new(None),
            pixels_done: AtomicUsize::new(0),
            bottom: SharedPixelLine::new(mb_cols),
            has_coeffs: Mutex::new(Vec::new()),
        })
        .collect();
    // One boolean decoder per partition a row routes to (the tolerant
    // §20.2 initialisation: a DCT partition shorter than two bytes is
    // legal).
    let decoders: Vec<Mutex<BoolDecoder<'_>>> = frame
        .partitions
        .iter()
        .map(|p| Mutex::new(BoolDecoder::init_partition(p)))
        .collect();

    let strips = Mutex::new(
        planes
            .y
            .chunks_mut(16 * y_stride)
            .zip(planes.u.chunks_mut(8 * uv_stride))
            .zip(planes.v.chunks_mut(8 * uv_stride))
            .enumerate()
            .map(|(mb_row, ((y, u), v))| RowStrips { mb_row, y, u, v }),
    );
    let next_token_row = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<(usize, DecodeError)>> = Mutex::new(None);
    let fail = |index: usize, error: DecodeError| {
        let mut slot = lock(&first_error);
        if slot.as_ref().is_none_or(|(at, _)| index < *at) {
            *slot = Some((index, error));
        }
        abort.store(true, Ordering::Release);
    };

    // A worker first takes token rows (if it is one of the token
    // workers), then reconstruction rows, each in row order.
    let worker = |tokens_first: bool| {
        let _guard = AbortOnUnwind(&abort);
        if tokens_first {
            loop {
                if abort.load(Ordering::Acquire) {
                    return;
                }
                let mb_row = next_token_row.fetch_add(1, Ordering::Relaxed);
                if mb_row >= mb_rows {
                    break;
                }
                if let Err((index, error)) =
                    decode_token_row(frame, mb_row, &shared, &decoders, &abort)
                {
                    fail(index, error);
                    return;
                }
            }
        }
        loop {
            if abort.load(Ordering::Acquire) {
                return;
            }
            let Some(row) = lock(&strips).next() else {
                return;
            };
            if let Err((index, error)) = reconstruct_row(frame, row, &shared, &abort) {
                fail(index, error);
                return;
            }
        }
    };
    let workers = threads.clamp(1, (2 * mb_rows).max(1));
    // Token rows of one partition chain, so more token workers than
    // partitions would only wait; and tokens are about a third of the
    // work of these two stages, so more than two reconstruction workers
    // per token worker would mostly wait for tokens.
    let token_workers = decoders.len().min(workers.div_ceil(3)).max(1);
    let workers = workers.min(3 * token_workers);
    if workers == 1 {
        worker(true);
    } else {
        std::thread::scope(|scope| {
            for n in 1..workers {
                let worker = &worker;
                scope.spawn(move || worker(n < token_workers));
            }
            worker(true);
        });
    }

    if let Some((_, error)) = lock(&first_error).take() {
        return Err(error);
    }
    let mut has_coeffs = Vec::with_capacity(mb_rows * mb_cols);
    for row in &shared {
        has_coeffs.append(&mut lock(&row.has_coeffs));
    }
    debug_assert_eq!(has_coeffs.len(), mb_rows * mb_cols);
    Ok((planes, has_coeffs))
}

/// Decode and dequantise the tokens of one macroblock row.
fn decode_token_row(
    frame: &RowDecode,
    mb_row: usize,
    shared: &[RowShared],
    decoders: &[Mutex<BoolDecoder<'_>>],
    abort: &AtomicBool,
) -> Result<(), (usize, DecodeError)> {
    let mb_cols = frame.mb_cols;
    let partition = mb_row % decoders.len();
    // The partition's previous row must have released the decoder.
    if let Some(previous) = mb_row.checked_sub(decoders.len())
        && !wait_until(abort, || {
            shared[previous].tokens_done.load(Ordering::Acquire) >= mb_cols
        })
    {
        return Ok(());
    }
    let mut dec = lock(&decoders[partition]);
    let above_row = mb_row.checked_sub(1).map(|r| &shared[r]);
    let this_row = &shared[mb_row];
    let mut coeffs: Vec<MbCoeffs> = Vec::with_capacity(mb_cols);
    let mut has_coeffs: Vec<bool> = Vec::with_capacity(mb_cols);
    let mut left = MbEntropyCtx::default();
    for mb_col in 0..mb_cols {
        let raster = mb_row * mb_cols + mb_col;
        let mut above_ctx = MbEntropyCtx::default();
        if let Some(above) = above_row {
            if !wait_until(abort, || above.tokens_done.load(Ordering::Acquire) > mb_col) {
                return Ok(());
            }
            above_ctx = above.entropy.load(mb_col);
        }
        let mb = &frame.modes[raster];
        let has_y2 = match frame.predictions[raster] {
            Prediction::Intra => mb.y_mode != IntraYMode::B,
            Prediction::Whole { .. } => true,
            Prediction::Split { .. } => false,
        };
        let segment = (mb.segment_id.unwrap_or(0) as usize).min(MAX_MB_SEGMENTS - 1);
        let mb_coeffs = decode_and_dequantize_mb(
            &mut dec,
            has_y2,
            mb.mb_skip_coeff,
            frame.coeff_probs,
            &frame.dequant_factors[segment],
            &mut above_ctx,
            &mut left,
        )
        .map_err(|source| {
            (
                raster,
                DecodeError::MbCoeffs {
                    index: raster,
                    source,
                },
            )
        })?;
        has_coeffs.push(crate::loop_filter::mb_has_coeffs(&mb_coeffs));
        coeffs.push(mb_coeffs);
        this_row.entropy.store(mb_col, &above_ctx);
        if mb_col + 1 == mb_cols {
            // Hand the row over before announcing its last macroblock.
            *lock(&this_row.has_coeffs) = core::mem::take(&mut has_coeffs);
            *lock(&this_row.coeffs) = Some(core::mem::take(&mut coeffs));
        }
        this_row.tokens_done.store(mb_col + 1, Ordering::Release);
    }
    Ok(())
}

/// Reconstruct one macroblock row into its strips of the frame planes.
fn reconstruct_row(
    frame: &RowDecode,
    RowStrips { mb_row, y, u, v }: RowStrips,
    shared: &[RowShared],
    abort: &AtomicBool,
) -> Result<(), (usize, DecodeError)> {
    let mb_cols = frame.mb_cols;
    let this_row = &shared[mb_row];
    if !wait_until(abort, || {
        this_row.tokens_done.load(Ordering::Acquire) >= mb_cols
    }) {
        return Ok(());
    }
    let coeffs = lock(&this_row.coeffs)
        .take()
        .expect("a complete token row has published its coefficients");
    let above_row = mb_row.checked_sub(1).map(|r| &shared[r]);
    let (y_stride, uv_stride) = (mb_cols * 16, mb_cols * 8);
    let mut left: Option<ReconstructedMb> = None;

    for (mb_col, c) in coeffs.iter().enumerate() {
        let raster = mb_row * mb_cols + mb_col;
        let mb = &frame.modes[raster];
        // Only intra prediction reads the row above.
        if let Some(above) = above_row
            && matches!(frame.predictions[raster], Prediction::Intra)
            && !wait_until(abort, || {
                above.pixels_done.load(Ordering::Acquire) >= (mb_col + 2).min(mb_cols)
            })
        {
            return Ok(());
        }
        let recon = match &frame.predictions[raster] {
            Prediction::Whole { reference, mv } => reconstruct_inter_mb(
                &frame.references[*reference],
                mb_col,
                mb_row,
                *mv,
                frame.full_pixel,
                frame.filters,
                mb.mb_skip_coeff,
                &c.y2,
                &c.y,
                &c.u,
                &c.v,
            ),
            Prediction::Split { reference, mvs } => reconstruct_split_mv_mb(
                &frame.references[*reference],
                mb_col,
                mb_row,
                mvs,
                frame.full_pixel,
                frame.filters,
                mb.mb_skip_coeff,
                &c.y,
                &c.u,
                &c.v,
            ),
            Prediction::Intra => {
                let mut neighbors = MbNeighbors::default();
                if let Some(above) = above_row {
                    above.bottom.load_above(mb_col, mb_cols, &mut neighbors);
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
                if mb.y_mode == IntraYMode::B {
                    crate::reconstruct::decode_keyframe_mb_bpred(
                        mb.subblock_modes.as_ref(),
                        mb.uv_mode,
                        mb.mb_skip_coeff,
                        &neighbors,
                        &c.y,
                        &c.u,
                        &c.v,
                    )
                } else {
                    crate::reconstruct::decode_keyframe_mb_non_bpred(
                        mb.y_mode,
                        mb.uv_mode,
                        mb.mb_skip_coeff,
                        &neighbors,
                        &c.y2,
                        &c.y,
                        &c.u,
                        &c.v,
                    )
                }
                .map_err(|source| {
                    (
                        raster,
                        DecodeError::Frame(FrameError::Macroblock {
                            index: raster,
                            source,
                        }),
                    )
                })?
            }
        };

        let (yx, cx) = (mb_col * 16, mb_col * 8);
        for r in 0..16 {
            let dst = r * y_stride + yx;
            y[dst..dst + 16].copy_from_slice(&recon.y[r * 16..r * 16 + 16]);
        }
        for r in 0..8 {
            let dst = r * uv_stride + cx;
            u[dst..dst + 8].copy_from_slice(&recon.u[r * 8..r * 8 + 8]);
            v[dst..dst + 8].copy_from_slice(&recon.v[r * 8..r * 8 + 8]);
        }
        this_row.bottom.store(mb_col, &recon);
        this_row.pixels_done.store(mb_col + 1, Ordering::Release);
        left = Some(recon);
    }
    Ok(())
}
