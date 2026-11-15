//! Inter-frame macroblock analysis as a macroblock-row wavefront.
//!
//! What couples macroblock `(r, c)` of an inter frame to the rest of the
//! frame during analysis:
//!
//! * the §16.3 neighbour census reads the mode records of `(r, c − 1)`,
//!   `(r − 1, c)` and `(r − 1, c − 1)`;
//! * an intra candidate predicts from reconstructed pixels of the same
//!   three neighbours;
//! * the §13.3 token decisions depend on the "above" non-zero context
//!   `(r − 1, c)` left behind.
//!
//! Motion-compensated prediction reads only the reference frames. So, as
//! for key frames (see `wavefront.rs`), row `r` may run as soon as row
//! `r − 1` is two macroblocks ahead; each row is one unit of work and a
//! worker waits per macroblock until the row above has published far
//! enough. A row shares with the row below its bottom pixel line, its
//! mode records and its token contexts.
//!
//! Each row tokenises itself onto a [`TokenTape`]: the §13.3 decisions
//! with the *slot* of the coefficient probability each was coded against
//! instead of the probability itself. The tape is what the §13.4
//! probability fit counts, and the frame driver replays it into the
//! partition's boolean encoder with whichever probability table ships —
//! so fitting the probabilities needs no second analysis pass.
//!
//! The serial path is the same code on the calling thread with rows in
//! order; every macroblock sees the same neighbour state either way, so
//! the output does not depend on the number of threads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use super::effort::EncodeEffort;
use super::wavefront::{AbortOnUnwind, lock};
use super::{
    BoolEncoder, BoolSink, BranchCounts, CoeffProbs, EncodeError, I420Frame,
    InterSegmentationConfig, IntraUvMode, IntraYMode, MbCoeffs, MbEntropyCtx, MbPixels,
    SplitMvCandidate, TrellisRateTables, TrellisStrength, bool_bits, classify_segment,
    clear_skip_ctx, encode_mb_tokens, mb_luma_variance, pick_intra_mb_all, pick_mb_for_ref,
    ref_frame_tree_bits,
};
use crate::backoff::{SharedEntropy, SharedPixelLine};
use crate::dequant::MbDequantFactors;
use crate::frame::KeyframePlanes;
use crate::loop_filter::MAX_MB_SEGMENTS;
use crate::motion_comp::{RefFrame, ReferencePlanes};
use crate::motion_vector::{Mv, MvContexts};
use crate::near_mv::{InterMode, MbInfo};
use crate::reconstruct::{MbNeighbors, ReconstructedMb};

/// Speed controls of one inter-frame encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct InterSpeed {
    /// The rate-distortion search at this effort level; `None` is the
    /// original exhaustive search of the option-less entry points.
    pub(crate) effort: Option<EncodeEffort>,
    /// Thread budget (`0` / `1` = serial). Never changes the bytes.
    pub(crate) threads: usize,
    /// Fit the §13.4 token-probability updates to this encode's own
    /// token statistics and ship them (the effort-level search only — its
    /// decisions do not depend on the shipped table, so no second
    /// analysis pass is needed).
    pub(crate) fit_token_probs: bool,
}

/// §13.3 token decisions of one macroblock row, in coding order.
///
/// An entry with bit 15 set is a coefficient-tree decision: the flat
/// index of its probability in the `[4][8][3][11]` table in bits 0..=10
/// and the bit in bit 11. Any other entry is a decision at a fixed
/// probability (extra bits, sign): the probability in the low byte, the
/// bit in bit 8.
#[derive(Debug, Default, Clone)]
pub(super) struct TokenTape(Vec<u16>);

const TAPE_TREE: u16 = 0x8000;
const TAPE_TREE_BIT: u16 = 0x0800;
const TAPE_SLOT_MASK: u16 = 0x07FF;

impl BoolSink for TokenTape {
    #[inline]
    fn put(&mut self, prob: u8, value: bool) {
        self.0.push(prob as u16 | (value as u16) << 8);
    }

    #[inline]
    fn put_tree(&mut self, slot: u16, _prob: u8, value: bool) {
        self.0
            .push(TAPE_TREE | slot | if value { TAPE_TREE_BIT } else { 0 });
    }
}

impl TokenTape {
    /// Code the tape into `enc` with the coefficient probabilities
    /// `probs`.
    pub(super) fn replay(&self, enc: &mut BoolEncoder, probs: &CoeffProbs) {
        let flat = probs.as_flattened().as_flattened().as_flattened();
        for &d in &self.0 {
            if d & TAPE_TREE != 0 {
                enc.write_bool(flat[(d & TAPE_SLOT_MASK) as usize], d & TAPE_TREE_BIT != 0);
            } else {
                enc.write_bool(d as u8, d >> 8 != 0);
            }
        }
    }

    /// Add the tape's coefficient-tree decisions to `counts`.
    pub(super) fn tally(&self, counts: &mut BranchCounts) {
        let flat = counts
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut();
        for &d in &self.0 {
            if d & TAPE_TREE != 0 {
                let slot = &mut flat[(d & TAPE_SLOT_MASK) as usize];
                if d & TAPE_TREE_BIT != 0 {
                    slot.1 += 1;
                } else {
                    slot.0 += 1;
                }
            }
        }
    }
}

/// One available reference picture.
#[derive(Clone, Copy)]
pub(super) struct RefPicture<'a> {
    pub(super) frame: RefFrame,
    pub(super) planes: ReferencePlanes<'a>,
}

/// §10 segmentation context of a frame: the configuration, each
/// segment's dequantisation factors and its RD multiplier.
pub(super) struct SegmentContext<'a> {
    pub(super) config: &'a InterSegmentationConfig,
    pub(super) factors: [MbDequantFactors; MAX_MB_SEGMENTS],
    pub(super) lambdas: [f64; MAX_MB_SEGMENTS],
}

/// Frame-constant inputs of the per-macroblock search.
pub(super) struct InterSearch<'a> {
    pub(super) frame: &'a I420Frame<'a>,
    /// Available references in §16.2 tree order (`LAST` first).
    pub(super) refs: &'a [RefPicture<'a>],
    pub(super) factors: &'a MbDequantFactors,
    pub(super) lambda: f64,
    pub(super) segments: Option<SegmentContext<'a>>,
    pub(super) coeff_probs: &'a CoeffProbs,
    pub(super) rate_tables: Option<&'a TrellisRateTables>,
    pub(super) trellis_strength: TrellisStrength,
    pub(super) mv_contexts: &'a MvContexts,
    /// §18.3 tap set of the frame's version.
    pub(super) filters: &'a [[i32; 6]; 8],
    /// Also consider the §11 whole-block intra modes.
    pub(super) pick_intra: bool,
    /// `None` selects the original exhaustive search.
    pub(super) effort: Option<EncodeEffort>,
    /// §17.2 differential costs (the effort-level search only).
    pub(super) mv_costs: Option<super::inter_search::MvCostTable>,
    pub(super) mb_cols: usize,
    pub(super) mb_rows: usize,
}

/// What the search decided for one macroblock.
pub(super) struct MbDecision {
    pub(super) is_intra: bool,
    /// §11.2 / §11.4 modes of an intra macroblock (`Dc` otherwise).
    pub(super) y_mode: IntraYMode,
    pub(super) uv_mode: IntraUvMode,
    pub(super) ref_frame: RefFrame,
    pub(super) mode: InterMode,
    pub(super) mv: Mv,
    pub(super) split: Option<SplitMvCandidate>,
    /// Quantised coefficients (raster order within each block).
    pub(super) coeffs: MbCoeffs,
    /// §11.1 `mb_skip_coeff`.
    pub(super) skip: bool,
    /// The macroblock codes no Y2 block (`SPLITMV`).
    pub(super) no_y2: bool,
    /// The pixels a decoder reconstructs, before the loop filter.
    pub(super) recon: ReconstructedMb,
}

/// The per-macroblock record the frame driver keeps (the decision without
/// its coefficients and pixels).
pub(super) struct MbRecord {
    pub(super) segment_id: Option<u8>,
    pub(super) is_intra: bool,
    pub(super) y_mode: IntraYMode,
    pub(super) uv_mode: IntraUvMode,
    pub(super) ref_frame: RefFrame,
    pub(super) mode: InterMode,
    pub(super) mv: Mv,
    pub(super) split: Option<SplitMvCandidate>,
    pub(super) skip: bool,
    /// Any non-zero coefficient (what the §15 loop filter asks).
    pub(super) has_coeffs: bool,
}

/// Result of the analysis over a whole frame.
pub(super) struct InterAnalysis {
    /// Per-macroblock records, raster order.
    pub(super) records: Vec<MbRecord>,
    /// One token tape per macroblock row.
    pub(super) tapes: Vec<TokenTape>,
    /// Macroblocks per §10 segment.
    pub(super) segment_counts: [u32; MAX_MB_SEGMENTS],
}

/// What a finished macroblock leaves for the row below — lock-free, as
/// both rows touch it once per macroblock (see `backoff.rs`).
struct RowBottom {
    pixels: SharedPixelLine,
    infos: SharedMbInfos,
    entropy: SharedEntropy,
}

/// One §16.3 neighbour record ([`MbInfo`]) per macroblock column, shared
/// like [`crate::backoff::SharedBytes`]: nine words per macroblock — the
/// reference, flags and whole-block vector, then the sixteen sub-block
/// vectors two to a word.
struct SharedMbInfos {
    words: Vec<AtomicU64>,
}

impl SharedMbInfos {
    const WORDS: usize = 9;

    fn new(mb_cols: usize) -> Self {
        SharedMbInfos {
            words: (0..mb_cols * Self::WORDS)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    fn pack(mv: Mv) -> u64 {
        (mv.row as u16 as u64) | (mv.col as u16 as u64) << 16
    }

    fn unpack(bits: u64) -> Mv {
        Mv {
            row: bits as u16 as i16,
            col: (bits >> 16) as u16 as i16,
        }
    }

    fn store(&self, mb_col: usize, info: &MbInfo) {
        let words = &self.words[mb_col * Self::WORDS..(mb_col + 1) * Self::WORDS];
        let reference = match info.ref_frame {
            None => 0u64,
            Some(RefFrame::Last) => 1,
            Some(RefFrame::Golden) => 2,
            Some(RefFrame::AltRef) => 3,
        };
        let head = Self::pack(info.mv)
            | reference << 32
            | (info.is_split as u64) << 34
            | (info.split_mvs.is_some() as u64) << 35;
        words[0].store(head, Ordering::Relaxed);
        if let Some(mvs) = &info.split_mvs {
            for (word, pair) in words[1..].iter().zip(mvs.as_chunks::<2>().0) {
                word.store(
                    Self::pack(pair[0]) | Self::pack(pair[1]) << 32,
                    Ordering::Relaxed,
                );
            }
        }
    }

    fn load(&self, mb_col: usize) -> MbInfo {
        let words = &self.words[mb_col * Self::WORDS..(mb_col + 1) * Self::WORDS];
        let head = words[0].load(Ordering::Relaxed);
        let split_mvs = (head >> 35 & 1 != 0).then(|| {
            let mut mvs = [Mv::default(); 16];
            for (pair, word) in mvs.as_chunks_mut::<2>().0.iter_mut().zip(&words[1..]) {
                let bits = word.load(Ordering::Relaxed);
                *pair = [Self::unpack(bits), Self::unpack(bits >> 32)];
            }
            mvs
        });
        MbInfo {
            ref_frame: match head >> 32 & 3 {
                0 => None,
                1 => Some(RefFrame::Last),
                2 => Some(RefFrame::Golden),
                _ => Some(RefFrame::AltRef),
            },
            mv: Self::unpack(head),
            is_split: head >> 34 & 1 != 0,
            split_mvs,
        }
    }
}

struct RowOutput {
    records: Vec<MbRecord>,
    tape: TokenTape,
    segment_counts: [u32; MAX_MB_SEGMENTS],
}

struct RowShared {
    /// Valid for macroblock columns `< done`.
    bottom: RowBottom,
    done: AtomicUsize,
    output: Mutex<Option<RowOutput>>,
}

struct RowJob<'p> {
    mb_row: usize,
    y: &'p mut [u8],
    u: &'p mut [u8],
    v: &'p mut [u8],
}

/// Analyse every macroblock of an inter frame (writing the unfiltered
/// reconstruction into `planes`) on up to `threads` threads.
pub(super) fn analyse_inter_frame(
    search: &InterSearch,
    planes: &mut KeyframePlanes,
    threads: usize,
) -> Result<InterAnalysis, EncodeError> {
    let (mb_cols, mb_rows) = (search.mb_cols, search.mb_rows);
    let shared: Vec<RowShared> = (0..mb_rows)
        .map(|_| RowShared {
            bottom: RowBottom {
                pixels: SharedPixelLine::new(mb_cols),
                infos: SharedMbInfos::new(mb_cols),
                entropy: SharedEntropy::new(mb_cols),
            },
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
    let mut records = Vec::with_capacity(mb_rows * mb_cols);
    let mut tapes = Vec::with_capacity(mb_rows);
    let mut segment_counts = [0u32; MAX_MB_SEGMENTS];
    for row in shared {
        let output = row
            .output
            .into_inner()
            .unwrap_or_else(|p| p.into_inner())
            .expect("every row was analysed");
        records.extend(output.records);
        tapes.push(output.tape);
        for (total, n) in segment_counts.iter_mut().zip(output.segment_counts) {
            *total += n;
        }
    }
    Ok(InterAnalysis {
        records,
        tapes,
        segment_counts,
    })
}

/// Analyse one macroblock row, left to right.
fn analyse_row(
    search: &InterSearch,
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
    let mut records = Vec::with_capacity(mb_cols);
    let mut tape = TokenTape(Vec::with_capacity(mb_cols * 64));
    let mut segment_counts = [0u32; MAX_MB_SEGMENTS];
    let mut left_ctx = MbEntropyCtx::default();
    let mut left: Option<ReconstructedMb> = None;
    let mut left_info = MbInfo::border();
    let mut aboveleft_info = MbInfo::border();
    let shape = search.effort.map(EncodeEffort::inter_shape);

    for mb_col in 0..mb_cols {
        let mut neighbors = MbNeighbors::default();
        let mut above_info = MbInfo::border();
        let mut above_ctx = MbEntropyCtx::default();
        if let Some(above) = above_row {
            let need = (mb_col + 2).min(mb_cols);
            let mut backoff = crate::backoff::Backoff::new();
            while above.done.load(Ordering::Acquire) < need {
                if abort.load(Ordering::Acquire) {
                    return Ok(());
                }
                backoff.pause();
            }
            let bottom = &above.bottom;
            above_info = bottom.infos.load(mb_col);
            above_ctx = bottom.entropy.load(mb_col);
            if search.pick_intra {
                bottom.pixels.load_above(mb_col, mb_cols, &mut neighbors);
            }
        }
        if search.pick_intra
            && let Some(prev) = &left
        {
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
        // §10 segment: source-luma variance against the configured
        // boundaries; the segment's quantiser and RD multiplier.
        let segment_id = search.segments.as_ref().map(|seg| {
            let sid =
                classify_segment(mb_luma_variance(&pixels.y), &seg.config.variance_boundaries);
            segment_counts[sid as usize] += 1;
            sid
        });
        let (factors, lambda) = match (&search.segments, segment_id) {
            (Some(seg), Some(sid)) => (&seg.factors[sid as usize], seg.lambdas[sid as usize]),
            _ => (search.factors, search.lambda),
        };
        let place = MbPlace {
            mb_row,
            mb_col,
            above: &above_info,
            left: &left_info,
            aboveleft: &aboveleft_info,
            neighbors: &neighbors,
            factors,
            lambda,
        };
        let decision = match &shape {
            None => decide_mb_exhaustive(search, &pixels, &place),
            Some(shape) => super::inter_search::decide_mb_rd(search, &pixels, &place, shape),
        };

        let (yx, cx) = (mb_col * 16, mb_col * 8);
        let recon = &decision.recon;
        for r in 0..16 {
            let dst = r * y_stride + yx;
            job.y[dst..dst + 16].copy_from_slice(&recon.y[r * 16..r * 16 + 16]);
        }
        for r in 0..8 {
            let dst = r * uv_stride + cx;
            job.u[dst..dst + 8].copy_from_slice(&recon.u[r * 8..r * 8 + 8]);
            job.v[dst..dst + 8].copy_from_slice(&recon.v[r * 8..r * 8 + 8]);
        }

        // §13.3 token decisions. A skip macroblock (§13.1) emits none but
        // clears its predictor slots.
        if decision.skip {
            clear_skip_ctx(decision.no_y2, &mut above_ctx, &mut left_ctx);
        } else {
            encode_mb_tokens(
                &mut tape,
                &decision.coeffs,
                decision.no_y2,
                search.coeff_probs,
                &mut above_ctx,
                &mut left_ctx,
            )
            .map_err(EncodeError::Token)?;
        }

        // The record the §16.3 census of later macroblocks reads. An
        // intra macroblock counts like the frame border.
        let info = if decision.is_intra {
            MbInfo {
                ref_frame: None,
                mv: Mv::default(),
                is_split: false,
                split_mvs: None,
            }
        } else if let Some(cand) = &decision.split {
            MbInfo {
                ref_frame: Some(decision.ref_frame),
                mv: cand.split_mvs[15],
                is_split: true,
                split_mvs: Some(cand.split_mvs),
            }
        } else {
            MbInfo {
                ref_frame: Some(decision.ref_frame),
                mv: decision.mv,
                is_split: false,
                split_mvs: None,
            }
        };
        this_row.bottom.infos.store(mb_col, &info);
        this_row.bottom.entropy.store(mb_col, &above_ctx);
        this_row.bottom.pixels.store(mb_col, recon);
        this_row.done.store(mb_col + 1, Ordering::Release);

        records.push(MbRecord {
            segment_id,
            is_intra: decision.is_intra,
            y_mode: decision.y_mode,
            uv_mode: decision.uv_mode,
            ref_frame: decision.ref_frame,
            mode: decision.mode,
            mv: decision.mv,
            split: decision.split,
            skip: decision.skip,
            has_coeffs: crate::loop_filter::mb_has_coeffs(&decision.coeffs),
        });
        aboveleft_info = above_info;
        left_info = info;
        left = Some(decision.recon);
    }

    *lock(&this_row.output) = Some(RowOutput {
        records,
        tape,
        segment_counts,
    });
    Ok(())
}

/// Where a macroblock sits and what its neighbours left behind.
pub(super) struct MbPlace<'a> {
    pub(super) mb_row: usize,
    pub(super) mb_col: usize,
    /// §16.3 neighbour records.
    pub(super) above: &'a MbInfo,
    pub(super) left: &'a MbInfo,
    pub(super) aboveleft: &'a MbInfo,
    /// Reconstructed neighbour pixels (filled only when the search may
    /// pick intra).
    pub(super) neighbors: &'a MbNeighbors,
    /// The macroblock's (segment's) dequantisation factors and RD
    /// multiplier.
    pub(super) factors: &'a MbDequantFactors,
    pub(super) lambda: f64,
}

/// The exhaustive search: every available reference runs the full
/// per-macroblock picker (motion search, the four whole-block modes, all
/// `SPLITMV` partitions) and is transformed, quantised and reconstructed;
/// the references — and, with `pick_intra`, the sixteen whole-block intra
/// mode pairs — are compared on prediction error plus signalling bits.
fn decide_mb_exhaustive(search: &InterSearch, pixels: &MbPixels, place: &MbPlace) -> MbDecision {
    // The §16.2 reference selector and the intra flag are priced at even
    // odds during the search; the probabilities that ship are fitted to
    // the decisions afterwards.
    const PROB_PICK: u8 = 128;
    let mut best: Option<super::PickedMbForRef> = None;
    let mut best_total = f64::INFINITY;
    for reference in search.refs {
        let luma_ref = crate::motion_search::LumaRef {
            plane: reference.planes.y,
            stride: reference.planes.y_stride,
            width: reference.planes.mb_cols * 16,
            height: reference.planes.mb_rows * 16,
        };
        let pick = pick_mb_for_ref(
            reference.frame,
            &reference.planes,
            luma_ref,
            pixels,
            place.above,
            place.left,
            place.aboveleft,
            place.mb_col,
            place.mb_row,
            search.mb_cols,
            search.mb_rows,
            search.mv_contexts,
            place.lambda,
            search.filters,
            place.factors,
            Some(search.coeff_probs),
            search.rate_tables,
            search.trellis_strength,
        );
        let total =
            pick.j + place.lambda * ref_frame_tree_bits(reference.frame, PROB_PICK, PROB_PICK);
        if total < best_total {
            best_total = total;
            best = Some(pick);
        }
    }
    let chosen = best.expect("at least one reference (LAST) was scored");
    let mut decision = MbDecision {
        is_intra: false,
        y_mode: IntraYMode::Dc,
        uv_mode: IntraUvMode::Dc,
        ref_frame: chosen.ref_frame,
        mode: chosen.chosen_mode,
        mv: chosen.chosen_mv,
        split: chosen.chosen_split,
        coeffs: chosen.raw_coeffs,
        skip: chosen.mb_skip_coeff,
        no_y2: chosen.use_bpred,
        recon: chosen.recon,
    };

    if search.pick_intra {
        let intra = pick_intra_mb_all(
            pixels,
            place.neighbors,
            place.factors,
            place.lambda,
            Some(search.coeff_probs),
            search.rate_tables,
            search.trellis_strength,
        );
        let inter_total = chosen.j
            + place.lambda * ref_frame_tree_bits(decision.ref_frame, PROB_PICK, PROB_PICK)
            + place.lambda * bool_bits(PROB_PICK, true);
        let intra_total = intra.j + place.lambda * bool_bits(PROB_PICK, false);
        if intra_total < inter_total {
            decision.is_intra = true;
            decision.y_mode = intra.y_mode;
            decision.uv_mode = intra.uv_mode;
            decision.mv = Mv::default();
            decision.split = None;
            decision.coeffs = intra.raw_coeffs;
            decision.skip = intra.mb_skip_coeff;
            decision.no_y2 = false;
            decision.recon = intra.recon;
        }
    }
    decision
}
