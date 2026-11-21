//! Search-effort levels for the intra (key-frame) encoder and the
//! model-driven per-macroblock search the reduced levels run.
//!
//! RFC 6386 specifies decoding only; how much work an encoder spends
//! choosing among the §12 prediction modes is its own business. The
//! exhaustive search ([`EncodeEffort::Thorough`]) carries
//! every candidate mode through transform, §13 trellis quantisation,
//! reconstruction and exact token pricing. The reduced levels replace
//! that per-candidate chain with a closed-form cost model:
//!
//! * **distortion** — the squared quantisation error of each transform
//!   coefficient under plain rounding, weighted by the energy its basis
//!   vector carries into the pixel domain (the same weights the trellis
//!   uses for 4×4 DCT blocks; a flat `1/16` for the Y2 block, whose
//!   coefficients each spread over all 256 luma pixels through the
//!   inverse WHT and the DC-only inverse DCT);
//! * **rate** — the §13.3 token bits of the rounded levels, read from the
//!   per-frame rate tables along the real context chain.
//!
//! Only the winning mode is then quantised for real (through the trellis
//! where the level keeps it). The fastest levels additionally shortlist
//! candidates by the squared prediction error before the model sees them.
//!
//! Every level produces a conformant stream; for a given level the bytes
//! do not depend on the thread budget.

use super::{
    BMODE_CANDIDATES, BlockType, COEFF_BANDS, ENC_BPRED_ORIGIN, ENC_BPRED_STRIDE,
    IDCT_BASIS_NORM_SQ, IntraBmode, IntraUvMode, IntraYMode, KF_BMODE_BITS, KF_UV_MODE_PROB,
    KF_YMODE_PATHS, KF_YMODE_PROB, MbAnalysis, MbCoeffs, MbPixels, MbRdCtx, TrellisBlockSpec,
    TrellisRateTables, TrellisStrength, UV_MODE_PATHS, block_ssd, build_enc_bpred_buffer,
    forward_dct_4x4, forward_wht_4x4, predict_b4x4, predict_uv8x8, predict_y16x16,
    reconstruct_block_4x4, rounded_level, transform_chroma_plane_rd, transform_whole_block_luma_rd,
    treed_bits_cached, trellis_quantize_block,
};
use crate::coded_header::TokenProbUpdates;
use crate::intra_predict::DEFAULT_ABOVE_PIXEL;
use crate::reconstruct::MbNeighbors;

/// How much mode search the key-frame encoder spends per macroblock.
///
/// The levels trade output size for encode time; all of them emit a
/// conformant RFC 6386 key frame that this crate's decoder (and any
/// other) reconstructs to exactly the picture the encoder predicted
/// from. [`Balanced`](Self::Balanced) is the default;
/// [`Thorough`](Self::Thorough) is the search of the option-less entry
/// points. See the crate README for the measured time / size / quality
/// of each level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum EncodeEffort {
    /// Level 0. Modes are shortlisted by squared prediction error (the
    /// best two of the ten §12.3 sub-block modes, the single best §12.2
    /// luma and chroma mode); the survivors are compared with the cost
    /// model; coefficients are plain-rounded — the §13 trellis is off
    /// regardless of [`TrellisStrength`]. `B_PRED` is not searched for a
    /// macroblock whose best whole-block mode leaves no luma AC residual.
    Fastest,
    /// Level 1. As [`Fastest`](Self::Fastest) but the cost model sees the
    /// best four sub-block modes and all §12.2 luma / chroma modes, and
    /// the winners are trellis-quantised.
    Fast,
    /// Level 2 (default). Every candidate mode is compared with the cost
    /// model; only the winners go through the trellis and
    /// reconstruction. About half the time of
    /// [`Thorough`](Self::Thorough) for the same picture quality within
    /// a percent of its size.
    #[default]
    Balanced,
    /// Level 3. Exhaustive rate-distortion search: every candidate is
    /// transformed, trellis-quantised, reconstructed and priced exactly
    /// — the search of the option-less entry points.
    Thorough,
}

impl EncodeEffort {
    /// The numeric level, `0` (fastest) ..= `3` (thorough).
    pub const fn level(self) -> u8 {
        match self {
            EncodeEffort::Fastest => 0,
            EncodeEffort::Fast => 1,
            EncodeEffort::Balanced => 2,
            EncodeEffort::Thorough => 3,
        }
    }

    /// The level for a number: `0`, `1`, `2`, and [`Thorough`](Self::Thorough)
    /// for `3` and anything above.
    pub const fn from_level(level: u8) -> Self {
        match level {
            0 => EncodeEffort::Fastest,
            1 => EncodeEffort::Fast,
            2 => EncodeEffort::Balanced,
            _ => EncodeEffort::Thorough,
        }
    }

    /// The level's lower-case name (`"fastest"`, `"fast"`, `"balanced"`,
    /// `"thorough"`) — the spelling the registry `effort` option accepts.
    pub const fn name(self) -> &'static str {
        match self {
            EncodeEffort::Fastest => "fastest",
            EncodeEffort::Fast => "fast",
            EncodeEffort::Balanced => "balanced",
            EncodeEffort::Thorough => "thorough",
        }
    }

    /// Parse a level from its [`name`](Self::name) or its decimal number.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "fastest" | "0" => Some(EncodeEffort::Fastest),
            "fast" | "1" => Some(EncodeEffort::Fast),
            "balanced" | "2" => Some(EncodeEffort::Balanced),
            "thorough" | "3" => Some(EncodeEffort::Thorough),
            _ => None,
        }
    }

    /// All levels, fastest first.
    pub const ALL: [EncodeEffort; 4] = [
        EncodeEffort::Fastest,
        EncodeEffort::Fast,
        EncodeEffort::Balanced,
        EncodeEffort::Thorough,
    ];

    /// The model-search shape of a reduced level (`None` for
    /// [`Thorough`](Self::Thorough), which does not use the model).
    fn model_search(self) -> Option<ModelSearch> {
        match self {
            EncodeEffort::Fastest => Some(ModelSearch {
                bmode_shortlist: 2,
                whole_modes_by_prediction_error: true,
                skip_bpred_without_luma_ac: true,
                trellis: false,
            }),
            EncodeEffort::Fast => Some(ModelSearch {
                bmode_shortlist: 4,
                whole_modes_by_prediction_error: false,
                skip_bpred_without_luma_ac: true,
                trellis: true,
            }),
            EncodeEffort::Balanced => Some(ModelSearch {
                bmode_shortlist: BMODE_CANDIDATES.len(),
                whole_modes_by_prediction_error: false,
                skip_bpred_without_luma_ac: false,
                trellis: true,
            }),
            EncodeEffort::Thorough => None,
        }
    }
}

/// Controls for one key-frame encode — the options the
/// `encode_keyframe_with_options` family takes next to
/// [`KeyframeParams`](super::KeyframeParams).
///
/// `Default` is the recommended still-picture encode: the default
/// [`EncodeEffort`], serial, the loop filter of the `KeyframeParams`,
/// and the §13.4 token probabilities fitted to the picture within the
/// one search ([`token_prob_fit`](Self::token_prob_fit)).
/// [`KeyframeEncodeOptions::exhaustive`] is the option-less
/// [`encode_keyframe`](super::encode_keyframe), byte for byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct KeyframeEncodeOptions {
    /// Mode-search effort.
    pub effort: EncodeEffort,
    /// Thread budget for the encode. `0` and `1` both mean serial (the
    /// default); with `n > 1` the macroblock analysis runs as a
    /// macroblock-row wavefront on up to `n` threads (scoped
    /// `std::thread`s that end with the call). The output bytes are the
    /// same for every budget.
    pub threads: usize,
    /// Measure every §9.4 loop-filter level / sharpness candidate on the
    /// reconstruction and keep the best (ignoring
    /// `KeyframeParams::loop_filter_level`), as
    /// [`encode_keyframe_auto_loop_filter`](super::encode_keyframe_auto_loop_filter) does.
    pub auto_loop_filter: bool,
    /// Run the mode search a second time against the token probabilities
    /// fitted to the first search, and keep the smaller stream (roughly
    /// doubles the encode time for a further few tenths of a percent;
    /// [`encode_keyframe_with_fitted_token_prob_updates`](super::encode_keyframe_with_fitted_token_prob_updates)
    /// is this with the exhaustive search).
    pub fitted_token_prob_updates: bool,
    /// Fit the §13.4 token-probability updates to the token statistics
    /// of the search itself and code the tokens against the fitted
    /// table (on by default). The search records its token decisions
    /// instead of bits, so this costs one walk over the recorded
    /// decisions — no second search — and never enlarges the stream:
    /// an update ships only where it saves more bits than it costs.
    pub token_prob_fit: bool,
    /// §13.4 updates the mode search prices tokens against instead of
    /// the §13.5 defaults — typically the
    /// [`KeyframeEncoded::token_prob_updates`] a similar picture (the
    /// previous frame, a neighbouring tile, a first quick encode) came
    /// out with. With [`token_prob_fit`](Self::token_prob_fit) they only
    /// steer the search and the stream carries this picture's own fit;
    /// without it they are shipped as given.
    pub search_token_probs: Option<TokenProbUpdates>,
    /// Choose the §9.4 loop-filter level and sharpness on a sample of
    /// the picture (three to six bands of macroblock rows spread over
    /// its height, searched coarse to fine), ignoring
    /// `KeyframeParams::loop_filter_level` — most of the gain of
    /// [`auto_loop_filter`](Self::auto_loop_filter) for a small fraction
    /// of its time. `auto_loop_filter` takes precedence when both are
    /// set.
    pub sampled_loop_filter: bool,
}

impl Default for KeyframeEncodeOptions {
    fn default() -> Self {
        KeyframeEncodeOptions {
            effort: EncodeEffort::default(),
            threads: 0,
            auto_loop_filter: false,
            fitted_token_prob_updates: false,
            token_prob_fit: true,
            search_token_probs: None,
            sampled_loop_filter: false,
        }
    }
}

impl KeyframeEncodeOptions {
    /// The defaults: the default [`EncodeEffort`], serial, token
    /// probabilities fitted in the one search.
    pub fn new() -> Self {
        Self::default()
    }

    /// The exhaustive encode of the option-less entry points:
    /// [`EncodeEffort::Thorough`], serial, the §13.5 default token
    /// probabilities — the bytes of
    /// [`encode_keyframe`](super::encode_keyframe).
    pub fn exhaustive() -> Self {
        Self::default()
            .with_effort(EncodeEffort::Thorough)
            .with_token_prob_fit(false)
    }

    /// Set `effort`.
    #[must_use]
    pub fn with_effort(mut self, v: EncodeEffort) -> Self {
        self.effort = v;
        self
    }

    /// Set `threads`.
    #[must_use]
    pub fn with_threads(mut self, v: usize) -> Self {
        self.threads = v;
        self
    }

    /// Set `auto_loop_filter`.
    #[must_use]
    pub fn with_auto_loop_filter(mut self, v: bool) -> Self {
        self.auto_loop_filter = v;
        self
    }

    /// Set `fitted_token_prob_updates`.
    #[must_use]
    pub fn with_fitted_token_prob_updates(mut self, v: bool) -> Self {
        self.fitted_token_prob_updates = v;
        self
    }

    /// Set `token_prob_fit`.
    #[must_use]
    pub fn with_token_prob_fit(mut self, v: bool) -> Self {
        self.token_prob_fit = v;
        self
    }

    /// Set `search_token_probs`.
    #[must_use]
    pub fn with_search_token_probs(mut self, v: Option<TokenProbUpdates>) -> Self {
        self.search_token_probs = v;
        self
    }

    /// Set `sampled_loop_filter`.
    #[must_use]
    pub fn with_sampled_loop_filter(mut self, v: bool) -> Self {
        self.sampled_loop_filter = v;
        self
    }
}

/// A finished key-frame encode with what the encoder settled on along
/// the way — the result of
/// [`encode_keyframe_detailed`](super::encode_keyframe_detailed).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KeyframeEncoded {
    /// The VP8 key frame.
    pub bytes: Vec<u8>,
    /// The macroblock-aligned post-§15 reconstruction — the planes a
    /// decoder rebuilds from `bytes`.
    pub reconstruction: crate::frame::KeyframePlanes,
    /// The §13.4 token-probability updates the frame carries (`None`
    /// when it codes against the §13.5 defaults). Feed them to
    /// [`KeyframeEncodeOptions::with_search_token_probs`] for the next
    /// similar picture.
    pub token_prob_updates: Option<TokenProbUpdates>,
    /// The §9.4 `loop_filter_level` in the frame header.
    pub loop_filter_level: u8,
    /// The §9.4 `sharpness_level` in the frame header.
    pub sharpness_level: u8,
}

/// Shape of the model-driven search at one reduced effort level.
#[derive(Clone, Copy)]
struct ModelSearch {
    /// How many §12.3 sub-block modes (ranked by squared prediction
    /// error plus mode-signal cost) the cost model compares; all ten
    /// when equal to the candidate count.
    bmode_shortlist: usize,
    /// Choose the §12.2 luma and chroma modes by squared prediction
    /// error (plus mode-signal cost) instead of the cost model.
    whole_modes_by_prediction_error: bool,
    /// Do not search `B_PRED` when the whole-block winner quantises to
    /// no luma AC coefficient.
    skip_bpred_without_luma_ac: bool,
    /// Trellis-quantise the winners (else plain rounding).
    trellis: bool,
}

/// Fixed-point scale of the model's distortion weights.
pub(super) const WEIGHT_SCALE: f64 = 4096.0;

/// The trellis's inverse-DCT basis energies ([`IDCT_BASIS_NORM_SQ`], by
/// scan position) in `1/4096` fixed point, so the model's distortion is
/// an exact integer sum.
pub(super) static DCT_WEIGHTS: std::sync::LazyLock<[u64; 16]> = std::sync::LazyLock::new(|| {
    let norms = &*IDCT_BASIS_NORM_SQ;
    core::array::from_fn(|k| (norms[k] * WEIGHT_SCALE).round() as u64)
});

/// Pixel-domain energy one unit of error in a Y2 (WHT) coefficient
/// injects, in `1/4096` fixed point: the inverse WHT moves each of the
/// sixteen sub-block DCs by `1/8`, and a DC-only inverse DCT moves each
/// of that sub-block's sixteen pixels by a further `1/8` — 256 pixels at
/// `1/64` each, i.e. `1/16` per unit squared.
pub(super) const Y2_WEIGHTS: [u64; 16] = [(WEIGHT_SCALE / 16.0) as u64; 16];

/// What the cost model says about one plain-rounded block.
pub(super) struct BlockModel {
    /// Weighted squared rounding error of the coded positions, in
    /// `1/4096` units ([`WEIGHT_SCALE`]).
    pub(super) dist: u64,
    /// The same error with every level forced to zero.
    pub(super) dist_zero: u64,
    /// §13.3 token bits of the rounded levels, through the end-of-block
    /// token.
    pub(super) rate: f64,
    /// Whether any level is non-zero.
    pub(super) nonzero: bool,
}

/// Plain-round one raster-order coefficient block in place and model its
/// cost: the weighted squared rounding error over the coded positions
/// (`first_coeff..16`) and the §13.3 token bits of the rounded levels
/// from `spec.entry_ctx3` through the end-of-block token. The levels
/// written are exactly what [`enc_quantize_block`](super::enc_quantize_block)
/// produces (positions below `first_coeff` are zeroed).
pub(super) fn model_block_parts(
    block: &mut [i16; 16],
    spec: TrellisBlockSpec,
    weights: &[u64; 16],
    tables: &TrellisRateTables,
) -> BlockModel {
    let first = spec.first_coeff;
    debug_assert!(first <= 1, "§13.3 blocks start at coefficient 0 or 1");
    let (dc_q, ac_q) = (spec.dc as u32, spec.ac as u32);
    // Magnitudes in scan order (an uncoded position 0 counts as zero),
    // the all-zero distortion, and whether any coded position rounds to
    // a level (`|x| + q/2 >= q`, i.e. `|x| >= q − q/2`). All-zero is the
    // common case on inter residue and is settled here without a
    // division.
    let zigzag = &crate::dct_tokens::ZIGZAG;
    let mut magnitude = [0u32; 16];
    for (m, &pos) in magnitude.iter_mut().zip(zigzag.iter()) {
        *m = block[pos].unsigned_abs() as u32;
    }
    if first != 0 {
        magnitude[0] = 0;
    }
    let mut thresholds = [ac_q - ac_q / 2; 16];
    thresholds[0] = dc_q - dc_q / 2;
    let mut dist_zero = 0u64;
    let mut any_level = false;
    for k in 0..16 {
        let ax = magnitude[k];
        dist_zero += weights[k] * (ax as u64 * ax as u64);
        any_level |= ax >= thresholds[k];
    }
    let mut ctx3 = spec.entry_ctx3;
    if !any_level {
        let rate = tables.eob[spec.plane][COEFF_BANDS[first]][ctx3];
        *block = [0i16; 16];
        return BlockModel {
            dist: dist_zero,
            dist_zero,
            rate,
            nonzero: false,
        };
    }
    let mut levels = [0u32; 16];
    let mut negative = 0u32;
    let mut coded = 0u32;
    let mut dist = 0u64;
    for k in first..16 {
        let ax = magnitude[k];
        let q = if k == 0 { dc_q } else { ac_q };
        let level = rounded_level(ax, q);
        let err = (ax as i64 - level as i64 * q as i64).unsigned_abs();
        dist += weights[k] * err * err;
        levels[k] = level;
        negative |= ((block[crate::dct_tokens::ZIGZAG[k]] < 0) as u32) << k;
        coded |= ((level != 0) as u32) << k;
    }
    *block = [0i16; 16];
    let small = &tables.small[spec.plane];
    debug_assert!(coded != 0, "a level at or past half a step rounds up");
    let end = 32 - coded.leading_zeros() as usize;

    let mut rate = 0.0;
    let mut prev_was_zero = 0usize;
    for k in first..end {
        let level = levels[k];
        let band = COEFF_BANDS[k];
        rate += if level <= super::TRELLIS_SMALL_LEVEL_MAX as u32 {
            small[band][ctx3][prev_was_zero][level as usize]
        } else {
            TrellisRateTables::level_rate(
                &small[band],
                &tables.cat_path[spec.plane][band],
                ctx3,
                prev_was_zero == 1,
                level,
            )
        };
        ctx3 = (level as usize).min(2);
        prev_was_zero = (level == 0) as usize;
        let signed = level as i16;
        block[crate::dct_tokens::ZIGZAG[k]] = if (negative >> k) & 1 != 0 {
            -signed
        } else {
            signed
        };
    }
    if end < 16 {
        rate += tables.eob[spec.plane][COEFF_BANDS[end]][ctx3];
    }
    BlockModel {
        dist,
        dist_zero,
        rate,
        nonzero: true,
    }
}

/// The cost `D + lambda·R` [`model_block_parts`] models for a block and
/// whether any level is non-zero, without writing the levels — the form
/// the key-frame search compares candidates with.
///
/// Same model, evaluated through the block's sparsity: a position whose
/// magnitude rounds to zero contributes its full squared magnitude to
/// the distortion and a `DCT_0` token to the rate, so the all-zero
/// totals are formed first (one branch-free pass that also marks the
/// positions that round to a level), and only the marked positions are
/// then visited to exchange their zero terms for the real ones. The
/// zeros between two levels are priced from the per-plane run table.
/// (The rate is summed in a different order than `model_block_parts`
/// sums it, so the two agree to rounding, not bit for bit.)
#[inline]
pub(super) fn model_block_cost(
    block: &[i16; 16],
    spec: TrellisBlockSpec,
    weights: &[u64; 16],
    tables: &TrellisRateTables,
    lambda: f64,
) -> (f64, bool) {
    let first = spec.first_coeff;
    debug_assert!(first <= 1, "§13.3 blocks start at coefficient 0 or 1");
    let (dc_q, ac_q) = (spec.dc as u32, spec.ac as u32);
    let mut magnitude = [0u32; 16];
    for (m, &pos) in magnitude.iter_mut().zip(crate::dct_tokens::ZIGZAG.iter()) {
        *m = block[pos].unsigned_abs() as u32;
    }
    if first != 0 {
        magnitude[0] = 0;
    }
    // `|x| + q/2 >= q` ⇔ the magnitude rounds to a level.
    let mut thresholds = [ac_q - ac_q / 2; 16];
    thresholds[0] = dc_q - dc_q / 2;
    let mut dist = 0u64;
    let mut coded = 0u32;
    for k in 0..16 {
        let ax = magnitude[k];
        dist += weights[k] * (ax as u64 * ax as u64);
        coded |= ((ax >= thresholds[k]) as u32) << k;
    }
    let plane = spec.plane;
    let eob = &tables.eob[plane];
    let mut ctx3 = spec.entry_ctx3;
    if coded == 0 {
        let rate = eob[COEFF_BANDS[first]][ctx3];
        return (dist as f64 / WEIGHT_SCALE + lambda * rate, false);
    }
    let small = &tables.small[plane];
    let zero_run = &tables.zero_run[plane];
    let mut rate = 0.0;
    let mut pos = first;
    let mut after_zero = 0usize;
    let mut pending = coded;
    while pending != 0 {
        let k = pending.trailing_zeros() as usize;
        pending &= pending - 1;
        if k > pos {
            // The first zero of the run sees the carried context; the
            // rest follow a zero.
            rate += small[COEFF_BANDS[pos]][ctx3][after_zero][0];
            rate += zero_run[k] - zero_run[pos + 1];
            ctx3 = 0;
            after_zero = 1;
        }
        let ax = magnitude[k];
        let q = if k == 0 { dc_q } else { ac_q };
        let level = rounded_level(ax, q);
        let err = (ax as i64 - level as i64 * q as i64).unsigned_abs();
        // `|err| <= q/2 <= ax`, so the exchange never underflows.
        dist -= weights[k] * (ax as u64 * ax as u64 - err * err);
        let band = COEFF_BANDS[k];
        rate += if level <= super::TRELLIS_SMALL_LEVEL_MAX as u32 {
            small[band][ctx3][after_zero][level as usize]
        } else {
            TrellisRateTables::level_rate(
                &small[band],
                &tables.cat_path[plane][band],
                ctx3,
                after_zero == 1,
                level,
            )
        };
        ctx3 = (level as usize).min(2);
        after_zero = 0;
        pos = k + 1;
    }
    if pos < 16 {
        rate += eob[COEFF_BANDS[pos]][ctx3];
    }
    (dist as f64 / WEIGHT_SCALE + lambda * rate, true)
}

/// Forward-DCT the 4×4 residual `src − pred` at `(row, col)` of two
/// equally laid out `stride`-wide pixel blocks.
#[inline]
pub(super) fn residual_dct(
    src: &[u8],
    pred: &[u8],
    stride: usize,
    row: usize,
    col: usize,
) -> [i16; 16] {
    let mut residual = [0i16; 16];
    for r in 0..4 {
        let off = (row + r) * stride + col;
        for c in 0..4 {
            residual[r * 4 + c] = src[off + c] as i16 - pred[off + c] as i16;
        }
    }
    let mut coeffs = [0i16; 16];
    forward_dct_4x4(&residual, &mut coeffs);
    coeffs
}

/// Model cost of coding the luma plane with one §12.2 whole-block
/// prediction (mode-signal bits excluded), and whether any luma AC
/// level survives rounding — or `None` as soon as the running cost
/// reaches `budget` (block costs are non-negative, so the total could
/// only be at or above it: the candidate cannot win a strict comparison
/// against a cost of `budget`).
fn model_whole_luma(
    src: &[u8; 256],
    pred: &[u8; 256],
    rd: &MbRdCtx,
    tables: &TrellisRateTables,
    budget: f64,
) -> Option<(f64, bool)> {
    let factors = rd.factors;
    let dct_weights = &*DCT_WEIGHTS;
    let mut dcs = [0i16; 16];
    let mut above = [false; 4];
    let mut left = [false; 4];
    let mut cost = 0.0;
    let mut any_ac = false;
    for i in 0..4 {
        for j in 0..4 {
            let mut coeffs = residual_dct(src, pred, 16, i * 4, j * 4);
            dcs[i * 4 + j] = coeffs[0];
            coeffs[0] = 0;
            let (c, nonzero) = model_block_cost(
                &coeffs,
                TrellisBlockSpec {
                    dc: factors.y1_dc,
                    ac: factors.y1_ac,
                    first_coeff: BlockType::YAfterY2.first_coeff(),
                    entry_ctx3: above[j] as usize + left[i] as usize,
                    plane: BlockType::YAfterY2.plane_index(),
                },
                dct_weights,
                tables,
                rd.lambda,
            );
            cost += c;
            above[j] = nonzero;
            left[i] = nonzero;
            any_ac |= nonzero;
        }
        if cost >= budget {
            return None;
        }
    }
    let mut y2 = [0i16; 16];
    forward_wht_4x4(&dcs, &mut y2);
    let (c, _) = model_block_cost(
        &y2,
        TrellisBlockSpec {
            dc: factors.y2_dc,
            ac: factors.y2_ac,
            first_coeff: BlockType::Y2.first_coeff(),
            entry_ctx3: 0,
            plane: BlockType::Y2.plane_index(),
        },
        &Y2_WEIGHTS,
        tables,
        rd.lambda,
    );
    Some((cost + c, any_ac))
}

/// Model cost of coding one 8×8 chroma plane against `pred`.
fn model_chroma_plane(
    src: &[u8; 64],
    pred: &[u8; 64],
    rd: &MbRdCtx,
    tables: &TrellisRateTables,
) -> f64 {
    let factors = rd.factors;
    let dct_weights = &*DCT_WEIGHTS;
    let mut above = [false; 2];
    let mut left = [false; 2];
    let mut cost = 0.0;
    for (i, left_nonzero) in left.iter_mut().enumerate() {
        for (j, above_nonzero) in above.iter_mut().enumerate() {
            let coeffs = residual_dct(src, pred, 8, i * 4, j * 4);
            let (c, nonzero) = model_block_cost(
                &coeffs,
                TrellisBlockSpec {
                    dc: factors.uv_dc,
                    ac: factors.uv_ac,
                    first_coeff: BlockType::UV.first_coeff(),
                    entry_ctx3: *above_nonzero as usize + *left_nonzero as usize,
                    plane: BlockType::UV.plane_index(),
                },
                dct_weights,
                tables,
                rd.lambda,
            );
            cost += c;
            *above_nonzero = nonzero;
            *left_nonzero = nonzero;
        }
    }
    cost
}

/// The §11.3 sub-block-mode contexts a macroblock inherits: the modes
/// of the four sub-blocks directly above it and directly to its left
/// (a whole-block neighbour contributes the §11.3 projection of its
/// mode, a neighbour outside the frame `B_DC_PRED`) — the contexts the
/// mode layer codes the top-row and left-column sub-modes against.
pub(super) struct BmodeContext {
    pub(super) above: [IntraBmode; 4],
    pub(super) left: [IntraBmode; 4],
}

/// Outcome of the model-driven `B_PRED` pass.
struct ModelBpred {
    modes: [IntraBmode; 16],
    y_coeffs: [[i16; 16]; 16],
    cost: f64,
}

/// `B_PRED` with model-driven sub-mode choice. The sub-mode signal cost
/// is priced against the real §11.3 contexts, including the neighbouring
/// macroblocks' (`bmode_ctx`). Per sub-block the
/// candidate modes (all ten, or the `shortlist` best by squared
/// prediction error plus mode-signal cost) are compared by model cost;
/// the winner is then quantised for real against the §13.3 neighbour
/// context, reconstructed, and written back so later sub-blocks predict
/// from the pixels the decoder will hold. Returns `None` as soon as the
/// running cost reaches `budget` (sub-block costs are non-negative).
fn model_bpred_luma(
    src: &[u8; 256],
    neighbors: &MbNeighbors,
    bmode_ctx: &BmodeContext,
    rd: &MbRdCtx,
    tables: &TrellisRateTables,
    search: ModelSearch,
    budget: f64,
) -> Option<ModelBpred> {
    let factors = rd.factors;
    let mut buf = build_enc_bpred_buffer(neighbors);
    let mut modes = [IntraBmode::Dc; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut total = 0.0;
    let mut tok_above = [false; 4];
    let mut tok_left = [false; 4];
    let trellis = search.trellis && rd.trellis_active();
    let bmode_bits = &*KF_BMODE_BITS;
    let dct_weights = &*DCT_WEIGHTS;

    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let sb_base = ENC_BPRED_ORIGIN + (i * 4) * ENC_BPRED_STRIDE + (j * 4);
            let above_base = sb_base - ENC_BPRED_STRIDE;
            let mut above = [0u8; 8];
            above.copy_from_slice(&buf[above_base..above_base + 8]);
            let mut left = [0u8; 4];
            for (r, slot) in left.iter_mut().enumerate() {
                *slot = buf[sb_base + r * ENC_BPRED_STRIDE - 1];
            }
            let corner = buf[above_base - 1];
            let mut sb_src = [0u8; 16];
            for r in 0..4 {
                let off = (i * 4 + r) * 16 + j * 4;
                sb_src[r * 4..r * 4 + 4].copy_from_slice(&src[off..off + 4]);
            }
            let above_mode = if i == 0 {
                bmode_ctx.above[j]
            } else {
                modes[(i - 1) * 4 + j]
            };
            let left_mode = if j == 0 {
                bmode_ctx.left[i]
            } else {
                modes[i * 4 + (j - 1)]
            };
            let mode_bits = &bmode_bits[above_mode.idx()][left_mode.idx()];

            // Predictions, mode-signal cost, and (for a shortlist) the
            // squared-prediction-error ranking key of every sub-mode.
            let mut preds = [[0u8; 16]; 10];
            let mut mode_cost = [0f64; 10];
            let mut order = [0usize; 10];
            let mut rank = [0f64; 10];
            let shortlisted = search.bmode_shortlist < BMODE_CANDIDATES.len();
            for (n, &mode) in BMODE_CANDIDATES.iter().enumerate() {
                predict_b4x4(&mut preds[n], mode, &above, &left, corner);
                mode_cost[n] = rd.lambda * mode_bits[mode.idx()];
                order[n] = n;
                if shortlisted {
                    rank[n] = block_ssd(&preds[n], &sb_src) as f64 + mode_cost[n];
                }
            }
            let considered = if shortlisted {
                // Stable selection of the `bmode_shortlist` smallest keys
                // (ties keep `BMODE_CANDIDATES` order).
                for slot in 0..search.bmode_shortlist {
                    let mut best = slot;
                    for probe in slot + 1..order.len() {
                        if rank[order[probe]] < rank[order[best]] {
                            best = probe;
                        }
                    }
                    order[slot..=best].rotate_right(1);
                }
                search.bmode_shortlist
            } else {
                order.len()
            };

            let entry_ctx3 = tok_above[j] as usize + tok_left[i] as usize;
            let spec = TrellisBlockSpec {
                dc: factors.y1_dc,
                ac: factors.y1_ac,
                first_coeff: BlockType::YNoY2.first_coeff(),
                entry_ctx3,
                plane: BlockType::YNoY2.plane_index(),
            };
            let mut best_n = order[0];
            let mut best_cost = f64::INFINITY;
            let mut best_dct = [0i16; 16];
            for &n in &order[..considered] {
                let mut residual = [0i16; 16];
                for k in 0..16 {
                    residual[k] = sb_src[k] as i16 - preds[n][k] as i16;
                }
                let mut dct = [0i16; 16];
                forward_dct_4x4(&residual, &mut dct);
                let (c, _) = model_block_cost(&dct, spec, dct_weights, tables, rd.lambda);
                let cost = c + mode_cost[n];
                if cost < best_cost {
                    best_cost = cost;
                    best_n = n;
                    best_dct = dct;
                }
            }

            if total + best_cost >= budget {
                return None;
            }
            // Quantise the winner for real.
            let coeffs = if trellis {
                let mut c = best_dct;
                trellis_quantize_block(
                    &mut c,
                    spec,
                    rd.coeff_probs,
                    Some(tables),
                    rd.trellis_lambda(),
                );
                c
            } else {
                let mut c = best_dct;
                super::enc_quantize_block_range(&mut c, spec.dc, spec.ac, spec.first_coeff);
                c
            };
            let recon =
                reconstruct_block_4x4(&preds[best_n], &coeffs, factors.y1_dc, factors.y1_ac);
            modes[idx] = BMODE_CANDIDATES[best_n];
            y_coeffs[idx] = coeffs;
            total += best_cost;
            let nonzero = coeffs.iter().any(|&v| v != 0);
            tok_above[j] = nonzero;
            tok_left[i] = nonzero;
            for r in 0..4 {
                let dst = sb_base + r * ENC_BPRED_STRIDE;
                buf[dst..dst + 4].copy_from_slice(&recon[r * 4..r * 4 + 4]);
            }
        }
    }
    Some(ModelBpred {
        modes,
        y_coeffs,
        cost: total,
    })
}

/// The model-driven per-macroblock search of the reduced effort levels.
/// `rd.rate_tables` must be present (every frame driver attaches them).
pub(super) fn analyze_mb_model(
    pixels: &MbPixels,
    neighbors: &MbNeighbors,
    bmode_ctx: &BmodeContext,
    rd: &MbRdCtx,
    effort: EncodeEffort,
) -> MbAnalysis {
    let search = effort
        .model_search()
        .expect("the exhaustive level does not use the model search");
    let tables = rd
        .rate_tables
        .expect("frame drivers attach the per-frame rate tables");
    // The context the winners are quantised with: the caller's, or the
    // same with the trellis switched off.
    let plain = MbRdCtx {
        factors: rd.factors,
        coeff_probs: rd.coeff_probs,
        rate_tables: rd.rate_tables,
        lambda: rd.lambda,
        trellis: TrellisStrength::OFF,
    };
    let final_rd = if search.trellis { rd } else { &plain };
    let corner = DEFAULT_ABOVE_PIXEL;

    // ---- §12.2 whole-block luma ------------------------------------
    const Y_MODES: [IntraYMode; 4] = [IntraYMode::Dc, IntraYMode::V, IntraYMode::H, IntraYMode::Tm];
    let ymode_paths = &*KF_YMODE_PATHS;
    let y_mode_cost = |mode: IntraYMode| {
        rd.lambda * treed_bits_cached(ymode_paths, |i| KF_YMODE_PROB[i], mode.leaf())
    };
    let mut y_preds = [[0u8; 256]; 4];
    for (pred, &mode) in y_preds.iter_mut().zip(&Y_MODES) {
        predict_y16x16(
            pred,
            mode,
            neighbors.y_above.as_ref(),
            neighbors.y_left.as_ref(),
            neighbors.y_topleft.unwrap_or(corner),
        )
        .expect("whole-block modes only");
    }
    let mut whole_n = 0usize;
    let mut whole_cost = f64::INFINITY;
    let mut whole_has_ac = false;
    if search.whole_modes_by_prediction_error {
        let mut best_key = f64::INFINITY;
        for (n, &mode) in Y_MODES.iter().enumerate() {
            let key = block_ssd(&y_preds[n], &pixels.y) as f64 + y_mode_cost(mode);
            if key < best_key {
                best_key = key;
                whole_n = n;
            }
        }
        let (c, has_ac) = model_whole_luma(&pixels.y, &y_preds[whole_n], rd, tables, f64::INFINITY)
            .expect("no budget");
        whole_cost = c + y_mode_cost(Y_MODES[whole_n]);
        whole_has_ac = has_ac;
    } else {
        for (n, &mode) in Y_MODES.iter().enumerate() {
            // A mode whose partial cost already reaches the best total
            // cannot replace it.
            let signal = y_mode_cost(mode);
            let Some((c, has_ac)) =
                model_whole_luma(&pixels.y, &y_preds[n], rd, tables, whole_cost - signal)
            else {
                continue;
            };
            let cost = c + signal;
            if cost < whole_cost {
                whole_cost = cost;
                whole_n = n;
                whole_has_ac = has_ac;
            }
        }
    }

    // ---- §12.3 B_PRED ----------------------------------------------
    let bpred = if search.skip_bpred_without_luma_ac && !whole_has_ac {
        None
    } else {
        // `B_PRED` must come in strictly under the whole-block winner;
        // the pass stops as soon as its running cost rules that out.
        let signal = y_mode_cost(IntraYMode::B);
        model_bpred_luma(
            &pixels.y,
            neighbors,
            bmode_ctx,
            final_rd,
            tables,
            search,
            whole_cost - signal,
        )
        .filter(|pass| pass.cost + signal < whole_cost)
    };

    // ---- §12.2 chroma ----------------------------------------------
    const UV_MODES: [IntraUvMode; 4] = [
        IntraUvMode::Dc,
        IntraUvMode::V,
        IntraUvMode::H,
        IntraUvMode::Tm,
    ];
    let uv_paths = &*UV_MODE_PATHS;
    let mut uv_n = 0usize;
    let mut uv_best = f64::INFINITY;
    let mut u_preds = [[0u8; 64]; 4];
    let mut v_preds = [[0u8; 64]; 4];
    for (n, &mode) in UV_MODES.iter().enumerate() {
        predict_uv8x8(
            &mut u_preds[n],
            mode,
            neighbors.u_above.as_ref(),
            neighbors.u_left.as_ref(),
            neighbors.u_topleft.unwrap_or(corner),
        );
        predict_uv8x8(
            &mut v_preds[n],
            mode,
            neighbors.v_above.as_ref(),
            neighbors.v_left.as_ref(),
            neighbors.v_topleft.unwrap_or(corner),
        );
        let mode_cost =
            rd.lambda * treed_bits_cached(uv_paths, |i| KF_UV_MODE_PROB[i], mode.leaf());
        let cost = mode_cost
            + if search.whole_modes_by_prediction_error {
                (block_ssd(&u_preds[n], &pixels.u) + block_ssd(&v_preds[n], &pixels.v)) as f64
            } else {
                // The second plane is only worth modelling while the
                // mode can still come in under the best one.
                let u_cost = model_chroma_plane(&pixels.u, &u_preds[n], rd, tables);
                if mode_cost + u_cost >= uv_best {
                    continue;
                }
                u_cost + model_chroma_plane(&pixels.v, &v_preds[n], rd, tables)
            };
        if cost < uv_best {
            uv_best = cost;
            uv_n = n;
        }
    }

    // ---- Quantise the winners --------------------------------------
    let mut coeffs = MbCoeffs::default();
    let (y_mode, b_modes) = match bpred {
        Some(pass) => {
            coeffs.y = pass.y_coeffs;
            (IntraYMode::B, Some(pass.modes))
        }
        None => {
            let (y, y2) = transform_whole_block_luma_rd(
                &pixels.y,
                &y_preds[whole_n],
                rd.factors,
                Some(final_rd),
            );
            coeffs.y = y;
            coeffs.y2 = y2;
            (Y_MODES[whole_n], None)
        }
    };
    coeffs.u = transform_chroma_plane_rd(&pixels.u, &u_preds[uv_n], rd.factors, Some(final_rd));
    coeffs.v = transform_chroma_plane_rd(&pixels.v, &v_preds[uv_n], rd.factors, Some(final_rd));
    MbAnalysis::from_coeffs(coeffs, y_mode, UV_MODES[uv_n], b_modes)
}

#[cfg(test)]
mod tests {
    use super::super::enc_quantize_block;
    use super::*;

    #[test]
    fn effort_levels_round_trip() {
        for (n, effort) in EncodeEffort::ALL.into_iter().enumerate() {
            assert_eq!(effort.level() as usize, n);
            assert_eq!(EncodeEffort::from_level(n as u8), effort);
            assert_eq!(EncodeEffort::parse(effort.name()), Some(effort));
            assert_eq!(EncodeEffort::parse(&n.to_string()), Some(effort));
        }
        assert_eq!(EncodeEffort::from_level(200), EncodeEffort::Thorough);
        assert_eq!(EncodeEffort::default(), EncodeEffort::Balanced);
        assert_eq!(EncodeEffort::parse("turbo"), None);
        assert!(EncodeEffort::Fastest < EncodeEffort::Thorough);
    }

    /// The model's plain-rounded levels are the ones the reference
    /// rounding quantiser writes.
    #[test]
    fn model_levels_match_plain_rounding() {
        let tables = TrellisRateTables::for_probs(&crate::DEFAULT_COEFF_PROBS);
        let mut lcg = 0x2468_ACE1u32;
        for case in 0..4000 {
            let mut block = [0i16; 16];
            for c in block.iter_mut() {
                lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let span = [3i32, 40, 300, 4000][case % 4];
                *c = (((lcg >> 12) as i32).rem_euclid(2 * span + 1) - span) as i16;
            }
            let (dc, ac) = (
                [4i16, 9, 31, 157][case % 4],
                [4i16, 12, 37, 157][(case / 4) % 4],
            );
            for (block_type, first) in [(BlockType::YNoY2, 0usize), (BlockType::YAfterY2, 1)] {
                let mut want = block;
                if first == 1 {
                    want[0] = 0;
                }
                enc_quantize_block(&mut want, dc, ac);
                let mut got = block;
                let parts = model_block_parts(
                    &mut got,
                    TrellisBlockSpec {
                        dc,
                        ac,
                        first_coeff: first,
                        entry_ctx3: case % 3,
                        plane: block_type.plane_index(),
                    },
                    &DCT_WEIGHTS,
                    &tables,
                );
                let (cost, nonzero) = (
                    parts.dist as f64 / WEIGHT_SCALE + 7.5 * parts.rate,
                    parts.nonzero,
                );
                assert_eq!(got, want);
                assert_eq!(nonzero, want.iter().any(|&v| v != 0));
                assert!(cost.is_finite() && cost >= 0.0);
                // The sparse form prices the same model.
                let (sparse, sparse_nonzero) = model_block_cost(
                    &block,
                    TrellisBlockSpec {
                        dc,
                        ac,
                        first_coeff: first,
                        entry_ctx3: case % 3,
                        plane: block_type.plane_index(),
                    },
                    &DCT_WEIGHTS,
                    &tables,
                    7.5,
                );
                assert_eq!(sparse_nonzero, nonzero);
                assert!(
                    (sparse - cost).abs() <= 1e-9 * cost.max(1.0),
                    "sparse {sparse} vs reference {cost}"
                );
            }
        }
    }
}
