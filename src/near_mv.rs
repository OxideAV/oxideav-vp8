//! VP8 inter-mode and near/nearest motion-vector census — RFC 6386
//! §16.2 / §16.3 / §18.1.
//!
//! On an interframe an inter-predicted macroblock does not carry an
//! explicit motion vector for the three implicit modes; instead it
//! borrows a vector from its already-decoded neighbours. This module
//! implements that derivation:
//!
//! 1. **§16.3 `vp8_find_near_mvs` census** ([`find_near_mvs`]). The three
//!    spatial neighbours — above, left, and above-left — are surveyed in
//!    that order. Each non-intra neighbour with a non-zero vector
//!    contributes its (sign-bias-adjusted) vector to a small candidate
//!    list, accumulating a weight; the above and left neighbours weigh 2,
//!    the above-left 1 (§16.3 page 99). The result is a sorted list whose
//!    `[CNT_BEST] / [CNT_NEAREST] / [CNT_NEAR]` slots feed the four
//!    implicit/explicit modes, plus a four-entry weight census `cnt`.
//! 2. **§16.3 `vp8_mv_ref_probs`** ([`mv_ref_probs`]). The census `cnt`
//!    selects four probabilities from [`MV_COUNTS_TO_PROBS`] (the §20.13
//!    `mv_counts_to_probs[6][4]` table) that parameterise the inter-mode
//!    tree.
//! 3. **§16.2 inter-mode tree** ([`read_inter_mode`]). The §20.13
//!    `mv_ref_tree` is walked against those probabilities to recover one
//!    of [`InterMode`]'s five values (`Zero` / `Nearest` / `Near` /
//!    `New` / `Split`).
//! 4. **§18.1 clamp** ([`clamp_mv`] / [`MvClampRect`]). The §16.3 /
//!    §20.11 `clamp_mv` confines a predictor to a one-macroblock border
//!    around the frame; NEWMV re-clamps the "best" predictor before adding
//!    the explicitly coded differential (§18.1 page 114 "additional
//!    clamping ... for NEWMV").
//!
//! [`resolve_inter_mb_mv`] ties these together: it runs the census,
//! derives the probabilities, reads the mode, and — for the four
//! whole-MB modes — produces the single resolved per-MB vector that the
//! §18.2 prediction layer ([`crate::motion_comp::predict_inter_mb`])
//! applies to all sixteen Y sub-blocks. SPLITMV (§16.4) is reported as a
//! mode but its per-sub-block walk is a follow-up; [`resolve_inter_mb_mv`]
//! surfaces it via [`ResolvedInterMode::Split`] with the clamped "best"
//! base so the caller can decide.
//!
//! ## Units
//!
//! The census, the candidate list, and the clamp all operate in the
//! **quarter-pixel** units that [`crate::motion_vector::read_mv`] emits
//! and [`crate::motion_comp::predict_inter_mb`] consumes (the latter does
//! the §18.1 stored-luma doubling to eighth-pixel internally). The §20.11
//! `dixie` reference works in already-doubled eighth-pixel units (its
//! `read_mv_component` returns `x << 1`); the algorithm is identical up to
//! that consistent factor-of-two, so every clamp bound here is the §20.11
//! bound divided by two (`<< 6` where dixie writes `<< 7`,
//! `(16 << 2)` where dixie writes `(16 << 3)`). The two formulations
//! clamp the same vectors.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::motion_comp::{reconstruct_inter_mb, RefFrame, ReferencePlanes};
use crate::motion_vector::{read_mv, Mv, MvContexts};
use crate::reconstruct::ReconstructedMb;

/// The whole-macroblock inter-prediction mode — RFC 6386 §16.2
/// `mv_ref` enumeration, in the §20.13 `mv_ref_tree` leaf order.
///
/// The five values mirror the dixie `NEARESTMV` / `NEARMV` / `ZEROMV` /
/// `NEWMV` / `SPLITMV`, but are ordered here by their appearance as tree
/// leaves so [`MV_REF_TREE`]'s negative entries are direct indices into
/// this enum's discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterMode {
    /// `ZEROMV` — predict from the co-located MB (zero vector). Tree
    /// leaf 0.
    Zero,
    /// `NEARESTMV` — use the census "nearest" vector. Tree leaf 1.
    Nearest,
    /// `NEARMV` — use the census "near" vector. Tree leaf 2.
    Near,
    /// `NEWMV` — explicitly coded differential added to the clamped
    /// "best" predictor. Tree leaf 3.
    New,
    /// `SPLITMV` — per-sub-block vectors (§16.4). Tree leaf 4.
    Split,
}

/// `mv_ref_tree` — RFC 6386 §16.2 / §20.13.
///
/// Transcribed from the §20.13 listing
/// `{ -ZEROMV, 2, -NEARESTMV, 4, -NEARMV, 6, -NEWMV, -SPLITMV }`, with the
/// symbolic leaves replaced by their [`InterMode`] discriminant index
/// (`Zero=0`, `Nearest=1`, `Near=2`, `New=3`, `Split=4`). Negative
/// entries are `-leaf_index`; positive entries are child node offsets.
pub const MV_REF_TREE: [i8; 8] = [
    -0, 2, // "0" => ZEROMV (leaf 0)
    -1, 4, // "10" => NEARESTMV (leaf 1)
    -2, 6, // "110" => NEARMV (leaf 2)
    -3, -4, // "1110" => NEWMV (leaf 3), "1111" => SPLITMV (leaf 4)
];

/// `mv_counts_to_probs[6][4]` — RFC 6386 §16.3 `vp8_mode_contexts` /
/// §20.13 `mv_counts_to_probs`. Each census count (0..=5) selects a row;
/// the four columns feed the four [`MV_REF_TREE`] probability slots.
pub const MV_COUNTS_TO_PROBS: [[u8; 4]; 6] = [
    [7, 1, 1, 143],
    [14, 18, 14, 107],
    [135, 64, 57, 68],
    [60, 56, 128, 65],
    [159, 134, 128, 34],
    [234, 188, 128, 28],
];

/// Census-slot index of the "best" predictor — RFC 6386 §16.3
/// `CNT_BEST` / `CNT_ZERO` (the two share slot 0).
const CNT_BEST: usize = 0;
/// Census-slot index of the "nearest" predictor — §16.3 `CNT_NEAREST`.
const CNT_NEAREST: usize = 1;
/// Census-slot index of the "near" predictor — §16.3 `CNT_NEAR`.
const CNT_NEAR: usize = 2;
/// Census-slot index of the SPLITMV usage weight — §16.3 `CNT_SPLITMV`.
const CNT_SPLITMV: usize = 3;

/// Per-macroblock decode record consulted by the §16.3 census — the
/// subset of the §20.5 `mb_info` the neighbour survey reads.
///
/// A neighbour outside the visible frame is the §16.3 "border of 1
/// macroblock filled with 0,0 motion vectors": construct it with
/// [`MbInfo::border`] (an intra record with a zero vector), which the
/// census skips exactly as it skips a real intra macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbInfo {
    /// The reference frame this MB predicted from, or `None` for an
    /// intra-coded MB (dixie `CURRENT_FRAME`). The census takes no action
    /// for an intra neighbour (§16.3 page 99).
    pub ref_frame: Option<RefFrame>,
    /// The MB's resolved motion vector (quarter-pixel). Zero for intra or
    /// ZEROMV macroblocks. For a SPLITMV MB this is the vector of
    /// sub-block 15 (dixie `this->base.mv = this->split.mvs[15]`).
    pub mv: Mv,
    /// Whether this MB was coded SPLITMV — feeds the §16.3
    /// `cnt[CNT_SPLITMV]` recomputation.
    pub is_split: bool,
}

impl MbInfo {
    /// The §16.3 off-frame border record: an intra MB with a zero vector.
    /// The census skips it (its `ref_frame` is `None`), matching the
    /// reference's 1-MB border of 0,0 vectors.
    pub const fn border() -> Self {
        MbInfo {
            ref_frame: None,
            mv: Mv { row: 0, col: 0 },
            is_split: false,
        }
    }
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo::border()
    }
}

/// The four §16.3 reference-frame sign-bias flags, indexed by the dixie
/// `reference_frame` enum (`CURRENT=0`, `LAST=1`, `GOLDEN=2`,
/// `ALTREF=3`) — RFC 6386 §9.7 / §20.5 `sign_bias[4]`.
///
/// `CURRENT` and `LAST` are always 0; `GOLDEN` / `ALTREF` carry the
/// per-frame `sign_bias_golden` / `sign_bias_alternate` header bits (0 on
/// key frames). The §16.3 `mv_bias` correction XORs two entries to decide
/// whether a borrowed neighbour vector must be negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SignBias {
    /// `sign_bias[GOLDEN_FRAME]` — the §9.7 `sign_bias_golden` bit.
    pub golden: bool,
    /// `sign_bias[ALTREF_FRAME]` — the §9.7 `sign_bias_alternate` bit.
    pub alternate: bool,
}

impl SignBias {
    /// Build the sign-bias state from the two header bits. On a key frame
    /// pass `false, false` (§9.7: both default 0).
    pub const fn new(sign_bias_golden: bool, sign_bias_alternate: bool) -> Self {
        SignBias {
            golden: sign_bias_golden,
            alternate: sign_bias_alternate,
        }
    }

    /// The sign-bias bit for one reference frame (the §20.11
    /// `sign_bias[ref_frame]` lookup). `None` (intra / `CURRENT_FRAME`)
    /// and `Last` are always 0.
    #[inline]
    fn for_ref(self, ref_frame: Option<RefFrame>) -> bool {
        match ref_frame {
            None | Some(RefFrame::Last) => false,
            Some(RefFrame::Golden) => self.golden,
            Some(RefFrame::AltRef) => self.alternate,
        }
    }
}

/// Apply the §16.3 / §20.11 `mv_bias` sign correction.
///
/// "if `sign_bias[mb->base.ref_frame] ^ sign_bias[ref_frame]` then negate
/// both components." A neighbour predicting from a frame whose sign bias
/// differs from the current MB's reference contributes a *negated* vector
/// (§16.3 page 100).
#[inline]
fn mv_bias(
    neighbour_ref: Option<RefFrame>,
    current_ref: RefFrame,
    sign_bias: SignBias,
    mv: Mv,
) -> Mv {
    if sign_bias.for_ref(neighbour_ref) ^ sign_bias.for_ref(Some(current_ref)) {
        Mv {
            row: mv.row.wrapping_neg(),
            col: mv.col.wrapping_neg(),
        }
    } else {
        mv
    }
}

/// The §16.3 / §20.11 motion-vector clamp rectangle, in quarter-pixel
/// units.
///
/// The four bounds confine a predictor to a one-macroblock border around
/// the frame. Build per-row/column with [`MvClampRect::for_mb`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvClampRect {
    /// Minimum column (leftmost), quarter-pixel.
    pub to_left: i32,
    /// Maximum column (rightmost), quarter-pixel.
    pub to_right: i32,
    /// Minimum row (topmost), quarter-pixel.
    pub to_top: i32,
    /// Maximum row (bottommost), quarter-pixel.
    pub to_bottom: i32,
}

impl MvClampRect {
    /// The §20.11 `vp8_dixie_modemv_process_row` per-MB bounds, in
    /// quarter-pixel units.
    ///
    /// dixie computes (eighth-pixel): `to_left = -((col+1) << 7)`,
    /// `to_right = (mb_cols - col) << 7`, `to_top = -((row+1) << 7)`,
    /// `to_bottom = (mb_rows - row) << 7`. In the quarter-pixel space this
    /// module works in, each `<< 7` becomes `<< 6` (one MB = 16 px = 64
    /// quarter-pixels). Equivalent clamping, half the magnitude.
    pub fn for_mb(mb_col: usize, mb_row: usize, mb_cols: usize, mb_rows: usize) -> Self {
        MvClampRect {
            to_left: -(((mb_col + 1) as i32) << 6),
            to_right: ((mb_cols - mb_col) as i32) << 6,
            to_top: -(((mb_row + 1) as i32) << 6),
            to_bottom: ((mb_rows - mb_row) as i32) << 6,
        }
    }
}

/// Clamp one vector to the §16.3 / §20.11 `clamp_mv` rectangle.
///
/// ```text
/// newmv.x = clamp(raw.x, to_left, to_right);
/// newmv.y = clamp(raw.y, to_top,  to_bottom);
/// ```
///
/// Confines a predictor to the one-macroblock border so the referenced
/// location stays inside the (bordered) reference-frame buffer.
#[inline]
pub fn clamp_mv(mv: Mv, bounds: &MvClampRect) -> Mv {
    let col = (mv.col as i32).clamp(bounds.to_left, bounds.to_right);
    let row = (mv.row as i32).clamp(bounds.to_top, bounds.to_bottom);
    Mv {
        row: row as i16,
        col: col as i16,
    }
}

/// The §16.3 `vp8_find_near_mvs` result: the candidate list (slot 0 is
/// the "best", 1 "nearest", 2 "near") and the four-entry weight census.
///
/// The vectors are **unclamped** here (the reference clamps only when
/// extracting `nearest` / `near` / `best`, which [`resolve_inter_mb_mv`]
/// does per mode); `cnt` is the weighted census the §16.3
/// `vp8_mv_ref_probs` consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NearMvs {
    /// `near_mvs[0..=2]` — best / nearest / near candidate vectors
    /// (quarter-pixel, unclamped).
    pub mvs: [Mv; 3],
    /// `cnt[0..=3]` — the §16.3 weighted census (`CNT_BEST/ZERO`,
    /// `CNT_NEAREST`, `CNT_NEAR`, `CNT_SPLITMV`).
    pub cnt: [i32; 4],
}

/// Run the §16.3 `vp8_find_near_mvs` census over the three spatial
/// neighbours — RFC 6386 §16.3 / §20.11 `find_near_mvs`.
///
/// `current_ref` is the reference frame the current MB selected (drives
/// the §16.3 `mv_bias` sign correction); `sign_bias` the per-frame flags.
/// `above` / `left` / `aboveleft` are the already-decoded neighbour
/// records (use [`MbInfo::border`] for off-frame neighbours). Returns the
/// candidate list and weight census; vectors are unclamped.
pub fn find_near_mvs(
    above: &MbInfo,
    left: &MbInfo,
    aboveleft: &MbInfo,
    current_ref: RefFrame,
    sign_bias: SignBias,
) -> NearMvs {
    // Candidate list (dixie `near_mvs[4]`, slot 3 unused for vectors) and
    // a moving write cursor `mv_idx` (dixie `mv`), plus the count cursor
    // `cnt_idx` (dixie `cntx`).
    let mut near_mvs = [Mv::default(); 4];
    let mut cnt = [0i32; 4];
    let mut mv_idx: usize = 0;
    let mut cnt_idx: usize = 0;

    // Process above (§20.11): weight 2.
    if above.ref_frame.is_some() {
        if above.mv != Mv::default() {
            mv_idx += 1;
            near_mvs[mv_idx] = mv_bias(above.ref_frame, current_ref, sign_bias, above.mv);
            cnt_idx += 1;
        }
        cnt[cnt_idx] += 2;
    }

    // Process left (§20.11): weight 2; dedupe against the current cursor.
    if left.ref_frame.is_some() {
        if left.mv != Mv::default() {
            let this_mv = mv_bias(left.ref_frame, current_ref, sign_bias, left.mv);
            if this_mv != near_mvs[mv_idx] {
                mv_idx += 1;
                near_mvs[mv_idx] = this_mv;
                cnt_idx += 1;
            }
            cnt[cnt_idx] += 2;
        } else {
            cnt[CNT_BEST] += 2;
        }
    }

    // Process above-left (§20.11): weight 1; dedupe against the cursor.
    if aboveleft.ref_frame.is_some() {
        if aboveleft.mv != Mv::default() {
            let this_mv = mv_bias(aboveleft.ref_frame, current_ref, sign_bias, aboveleft.mv);
            if this_mv != near_mvs[mv_idx] {
                mv_idx += 1;
                near_mvs[mv_idx] = this_mv;
                cnt_idx += 1;
            }
            cnt[cnt_idx] += 1;
        } else {
            cnt[CNT_BEST] += 1;
        }
    }

    // If three distinct MVs were found, try to merge the above-left into
    // NEAREST (§20.11). `cnt[CNT_SPLITMV]` here still holds the running
    // count cursor's overflow value before being overwritten below.
    if cnt[CNT_SPLITMV] != 0 && near_mvs[mv_idx] == near_mvs[CNT_NEAREST] {
        cnt[CNT_NEAREST] += 1;
    }

    // Recompute cnt[CNT_SPLITMV] from neighbour SPLITMV usage (§20.11).
    cnt[CNT_SPLITMV] =
        (above.is_split as i32 + left.is_split as i32) * 2 + aboveleft.is_split as i32;

    // Swap near and nearest if near outweighs nearest (§20.11).
    if cnt[CNT_NEAR] > cnt[CNT_NEAREST] {
        cnt.swap(CNT_NEAREST, CNT_NEAR);
        near_mvs.swap(CNT_NEAREST, CNT_NEAR);
    }

    // Store the "best" MV in slot 0 (shares the address with CNT_ZERO):
    // if nearest outweighs zero, best := nearest (§20.11).
    if cnt[CNT_NEAREST] >= cnt[CNT_BEST] {
        near_mvs[CNT_BEST] = near_mvs[CNT_NEAREST];
    }

    NearMvs {
        mvs: [
            near_mvs[CNT_BEST],
            near_mvs[CNT_NEAREST],
            near_mvs[CNT_NEAR],
        ],
        cnt,
    }
}

/// Derive the four §16.2 inter-mode tree probabilities from the §16.3
/// census — RFC 6386 §16.3 / §20.11 `vp8_mv_ref_probs`.
///
/// `probs[i] = mv_counts_to_probs[cnt[i]][i]`. Each census count is
/// `0..=5` (the §16.3 "largest possible weight value in each case is 5"),
/// so it indexes [`MV_COUNTS_TO_PROBS`] directly.
pub fn mv_ref_probs(cnt: &[i32; 4]) -> [u8; 4] {
    [
        MV_COUNTS_TO_PROBS[cnt[0] as usize][0],
        MV_COUNTS_TO_PROBS[cnt[1] as usize][1],
        MV_COUNTS_TO_PROBS[cnt[2] as usize][2],
        MV_COUNTS_TO_PROBS[cnt[3] as usize][3],
    ]
}

/// Walk the §16.2 `mv_ref_tree` to read the whole-MB inter mode — RFC
/// 6386 §16.2 / §20.11 `bool_read_tree(bool, mv_ref_tree, probs)`.
///
/// `probs` is the four-entry table from [`mv_ref_probs`]; each tree node
/// at offset `i` reads `probs[i >> 1]`.
pub fn read_inter_mode(
    dec: &mut BoolDecoder<'_>,
    probs: &[u8; 4],
) -> Result<InterMode, BoolDecoderError> {
    let mut i: usize = 0;
    loop {
        let prob = probs[i >> 1];
        let bit = dec.read_bool(prob)? as usize;
        let next = MV_REF_TREE[i + bit];
        if next <= 0 {
            return Ok(match -next {
                0 => InterMode::Zero,
                1 => InterMode::Nearest,
                2 => InterMode::Near,
                3 => InterMode::New,
                _ => InterMode::Split,
            });
        }
        i = next as usize;
    }
}

/// The resolved per-macroblock inter mode and (for whole-MB modes) the
/// final quarter-pixel vector — the output of [`resolve_inter_mb_mv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedInterMode {
    /// A whole-MB mode (`ZEROMV` / `NEARESTMV` / `NEARMV` / `NEWMV`): the
    /// one clamped quarter-pixel vector to apply to all sixteen Y
    /// sub-blocks via [`crate::motion_comp::predict_inter_mb`]. `mode` is
    /// the resolved [`InterMode`] (never [`InterMode::Split`] here).
    Whole {
        /// The resolved whole-MB mode.
        mode: InterMode,
        /// The clamped quarter-pixel vector.
        mv: Mv,
    },
    /// `SPLITMV` (§16.4): the per-sub-block walk is a follow-up. Carries
    /// the clamped "best" base vector a NEW4x4 sub-block adds its
    /// differential to, so a later round can complete the split decode.
    Split {
        /// The clamped "best" predictor (the SPLITMV NEW4x4 base).
        best: Mv,
    },
}

/// Resolve one inter-predicted macroblock's mode and motion vector — RFC
/// 6386 §16.2 / §16.3 / §18.1, the whole-MB analogue of the §20.11
/// `decode_mvs` mode switch.
///
/// Steps:
/// 1. [`find_near_mvs`] over the three neighbours (`current_ref` /
///    `sign_bias` drive the §16.3 sign correction).
/// 2. [`mv_ref_probs`] from the census, then [`read_inter_mode`].
/// 3. Per the resolved mode (§16.2 page 104 / §20.11):
///    * `ZEROMV` → zero vector.
///    * `NEARESTMV` → `clamp_mv(nearest)`.
///    * `NEARMV` → `clamp_mv(near)`.
///    * `NEWMV` → `clamp_mv(best)` then `read_mv` differential added
///      component-wise (the §18.1 secondary clamp is the clamp on `best`
///      *before* the add, exactly as §20.11 `decode_mvs` does it).
///    * `SPLITMV` → reported with the clamped `best` base (§16.4 walk
///      deferred).
///
/// `bounds` is the per-MB [`MvClampRect`]; `mv_contexts` the resolved §17
/// MV probability contexts the NEWMV differential reads against.
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §16.2/§16.3 input.
pub fn resolve_inter_mb_mv(
    dec: &mut BoolDecoder<'_>,
    above: &MbInfo,
    left: &MbInfo,
    aboveleft: &MbInfo,
    current_ref: RefFrame,
    sign_bias: SignBias,
    bounds: &MvClampRect,
    mv_contexts: &MvContexts,
) -> Result<ResolvedInterMode, BoolDecoderError> {
    let near = find_near_mvs(above, left, aboveleft, current_ref, sign_bias);
    let probs = mv_ref_probs(&near.cnt);
    let mode = read_inter_mode(dec, &probs)?;

    let resolved = match mode {
        InterMode::Zero => ResolvedInterMode::Whole {
            mode,
            mv: Mv::default(),
        },
        InterMode::Nearest => ResolvedInterMode::Whole {
            mode,
            mv: clamp_mv(near.mvs[1], bounds),
        },
        InterMode::Near => ResolvedInterMode::Whole {
            mode,
            mv: clamp_mv(near.mvs[2], bounds),
        },
        InterMode::New => {
            let best = clamp_mv(near.mvs[0], bounds);
            let diff = read_mv(dec, mv_contexts)?;
            ResolvedInterMode::Whole {
                mode,
                mv: Mv {
                    row: best.row.wrapping_add(diff.row),
                    col: best.col.wrapping_add(diff.col),
                },
            }
        }
        InterMode::Split => ResolvedInterMode::Split {
            best: clamp_mv(near.mvs[0], bounds),
        },
    };

    Ok(resolved)
}

/// Errors surfaced by [`decode_inter_mb`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterMbError {
    /// The bool decoder ran out of input during mode / MV decoding.
    BoolDecoder(BoolDecoderError),
    /// The resolved mode was `SPLITMV` (§16.4); the per-sub-block walk
    /// is not yet wired into the whole-MB reconstruction path. The
    /// clamped "best" base vector is carried so a future round can
    /// complete the split decode.
    SplitNotSupported {
        /// The clamped "best" predictor (the SPLITMV NEW4x4 base).
        best: Mv,
    },
}

impl core::fmt::Display for InterMbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterMbError::BoolDecoder(inner) => write!(f, "vp8 inter-mb: {inner}"),
            InterMbError::SplitNotSupported { .. } => f.write_str(
                "vp8 inter-mb: SPLITMV (§16.4) per-sub-block walk not yet implemented; \
                 the four whole-MB inter modes decode end-to-end",
            ),
        }
    }
}

impl std::error::Error for InterMbError {}

impl From<BoolDecoderError> for InterMbError {
    fn from(value: BoolDecoderError) -> Self {
        InterMbError::BoolDecoder(value)
    }
}

/// Decode and reconstruct one whole-MB inter-predicted macroblock
/// end-to-end — RFC 6386 §16.2 / §16.3 / §18.
///
/// This is the integration entry point that wires the §16.3 census
/// ([`find_near_mvs`]) + §16.2 inter-mode tree ([`read_inter_mode`]) +
/// §18.1 clamp ([`clamp_mv`]) through the §18.2/§18.3 prediction layer:
/// it calls [`resolve_inter_mb_mv`] to obtain the single per-MB vector,
/// then drives [`crate::motion_comp::reconstruct_inter_mb`] with it,
/// producing reconstructed Y/U/V pixels for the macroblock.
///
/// On `ZEROMV` / `NEARESTMV` / `NEARMV` / `NEWMV` the returned
/// [`ReconstructedMb`] is the §18.2 prediction plus the §14 dequantized
/// residual (the resolved vector is also returned, for the caller to
/// store as the neighbour record's `mv` in the next MB's census).
/// `SPLITMV` (§16.4) returns [`InterMbError::SplitNotSupported`] carrying
/// the clamped best base, since the per-sub-block walk is a follow-up.
///
/// `full_pixel` is the version-3 full-pel-chroma flag; `filters` the
/// §20.14 version-selected tap set; the coefficient arrays are
/// pre-dequantized (the caller's responsibility, matching the keyframe
/// path).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §16/§18 input.
pub fn decode_inter_mb(
    dec: &mut BoolDecoder<'_>,
    reference: &ReferencePlanes<'_>,
    mb_col: usize,
    mb_row: usize,
    above: &MbInfo,
    left: &MbInfo,
    aboveleft: &MbInfo,
    current_ref: RefFrame,
    sign_bias: SignBias,
    mv_contexts: &MvContexts,
    full_pixel: bool,
    filters: &[[i32; 6]; 8],
    mb_skip_coeff: bool,
    y2_coeffs_dequant: &[i16; 16],
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<(ReconstructedMb, Mv), InterMbError> {
    let bounds = MvClampRect::for_mb(mb_col, mb_row, reference.mb_cols, reference.mb_rows);
    let resolved = resolve_inter_mb_mv(
        dec,
        above,
        left,
        aboveleft,
        current_ref,
        sign_bias,
        &bounds,
        mv_contexts,
    )?;

    let mv = match resolved {
        ResolvedInterMode::Whole { mv, .. } => mv,
        ResolvedInterMode::Split { best } => return Err(InterMbError::SplitNotSupported { best }),
    };

    let recon = reconstruct_inter_mb(
        reference,
        mb_col,
        mb_row,
        mv,
        full_pixel,
        filters,
        mb_skip_coeff,
        y2_coeffs_dequant,
        y_coeffs_dequant,
        u_coeffs_dequant,
        v_coeffs_dequant,
    );

    Ok((recon, mv))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal test-side VP8 boolean *encoder*, mirroring the proven
    /// encoder used in `motion_vector::tests` / `bool_decoder::tests`.
    /// Constructs bitstreams the §16 routines decode back, so round-trips
    /// assert against bytes generated from the spec's coding rules — not
    /// any external reference.
    struct BoolEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl BoolEncoder {
        fn new() -> Self {
            BoolEncoder {
                out: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn write_bool(&mut self, prob: u8, val: bool) {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            if val {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                if (self.bottom >> 31) & 1 == 1 {
                    let mut i = self.out.len();
                    while i > 0 {
                        i -= 1;
                        if self.out[i] == 255 {
                            self.out[i] = 0;
                        } else {
                            self.out[i] = self.out[i].wrapping_add(1);
                            break;
                        }
                    }
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    let byte = (self.bottom >> 24) as u8;
                    self.out.push(byte);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            let c = self.bit_count;
            let v = self.bottom;
            if v & (1u32 << (32 - c)) != 0 {
                let mut i = self.out.len();
                while i > 0 {
                    i -= 1;
                    if self.out[i] == 255 {
                        self.out[i] = 0;
                    } else {
                        self.out[i] = self.out[i].wrapping_add(1);
                        break;
                    }
                }
            }
            let mut v = v;
            v <<= c & 7;
            let mut c_shift = c >> 3;
            while c_shift > 0 {
                v <<= 8;
                c_shift -= 1;
            }
            for _ in 0..4 {
                self.out.push((v >> 24) as u8);
                v <<= 8;
            }
            while self.out.len() < 2 {
                self.out.push(0);
            }
            self.out
        }

        /// Emit the `mv_ref_tree` path for `mode` against `probs`.
        fn write_inter_mode(&mut self, probs: &[u8; 4], mode: InterMode) {
            let target = match mode {
                InterMode::Zero => 0i8,
                InterMode::Nearest => 1,
                InterMode::Near => 2,
                InterMode::New => 3,
                InterMode::Split => 4,
            };
            // Walk MV_REF_TREE finding the bit path to `-target`.
            fn find_path(tree: &[i8], i: usize, target: i8, path: &mut Vec<bool>) -> bool {
                for bit in 0..2 {
                    let next = tree[i + bit];
                    path.push(bit == 1);
                    if next <= 0 {
                        if -next == target {
                            return true;
                        }
                    } else if find_path(tree, next as usize, target, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            let mut path = Vec::new();
            assert!(find_path(&MV_REF_TREE, 0, target, &mut path));
            let mut i = 0usize;
            for &bit in &path {
                self.write_bool(probs[i >> 1], bit);
                let next = MV_REF_TREE[i + bit as usize];
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }

        /// Emit one MV component the §17.1 way (mirrors the proven
        /// `motion_vector::tests` encoder).
        fn write_mv_component(&mut self, p: &[u8; crate::coded_header::MV_PROB_COUNT], value: i32) {
            const MVP_IS_SHORT: usize = 0;
            const MVP_SIGN: usize = 1;
            const MVP_SHORT: usize = 2;
            const MVP_BITS: usize = MVP_SHORT + 7;
            let a = value.unsigned_abs() as i32;
            if a > 7 {
                self.write_bool(p[MVP_IS_SHORT], true);
                for i in 0..3 {
                    self.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
                }
                let mut i = 9;
                loop {
                    self.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
                    i -= 1;
                    if i <= 3 {
                        break;
                    }
                }
                if (a & 0xfff0) != 0 {
                    self.write_bool(p[MVP_BITS + 3], (a >> 3) & 1 != 0);
                }
            } else {
                self.write_bool(p[MVP_IS_SHORT], false);
                self.write_small_mv(p, a);
            }
            if a != 0 {
                self.write_bool(p[MVP_SIGN], value < 0);
            }
        }

        fn write_small_mv(&mut self, p: &[u8; crate::coded_header::MV_PROB_COUNT], leaf: i32) {
            const MVP_SHORT: usize = 2;
            let tree = crate::motion_vector::SMALL_MVTREE;
            fn find_path(tree: &[i8], i: usize, target: i8, path: &mut Vec<bool>) -> bool {
                for bit in 0..2 {
                    let next = tree[i + bit];
                    path.push(bit == 1);
                    if next <= 0 {
                        if next == -target {
                            return true;
                        }
                    } else if find_path(tree, next as usize, target, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            let mut path = Vec::new();
            assert!(find_path(&tree, 0, leaf as i8, &mut path));
            let mut i = 0usize;
            for &bit in &path {
                self.write_bool(p[MVP_SHORT + (i >> 1)], bit);
                let next = tree[i + bit as usize];
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }

        fn write_mv(&mut self, contexts: &MvContexts, mv: Mv) {
            self.write_mv_component(&contexts[0], mv.row as i32);
            self.write_mv_component(&contexts[1], mv.col as i32);
        }
    }

    fn inter(ref_frame: RefFrame, mv: Mv) -> MbInfo {
        MbInfo {
            ref_frame: Some(ref_frame),
            mv,
            is_split: false,
        }
    }

    fn split(ref_frame: RefFrame, mv: Mv) -> MbInfo {
        MbInfo {
            ref_frame: Some(ref_frame),
            mv,
            is_split: true,
        }
    }

    // ---- Tables / tree shape ----------------------------------------

    #[test]
    fn mv_ref_tree_matches_spec() {
        // §20.13 mv_ref_tree, leaves mapped to InterMode discriminants.
        assert_eq!(MV_REF_TREE, [0, 2, -1, 4, -2, 6, -3, -4]);
    }

    #[test]
    fn mv_counts_to_probs_matches_spec() {
        assert_eq!(MV_COUNTS_TO_PROBS[0], [7, 1, 1, 143]);
        assert_eq!(MV_COUNTS_TO_PROBS[1], [14, 18, 14, 107]);
        assert_eq!(MV_COUNTS_TO_PROBS[2], [135, 64, 57, 68]);
        assert_eq!(MV_COUNTS_TO_PROBS[3], [60, 56, 128, 65]);
        assert_eq!(MV_COUNTS_TO_PROBS[4], [159, 134, 128, 34]);
        assert_eq!(MV_COUNTS_TO_PROBS[5], [234, 188, 128, 28]);
    }

    // ---- mode tree round-trips --------------------------------------

    fn roundtrip_mode(probs: &[u8; 4], mode: InterMode) -> InterMode {
        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(probs, mode);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        read_inter_mode(&mut dec, probs).unwrap()
    }

    #[test]
    fn every_inter_mode_round_trips() {
        let probs = [110u8, 90, 70, 50];
        for mode in [
            InterMode::Zero,
            InterMode::Nearest,
            InterMode::Near,
            InterMode::New,
            InterMode::Split,
        ] {
            assert_eq!(roundtrip_mode(&probs, mode), mode, "mode {mode:?}");
        }
    }

    #[test]
    fn inter_mode_round_trips_under_default_probs() {
        // The default census (all-intra neighbours → cnt all zero) gives
        // mv_counts_to_probs row 0 across the board.
        let probs = mv_ref_probs(&[0, 0, 0, 0]);
        assert_eq!(probs, [7, 1, 1, 143]);
        for mode in [
            InterMode::Zero,
            InterMode::Nearest,
            InterMode::Near,
            InterMode::New,
            InterMode::Split,
        ] {
            assert_eq!(roundtrip_mode(&probs, mode), mode);
        }
    }

    // ---- census: all-intra / border ---------------------------------

    #[test]
    fn all_border_neighbours_give_zero_census() {
        let b = MbInfo::border();
        let near = find_near_mvs(&b, &b, &b, RefFrame::Last, SignBias::default());
        assert_eq!(near.cnt, [0, 0, 0, 0]);
        assert_eq!(near.mvs, [Mv::default(); 3]);
    }

    #[test]
    fn border_is_intra_with_zero_mv() {
        let b = MbInfo::border();
        assert_eq!(b.ref_frame, None);
        assert_eq!(b.mv, Mv::default());
        assert!(!b.is_split);
        assert_eq!(MbInfo::default(), b);
    }

    // ---- census: single non-zero above ------------------------------

    #[test]
    fn single_above_vector_populates_nearest() {
        let v = Mv { row: 8, col: -4 };
        let above = inter(RefFrame::Last, v);
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        // above contributes weight 2 into the NEAREST slot.
        assert_eq!(near.cnt[CNT_NEAREST], 2);
        assert_eq!(near.mvs[1], v); // nearest
                                    // best := nearest because cnt[NEAREST] (2) >= cnt[BEST] (0).
        assert_eq!(near.mvs[0], v);
    }

    #[test]
    fn intra_above_takes_no_action() {
        // An intra above neighbour (ref_frame None) is skipped entirely.
        let above = MbInfo::border();
        let left = inter(RefFrame::Last, Mv { row: 12, col: 0 });
        let border = MbInfo::border();
        let near = find_near_mvs(&above, &left, &border, RefFrame::Last, SignBias::default());
        assert_eq!(near.cnt[CNT_NEAREST], 2); // only left counted
        assert_eq!(near.mvs[1], Mv { row: 12, col: 0 });
    }

    // ---- census: zero-vector inter neighbour scores CNT_ZERO --------

    #[test]
    fn zero_vector_above_scores_zero_slot() {
        // An inter above with a zero vector adds 2 to cnt[CNT_BEST/ZERO]
        // via the `*cntx += 2` with cntx still at slot 0.
        let above = inter(RefFrame::Last, Mv::default());
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        assert_eq!(near.cnt[CNT_BEST], 2);
        assert_eq!(near.cnt[CNT_NEAREST], 0);
    }

    #[test]
    fn zero_vector_left_scores_zero_slot_explicitly() {
        // The left/aboveleft else-branch routes a zero vector to
        // cnt[CNT_ZERO] directly.
        let above = MbInfo::border();
        let left = inter(RefFrame::Last, Mv::default());
        let border = MbInfo::border();
        let near = find_near_mvs(&above, &left, &border, RefFrame::Last, SignBias::default());
        assert_eq!(near.cnt[CNT_BEST], 2);
    }

    // ---- census: dedupe + weighting ---------------------------------

    #[test]
    fn identical_above_and_left_merge_weights() {
        let v = Mv { row: 16, col: 8 };
        let a = inter(RefFrame::Last, v);
        let l = inter(RefFrame::Last, v);
        let border = MbInfo::border();
        let near = find_near_mvs(&a, &l, &border, RefFrame::Last, SignBias::default());
        // Same vector: left dedupes onto above's slot → weight 2+2 = 4.
        assert_eq!(near.cnt[CNT_NEAREST], 4);
        assert_eq!(near.mvs[1], v);
    }

    #[test]
    fn distinct_above_and_left_make_two_candidates() {
        let va = Mv { row: 16, col: 0 };
        let vl = Mv { row: 0, col: 16 };
        let a = inter(RefFrame::Last, va);
        let l = inter(RefFrame::Last, vl);
        let border = MbInfo::border();
        let near = find_near_mvs(&a, &l, &border, RefFrame::Last, SignBias::default());
        // Two distinct vectors, each weight 2. near (vl) weight equals
        // nearest (va) weight → no swap (swap is strictly greater).
        assert_eq!(near.cnt[CNT_NEAREST], 2);
        assert_eq!(near.cnt[CNT_NEAR], 2);
        assert_eq!(near.mvs[1], va);
        assert_eq!(near.mvs[2], vl);
    }

    #[test]
    fn near_swaps_ahead_of_nearest_when_outweighing() {
        // above (va, weight 2) then left distinct (vl, weight 2) then
        // aboveleft == vl (weight 1) → near accumulates 3 > nearest 2,
        // so the swap fires.
        let va = Mv { row: 16, col: 0 };
        let vl = Mv { row: 0, col: 16 };
        let a = inter(RefFrame::Last, va);
        let l = inter(RefFrame::Last, vl);
        let al = inter(RefFrame::Last, vl);
        let near = find_near_mvs(&a, &l, &al, RefFrame::Last, SignBias::default());
        // After swap: nearest holds vl (weight 3), near holds va (weight 2).
        assert_eq!(near.cnt[CNT_NEAREST], 3);
        assert_eq!(near.cnt[CNT_NEAR], 2);
        assert_eq!(near.mvs[1], vl);
        assert_eq!(near.mvs[2], va);
    }

    // ---- census: SPLITMV weighting ----------------------------------

    #[test]
    fn splitmv_neighbours_score_cnt_splitmv() {
        // above split (×2) + left split (×2) + aboveleft split (×1) = 5.
        let v = Mv { row: 4, col: 4 };
        let a = split(RefFrame::Last, v);
        let l = split(RefFrame::Last, Mv { row: 8, col: 8 });
        let al = split(RefFrame::Last, Mv { row: 12, col: 12 });
        let near = find_near_mvs(&a, &l, &al, RefFrame::Last, SignBias::default());
        assert_eq!(near.cnt[CNT_SPLITMV], 5);
    }

    // ---- sign bias --------------------------------------------------

    #[test]
    fn sign_bias_negates_when_bits_differ() {
        // Neighbour predicts from GOLDEN (sign_bias golden = true),
        // current MB references LAST (sign_bias false) → XOR true → negate.
        let v = Mv { row: 20, col: -8 };
        let above = inter(RefFrame::Golden, v);
        let border = MbInfo::border();
        let sb = SignBias::new(true, false);
        let near = find_near_mvs(&above, &border, &border, RefFrame::Last, sb);
        assert_eq!(near.mvs[1], Mv { row: -20, col: 8 });
    }

    #[test]
    fn sign_bias_no_negate_when_bits_match() {
        let v = Mv { row: 20, col: -8 };
        let above = inter(RefFrame::Golden, v);
        let border = MbInfo::border();
        // Both golden and current-ref-golden sign bias true → XOR false.
        let sb = SignBias::new(true, false);
        let near = find_near_mvs(&above, &border, &border, RefFrame::Golden, sb);
        assert_eq!(near.mvs[1], v);
    }

    // ---- clamp ------------------------------------------------------

    #[test]
    fn clamp_rect_for_mb_matches_spec_scaled() {
        // Top-left MB of a 4x3 grid: to_left = -((0+1)<<6) = -64, etc.
        let r = MvClampRect::for_mb(0, 0, 4, 3);
        assert_eq!(r.to_left, -64);
        assert_eq!(r.to_top, -64);
        assert_eq!(r.to_right, 4 << 6);
        assert_eq!(r.to_bottom, 3 << 6);
    }

    #[test]
    fn clamp_confines_vector() {
        let r = MvClampRect {
            to_left: -64,
            to_right: 64,
            to_top: -64,
            to_bottom: 64,
        };
        assert_eq!(
            clamp_mv(
                Mv {
                    row: 200,
                    col: -200
                },
                &r
            ),
            Mv { row: 64, col: -64 }
        );
        assert_eq!(
            clamp_mv(Mv { row: 10, col: -10 }, &r),
            Mv { row: 10, col: -10 }
        );
    }

    // ---- mv_ref_probs ----------------------------------------------

    #[test]
    fn mv_ref_probs_indexes_per_column() {
        // cnt = [2,1,5,3] → probs pick column i from row cnt[i].
        let probs = mv_ref_probs(&[2, 1, 5, 3]);
        assert_eq!(
            probs,
            [
                MV_COUNTS_TO_PROBS[2][0],
                MV_COUNTS_TO_PROBS[1][1],
                MV_COUNTS_TO_PROBS[5][2],
                MV_COUNTS_TO_PROBS[3][3],
            ]
        );
    }

    // ---- resolve_inter_mb_mv end-to-end -----------------------------

    fn default_contexts() -> MvContexts {
        crate::motion_vector::default_mv_contexts()
    }

    #[test]
    fn resolve_zeromv_yields_zero_vector() {
        let near_mb = inter(RefFrame::Last, Mv { row: 40, col: 40 });
        let border = MbInfo::border();
        let cnt = find_near_mvs(
            &near_mb,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        )
        .cnt;
        let probs = mv_ref_probs(&cnt);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Zero);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let bounds = MvClampRect::for_mb(1, 1, 4, 4);
        let res = resolve_inter_mb_mv(
            &mut dec,
            &near_mb,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &bounds,
            &default_contexts(),
        )
        .unwrap();
        assert_eq!(
            res,
            ResolvedInterMode::Whole {
                mode: InterMode::Zero,
                mv: Mv::default()
            }
        );
    }

    #[test]
    fn resolve_nearestmv_uses_clamped_nearest() {
        let v = Mv { row: 12, col: -8 };
        let above = inter(RefFrame::Last, v);
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Nearest);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        // Generous bounds so the clamp is a no-op.
        let bounds = MvClampRect::for_mb(2, 2, 8, 8);
        let res = resolve_inter_mb_mv(
            &mut dec,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &bounds,
            &default_contexts(),
        )
        .unwrap();
        assert_eq!(
            res,
            ResolvedInterMode::Whole {
                mode: InterMode::Nearest,
                mv: v
            }
        );
    }

    #[test]
    fn resolve_nearestmv_clamp_is_applied() {
        let v = Mv {
            row: 1000,
            col: 1000,
        };
        let above = inter(RefFrame::Last, v);
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Nearest);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let bounds = MvClampRect::for_mb(0, 0, 2, 2);
        let res = resolve_inter_mb_mv(
            &mut dec,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &bounds,
            &default_contexts(),
        )
        .unwrap();
        let expected = clamp_mv(v, &bounds);
        assert_eq!(
            res,
            ResolvedInterMode::Whole {
                mode: InterMode::Nearest,
                mv: expected
            }
        );
        // Confirm the clamp actually moved the vector.
        assert_ne!(expected, v);
    }

    #[test]
    fn resolve_newmv_adds_differential_to_clamped_best() {
        // Set up a census so best is a known clamped vector, then encode
        // NEWMV followed by an explicit differential.
        let base = Mv { row: 16, col: 8 };
        let above = inter(RefFrame::Last, base);
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);
        let bounds = MvClampRect::for_mb(3, 3, 8, 8); // generous: clamp no-op
        let best = clamp_mv(near.mvs[0], &bounds);

        let contexts = default_contexts();
        let diff = Mv { row: -5, col: 30 };

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::New);
        enc.write_mv(&contexts, diff);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = resolve_inter_mb_mv(
            &mut dec,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &bounds,
            &contexts,
        )
        .unwrap();
        assert_eq!(
            res,
            ResolvedInterMode::Whole {
                mode: InterMode::New,
                mv: Mv {
                    row: best.row + diff.row,
                    col: best.col + diff.col
                }
            }
        );
    }

    #[test]
    fn resolve_splitmv_reports_clamped_best() {
        let base = Mv { row: 20, col: 20 };
        let above = inter(RefFrame::Last, base);
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);
        let bounds = MvClampRect::for_mb(0, 0, 2, 2);
        let best = clamp_mv(near.mvs[0], &bounds);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Split);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = resolve_inter_mb_mv(
            &mut dec,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &bounds,
            &default_contexts(),
        )
        .unwrap();
        assert_eq!(res, ResolvedInterMode::Split { best });
    }

    // ---- decode_inter_mb end-to-end (census → mode → predict) -------

    /// Build a reference frame whose planes carry a deterministic
    /// per-position value, mirroring `motion_comp::tests::build_reference`.
    fn build_reference(mb_cols: usize, mb_rows: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let lw = mb_cols * 16;
        let lh = mb_rows * 16;
        let cw = mb_cols * 8;
        let ch = mb_rows * 8;
        let y = (0..lw * lh).map(|i| (i % 251) as u8).collect();
        let u = (0..cw * ch).map(|i| ((i * 3 + 1) % 251) as u8).collect();
        let v = (0..cw * ch).map(|i| ((i * 7 + 2) % 251) as u8).collect();
        (y, u, v)
    }

    /// All-zero pre-dequantized coefficient blocks for a skip-coeff MB.
    struct ZeroCoeffs {
        y2: [i16; 16],
        y: [[i16; 16]; 16],
        u: [[i16; 16]; 4],
        v: [[i16; 16]; 4],
    }

    fn zero_coeffs() -> ZeroCoeffs {
        ZeroCoeffs {
            y2: [0i16; 16],
            y: [[0i16; 16]; 16],
            u: [[0i16; 16]; 4],
            v: [[0i16; 16]; 4],
        }
    }

    #[test]
    fn decode_inter_mb_zeromv_copies_colocated_block() {
        // A ZEROMV inter MB with skip_coeff must reconstruct to the
        // co-located reference MB verbatim — proving the census-resolved
        // zero vector flows through predict_inter_mb end-to-end.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        // Census: a single non-zero above neighbour, but we encode ZEROMV.
        let above = inter(RefFrame::Last, Mv { row: 40, col: 40 });
        let border = MbInfo::border();
        let cnt = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        )
        .cnt;
        let probs = mv_ref_probs(&cnt);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Zero);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let (recon, mv) = decode_inter_mb(
            &mut dec,
            &reference,
            1,
            1,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &default_contexts(),
            false,
            filters,
            true, // skip_coeff: prediction is the reconstruction
            &coeffs.y2,
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap();

        assert_eq!(mv, Mv::default());
        // Reconstructed luma == reference MB (1,1).
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(
                    recon.y[r * 16 + c],
                    y[(16 + r) * 32 + (16 + c)],
                    "y ({r},{c})"
                );
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(recon.u[r * 8 + c], u[(8 + r) * 16 + (8 + c)], "u ({r},{c})");
                assert_eq!(recon.v[r * 8 + c], v[(8 + r) * 16 + (8 + c)], "v ({r},{c})");
            }
        }
    }

    #[test]
    fn decode_inter_mb_nearestmv_uses_neighbour_vector() {
        // NEARESTMV: the census borrows the above neighbour's whole-pixel
        // vector and the reconstruction must equal the offset reference
        // block. Pick a vector that is whole-pixel after §18.1 doubling
        // in both planes (matching motion_comp's whole-pixel test):
        // quarter-pel (8, 16) → doubled (16, 32) → luma offset (2, 4),
        // chroma offset (1, 2).
        let (y, u, v) = build_reference(3, 3);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 48,
            uv_stride: 24,
            mb_cols: 3,
            mb_rows: 3,
        };
        let nv = Mv { row: 8, col: 16 };
        let above = inter(RefFrame::Last, nv);
        let border = MbInfo::border();
        let cnt = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        )
        .cnt;
        let probs = mv_ref_probs(&cnt);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Nearest);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let (recon, mv) = decode_inter_mb(
            &mut dec,
            &reference,
            1,
            1,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &default_contexts(),
            false,
            filters,
            true,
            &coeffs.y2,
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap();

        // The clamp for MB(1,1) on a 3×3 grid is generous enough not to
        // move (8,16): to_left=-128, to_right=128, etc.
        assert_eq!(mv, nv);
        // Luma: MB (1,1) origin (16,16) + offset (2,4).
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(
                    recon.y[r * 16 + c],
                    y[(16 + 2 + r) * 48 + (16 + 4 + c)],
                    "y ({r},{c})"
                );
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    recon.u[r * 8 + c],
                    u[(8 + 1 + r) * 24 + (8 + 2 + c)],
                    "u ({r},{c})"
                );
                assert_eq!(
                    recon.v[r * 8 + c],
                    v[(8 + 1 + r) * 24 + (8 + 2 + c)],
                    "v ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn decode_inter_mb_newmv_adds_differential() {
        // NEWMV: best predictor (the above neighbour's vector) plus a
        // decoded differential; with skip_coeff the reconstruction is the
        // prediction at the combined whole-pixel vector. Choose base +
        // diff so the sum is whole-pixel in both planes: base (8,16) +
        // diff (8,16) = (16,32) → doubled (32,64) → luma offset (4,8),
        // chroma offset (2,4).
        let (y, u, v) = build_reference(4, 4);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 64,
            uv_stride: 32,
            mb_cols: 4,
            mb_rows: 4,
        };
        let base = Mv { row: 8, col: 16 };
        let above = inter(RefFrame::Last, base);
        let border = MbInfo::border();
        let cnt = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        )
        .cnt;
        let probs = mv_ref_probs(&cnt);
        let contexts = default_contexts();
        let diff = Mv { row: 8, col: 16 };

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::New);
        enc.write_mv(&contexts, diff);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let (recon, mv) = decode_inter_mb(
            &mut dec,
            &reference,
            1,
            1,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &contexts,
            false,
            filters,
            true,
            &coeffs.y2,
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap();

        let combined = Mv {
            row: base.row + diff.row,
            col: base.col + diff.col,
        };
        assert_eq!(mv, combined);
        // Luma offset (4,8), chroma offset (2,4); MB(1,1).
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(
                    recon.y[r * 16 + c],
                    y[(16 + 4 + r) * 64 + (16 + 8 + c)],
                    "y ({r},{c})"
                );
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    recon.u[r * 8 + c],
                    u[(8 + 2 + r) * 32 + (8 + 4 + c)],
                    "u ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn decode_inter_mb_splitmv_surfaces_error_with_best() {
        // SPLITMV reaches decode_inter_mb but the per-sub-block walk is a
        // follow-up: it surfaces InterMbError::SplitNotSupported carrying
        // the clamped best base.
        let (y, u, v) = build_reference(2, 2);
        let reference = ReferencePlanes {
            y: &y,
            u: &u,
            v: &v,
            y_stride: 32,
            uv_stride: 16,
            mb_cols: 2,
            mb_rows: 2,
        };
        let above = inter(
            RefFrame::Last,
            Mv {
                row: 1000,
                col: 1000,
            },
        );
        let border = MbInfo::border();
        let near = find_near_mvs(
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
        );
        let probs = mv_ref_probs(&near.cnt);
        let bounds = MvClampRect::for_mb(0, 0, 2, 2);
        let expected_best = clamp_mv(near.mvs[0], &bounds);

        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Split);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let err = decode_inter_mb(
            &mut dec,
            &reference,
            0,
            0,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &default_contexts(),
            false,
            filters,
            true,
            &coeffs.y2,
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap_err();
        assert_eq!(
            err,
            InterMbError::SplitNotSupported {
                best: expected_best
            }
        );
    }
}
