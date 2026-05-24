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
use crate::motion_comp::{
    reconstruct_inter_mb, reconstruct_split_mv_mb, RefFrame, ReferencePlanes,
};
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
    /// The sixteen per-sub-block vectors (§16.4 `this->split.mvs[16]`) when
    /// this MB was coded SPLITMV; `None` for every other mode. Read by the
    /// §20.11 `above_block_mv` / `left_block_mv` neighbour lookup when the
    /// *next* macroblock is itself SPLITMV: a non-split neighbour falls back
    /// to its whole-MB [`mv`](MbInfo::mv), so this is only populated when
    /// [`is_split`](MbInfo::is_split) is `true`.
    pub split_mvs: Option<[Mv; 16]>,
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
            split_mvs: None,
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

// ───────────────────────── §16.4 SPLITMV ──────────────────────────────────

/// The four §16.4 partition shapes — RFC 6386 §16.4 `MVpartition` /
/// §20.13 `mv_partitions[4][16]`.
///
/// Each variant subdivides the 4×4 grid of Y sub-blocks (raster-indexed
/// 0..=15) into a fixed group of partitions; the order they're decoded in
/// is the partition-id order (0 first, then 1, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvPartition {
    /// §16.4 `mv_top_bottom` — two pieces, sub-blocks `{0..7}` (top
    /// half) and `{8..15}` (bottom half). Partition tree code "110".
    TopBottom,
    /// §16.4 `mv_left_right` — two pieces, sub-blocks
    /// `{0,1,4,5,8,9,12,13}` (left half) and
    /// `{2,3,6,7,10,11,14,15}` (right half). Partition tree code "111".
    LeftRight,
    /// §16.4 `mv_quarters` — four 2×2 quadrants
    /// (`{0,1,4,5}`, `{2,3,6,7}`, `{8,9,12,13}`, `{10,11,14,15}`).
    /// Partition tree code "10".
    Quarters,
    /// §16.4 `MV_16` — every sub-block carries its own vector. Partition
    /// tree code "0".
    Mv16,
}

/// `mv_partitions[4][16]` — RFC 6386 §16.4 / §20.13.
///
/// Indexed by `partition_id` (0=TopBottom, 1=LeftRight, 2=Quarters,
/// 3=Mv16) and sub-block raster index (0..=15). Each entry is the
/// partition-group id this sub-block belongs to; sub-blocks with the same
/// group id share a vector.
///
/// Transcribed verbatim from §20.13.
pub const MV_PARTITIONS: [[u8; 16]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];

/// `mvpartition_tree` — RFC 6386 §16.4 / §20.13 `split_mv_tree`.
///
/// `{-3, 2, -2, 4, -0, -1}` — leaf `-3` is `Mv16`, `-2` `Quarters`, `-0`
/// `TopBottom`, `-1` `LeftRight`. Tree node at offset `i` reads
/// `MV_PARTITION_PROBS[i >> 1]`.
pub const MV_PARTITION_TREE: [i8; 6] = [-3, 2, -2, 4, -0, -1];

/// `mvpartition_probs[3]` — RFC 6386 §16.4 / §20.13 `split_mv_probs`.
///
/// `{110, 111, 150}` — the fixed probability table the partition tree is
/// read against. There is no probability-update for this tree (the §17
/// `mv_prob_update()` updates only the per-component contexts).
pub const MV_PARTITION_PROBS: [u8; 3] = [110, 111, 150];

/// The four §16.4 sub-block inter-prediction modes — RFC 6386 §16.4
/// `sub_mv_ref` enumeration, in [`SUBMV_REF_TREE`] leaf order.
///
/// Each variant says how the per-sub-block vector is sourced:
///
/// * [`Left4x4`](SubMvRefMode::Left4x4) — copy the neighbour MV
///   immediately to the left of the partition's first sub-block.
/// * [`Above4x4`](SubMvRefMode::Above4x4) — copy the neighbour MV above
///   the partition's first sub-block.
/// * [`Zero4x4`](SubMvRefMode::Zero4x4) — zero motion vector.
/// * [`New4x4`](SubMvRefMode::New4x4) — `read_mv` differential added
///   component-wise to the clamped "best" base from `find_near_mvs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubMvRefMode {
    /// `LEFT4X4` — leaf "0" of [`SUBMV_REF_TREE`].
    Left4x4,
    /// `ABOVE4X4` — leaf "10".
    Above4x4,
    /// `ZERO4X4` — leaf "110".
    Zero4x4,
    /// `NEW4X4` — leaf "111".
    New4x4,
}

/// `sub_mv_ref_tree[6]` — RFC 6386 §16.4 / §20.13.
///
/// `{-LEFT4X4, 2, -ABOVE4X4, 4, -ZERO4X4, -NEW4X4}`, with the leaves
/// remapped to [`SubMvRefMode`] discriminant indices
/// (`Left4x4=0`, `Above4x4=1`, `Zero4x4=2`, `New4x4=3`).
pub const SUBMV_REF_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

/// `sub_mv_ref_prob[5][3]` — RFC 6386 §16.4 / §20.13 `submv_ref_probs2`.
///
/// Indexed by context (0..=4, see [`submv_ref_context`]), with three
/// entries each feeding [`SUBMV_REF_TREE`]'s three internal nodes.
/// Transcribed verbatim from §16.4 / §20.13.
pub const SUBMV_REF_PROBS: [[u8; 3]; 5] = [
    [147, 136, 18], // SUBMVREF_NORMAL
    [106, 145, 1],  // SUBMVREF_LEFT_ZED
    [179, 121, 1],  // SUBMVREF_ABOVE_ZED
    [223, 1, 34],   // SUBMVREF_LEFT_ABOVE_SAME
    [208, 1, 1],    // SUBMVREF_LEFT_ABOVE_ZED
];

/// `vp8_mvCont(l, a)` — RFC 6386 §16.4 sub-block MV context.
///
/// Picks one of five [`SUBMV_REF_PROBS`] rows from the left + above
/// sub-block neighbour vectors:
///
/// * Both zero and equal → `4` (`SUBMVREF_LEFT_ABOVE_ZED`).
/// * Equal (and non-zero) → `3` (`SUBMVREF_LEFT_ABOVE_SAME`).
/// * Above zero (left non-zero, unequal) → `2` (`SUBMVREF_ABOVE_ZED`).
/// * Left zero (above non-zero, unequal) → `1` (`SUBMVREF_LEFT_ZED`).
/// * Otherwise → `0` (`SUBMVREF_NORMAL`).
#[inline]
pub fn submv_ref_context(left: Mv, above: Mv) -> usize {
    let lez = left == Mv::default();
    let aez = above == Mv::default();
    let lea = left == above;
    if lea && lez {
        4
    } else if lea {
        3
    } else if aez {
        2
    } else if lez {
        1
    } else {
        0
    }
}

/// Walk the §16.4 `sub_mv_ref_tree` to read a sub-block mode — RFC 6386
/// §16.4 / §20.11 `submv_ref`.
///
/// Picks the context from `(left, above)`, then `bool_read_tree` against
/// [`SUBMV_REF_PROBS`].
pub fn submv_ref(
    dec: &mut BoolDecoder<'_>,
    left: Mv,
    above: Mv,
) -> Result<SubMvRefMode, BoolDecoderError> {
    let ctx = submv_ref_context(left, above);
    let probs = &SUBMV_REF_PROBS[ctx];
    let mut i: usize = 0;
    loop {
        let prob = probs[i >> 1];
        let bit = dec.read_bool(prob)? as usize;
        let next = SUBMV_REF_TREE[i + bit];
        if next <= 0 {
            return Ok(match -next {
                0 => SubMvRefMode::Left4x4,
                1 => SubMvRefMode::Above4x4,
                2 => SubMvRefMode::Zero4x4,
                _ => SubMvRefMode::New4x4,
            });
        }
        i = next as usize;
    }
}

/// Walk the §16.4 `mvpartition_tree` to read a partition id — RFC 6386
/// §16.4 / §20.11 `bool_read_tree(split_mv_tree, split_mv_probs)`.
///
/// `{-3, 2, -2, 4, -0, -1}` encodes `Mv16 = "0"`, `Quarters = "10"`,
/// `TopBottom = "110"`, `LeftRight = "111"`.
pub fn read_mv_partition(dec: &mut BoolDecoder<'_>) -> Result<MvPartition, BoolDecoderError> {
    let mut i: usize = 0;
    loop {
        let prob = MV_PARTITION_PROBS[i >> 1];
        let bit = dec.read_bool(prob)? as usize;
        let next = MV_PARTITION_TREE[i + bit];
        if next <= 0 {
            // Leaves encode the partition id in their absolute value:
            // -0 = TopBottom, -1 = LeftRight, -2 = Quarters, -3 = Mv16.
            return Ok(match -next {
                0 => MvPartition::TopBottom,
                1 => MvPartition::LeftRight,
                2 => MvPartition::Quarters,
                _ => MvPartition::Mv16,
            });
        }
        i = next as usize;
    }
}

/// Look up the §20.11 `above_block_mv` neighbour vector for sub-block `b`.
///
/// * `b < 4` (top row of the current MB) — the neighbour is in the above
///   MB. If the above MB is SPLITMV, its bottom row sub-block `b + 12`
///   provides the vector; otherwise its whole-MB `mv`. An intra above MB
///   contributes zero (per §16.4 "subblocks within an intra-predicted
///   macroblock take their MV to be zero").
/// * `b >= 4` — the neighbour is sub-block `b - 4` of the current MB,
///   which (per the §16.4 in-order constraint) is already filled.
#[inline]
pub fn above_block_mv(this_split: &[Mv; 16], above: &MbInfo, b: usize) -> Mv {
    if b < 4 {
        if above.ref_frame.is_none() {
            return Mv::default();
        }
        if above.is_split {
            if let Some(ref mvs) = above.split_mvs {
                return mvs[b + 12];
            }
            // SPLITMV neighbour but no per-sub-block detail surfaced — the
            // §16.4 fallback is the whole-MB vector.
            return above.mv;
        }
        above.mv
    } else {
        this_split[b - 4]
    }
}

/// Look up the §20.11 `left_block_mv` neighbour vector for sub-block `b`.
///
/// * `b & 3 == 0` (left column of the current MB) — the neighbour is in
///   the left MB. SPLITMV left neighbour: sub-block `b + 3`; otherwise the
///   left MB's whole-MB `mv`. Intra left MB contributes zero.
/// * Otherwise — sub-block `b - 1` of the current MB.
#[inline]
pub fn left_block_mv(this_split: &[Mv; 16], left: &MbInfo, b: usize) -> Mv {
    if b & 3 == 0 {
        if left.ref_frame.is_none() {
            return Mv::default();
        }
        if left.is_split {
            if let Some(ref mvs) = left.split_mvs {
                return mvs[b + 3];
            }
            return left.mv;
        }
        left.mv
    } else {
        this_split[b - 1]
    }
}

/// The §16.4 SPLITMV decode result: the resolved partition shape and the
/// sixteen per-sub-block vectors.
///
/// `partition` is the [`MvPartition`] read from [`read_mv_partition`];
/// `split_mvs` is the §20.11 `this->split.mvs[16]` array filled by the
/// partition walk (sub-blocks belonging to the same partition share a
/// vector; for `Mv16` every entry is distinct).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMvResult {
    /// The resolved partition shape.
    pub partition: MvPartition,
    /// The sixteen per-sub-block vectors (quarter-pixel), in raster order.
    pub split_mvs: [Mv; 16],
}

/// Decode one SPLITMV macroblock's per-sub-block motion vectors — RFC 6386
/// §16.4 / §20.11 `decode_split_mv`.
///
/// Steps (verbatim from §20.11):
///
/// 1. Read the partition id via [`read_mv_partition`].
/// 2. For each partition group `j` in order, find the first sub-block
///    `k` whose partition entry is `j` (the "anchor" sub-block).
/// 3. Resolve the anchor's left and above neighbour MVs via
///    [`left_block_mv`] / [`above_block_mv`] (these consult earlier
///    sub-blocks of the current MB plus the [`MbInfo`] neighbours).
/// 4. Run [`submv_ref`] to read the sub-block mode and produce the
///    partition's vector:
///    * `LEFT4X4` → left neighbour, `ABOVE4X4` → above neighbour,
///    * `ZERO4X4` → `(0, 0)`,
///    * `NEW4X4` → `read_mv(mv_contexts)` differential added
///      component-wise to `best_mv` (the §18.1-clamped "best" predictor
///      from the prior `find_near_mvs`).
/// 5. Write the partition's vector into every sub-block belonging to
///    group `j`.
///
/// `best_mv` is the clamped "best" predictor (the
/// [`ResolvedInterMode::Split::best`] field). `mv_contexts` is the
/// resolved §17 MV component probability pair.
pub fn decode_split_mv(
    dec: &mut BoolDecoder<'_>,
    above: &MbInfo,
    left: &MbInfo,
    best_mv: Mv,
    mv_contexts: &MvContexts,
) -> Result<SplitMvResult, BoolDecoderError> {
    let partition = read_mv_partition(dec)?;
    let part_id = partition_id(partition);
    let table = &MV_PARTITIONS[part_id];

    // `num_groups` is how many distinct partition groups this shape carries
    // — 2 for top/bottom or left/right, 4 for quarters, 16 for MV_16.
    let num_groups: usize = match partition {
        MvPartition::TopBottom | MvPartition::LeftRight => 2,
        MvPartition::Quarters => 4,
        MvPartition::Mv16 => 16,
    };

    let mut split_mvs = [Mv::default(); 16];
    let mut filled = [false; 16];

    for j in 0..num_groups {
        // Find the first sub-block belonging to partition group `j`.
        // (`mv_partitions` is laid out so that group ids appear in order:
        // group 0's first member is always at the lowest index that hasn't
        // been claimed; the `for (k = 0; j != partition[k]; k++)` in
        // §20.11 finds it directly.)
        let mut k: usize = 0;
        while table[k] as usize != j {
            k += 1;
            debug_assert!(k < 16, "partition group {j} has no member in table");
        }

        // Resolve neighbour vectors at sub-block `k` and pick a mode.
        let left_mv = left_block_mv(&split_mvs, left, k);
        let above_mv = above_block_mv(&split_mvs, above, k);
        let mode = submv_ref(dec, left_mv, above_mv)?;

        let mv = match mode {
            SubMvRefMode::Left4x4 => left_mv,
            SubMvRefMode::Above4x4 => above_mv,
            SubMvRefMode::Zero4x4 => Mv::default(),
            SubMvRefMode::New4x4 => {
                let diff = read_mv(dec, mv_contexts)?;
                Mv {
                    row: best_mv.row.wrapping_add(diff.row),
                    col: best_mv.col.wrapping_add(diff.col),
                }
            }
        };

        // Fill every sub-block in this partition group.
        for (idx, &g) in table.iter().enumerate() {
            if g as usize == j {
                split_mvs[idx] = mv;
                filled[idx] = true;
            }
        }
    }

    debug_assert!(filled.iter().all(|&f| f), "every sub-block must be filled");

    Ok(SplitMvResult {
        partition,
        split_mvs,
    })
}

/// The partition-id of an [`MvPartition`] (its index into
/// [`MV_PARTITIONS`]).
#[inline]
pub fn partition_id(p: MvPartition) -> usize {
    match p {
        MvPartition::TopBottom => 0,
        MvPartition::LeftRight => 1,
        MvPartition::Quarters => 2,
        MvPartition::Mv16 => 3,
    }
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

/// Decode and reconstruct one SPLITMV-predicted macroblock end-to-end —
/// RFC 6386 §16.2 / §16.3 / §16.4 / §18.
///
/// The SPLITMV analogue of [`decode_inter_mb`]: runs the §16.3 census +
/// §16.2 inter-mode tree, asserts the resolved mode is SPLITMV (errors
/// otherwise — the caller dispatches to [`decode_inter_mb`] for whole-MB
/// modes), then runs the §16.4 partition + per-sub-block walk
/// ([`decode_split_mv`]) and the §18 reconstruction
/// ([`reconstruct_split_mv_mb`]).
///
/// Returns the reconstructed pixels and the [`SplitMvResult`] (caller
/// stores `split_mvs[15]` as the MB's `mv` and `Some(split_mvs)` as the
/// next neighbour's [`MbInfo::split_mvs`]).
///
/// `full_pixel` is the version-3 full-pel-chroma flag; `filters` the
/// §20.14 version-selected tap set; the coefficient arrays are
/// pre-dequantized (matching the [`decode_inter_mb`] convention).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct §16/§18 input.
pub fn decode_split_mv_mb(
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
    y_coeffs_dequant: &[[i16; 16]; 16],
    u_coeffs_dequant: &[[i16; 16]; 4],
    v_coeffs_dequant: &[[i16; 16]; 4],
) -> Result<(ReconstructedMb, SplitMvResult), InterMbError> {
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

    let best = match resolved {
        ResolvedInterMode::Split { best } => best,
        ResolvedInterMode::Whole { .. } => {
            // Programmer error: the caller should have routed a non-Split
            // resolution to `decode_inter_mb`. Surface a clean error so a
            // mis-dispatch is caught loudly.
            return Err(InterMbError::SplitNotSupported {
                best: Mv::default(),
            });
        }
    };

    let split = decode_split_mv(dec, above, left, best, mv_contexts)?;

    let recon = reconstruct_split_mv_mb(
        reference,
        mb_col,
        mb_row,
        &split.split_mvs,
        full_pixel,
        filters,
        mb_skip_coeff,
        y_coeffs_dequant,
        u_coeffs_dequant,
        v_coeffs_dequant,
    );

    Ok((recon, split))
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

        /// Emit the §16.4 `mvpartition_tree` path for `partition`.
        fn write_mv_partition(&mut self, partition: MvPartition) {
            // Target leaf value (matches the negative leaves in
            // MV_PARTITION_TREE: -3 Mv16, -2 Quarters, -0 TopBottom, -1 LR).
            let target: i8 = match partition {
                MvPartition::Mv16 => -3,
                MvPartition::Quarters => -2,
                MvPartition::TopBottom => 0, // leaf value is "-0" = 0
                MvPartition::LeftRight => -1,
            };
            // The tree has a "-0" leaf (TopBottom), which collides numerically
            // with internal node 0. Walk explicitly: tree is fixed and small.
            // {-3, 2, -2, 4, -0, -1}: Mv16 = "0", Quarters = "10",
            // TopBottom = "110", LeftRight = "111".
            let bits: &[bool] = match partition {
                MvPartition::Mv16 => &[false],
                MvPartition::Quarters => &[true, false],
                MvPartition::TopBottom => &[true, true, false],
                MvPartition::LeftRight => &[true, true, true],
            };
            // Each bit reads MV_PARTITION_PROBS[i>>1]; node indices walked:
            // start 0; bit=true → next at offset 2; bit=true → next at
            // offset 4.
            let mut i: usize = 0;
            for &bit in bits {
                self.write_bool(MV_PARTITION_PROBS[i >> 1], bit);
                let next = MV_PARTITION_TREE[i + bit as usize];
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
            let _ = target; // documented above; explicit walk avoids it.
        }

        /// Emit the §16.4 `sub_mv_ref_tree` path for `mode` under
        /// `(left, above)` context — uses the same `bool_read_tree` shape as
        /// the decoder.
        fn write_submv_ref(&mut self, left: Mv, above: Mv, mode: SubMvRefMode) {
            let ctx = submv_ref_context(left, above);
            let probs = &SUBMV_REF_PROBS[ctx];
            // Tree: {-0 LEFT, 2, -1 ABOVE, 4, -2 ZERO, -3 NEW}
            // LEFT = "0", ABOVE = "10", ZERO = "110", NEW = "111".
            let bits: &[bool] = match mode {
                SubMvRefMode::Left4x4 => &[false],
                SubMvRefMode::Above4x4 => &[true, false],
                SubMvRefMode::Zero4x4 => &[true, true, false],
                SubMvRefMode::New4x4 => &[true, true, true],
            };
            let mut i: usize = 0;
            for &bit in bits {
                self.write_bool(probs[i >> 1], bit);
                let next = SUBMV_REF_TREE[i + bit as usize];
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }
    }

    fn inter(ref_frame: RefFrame, mv: Mv) -> MbInfo {
        MbInfo {
            ref_frame: Some(ref_frame),
            mv,
            is_split: false,
            split_mvs: None,
        }
    }

    fn split(ref_frame: RefFrame, mv: Mv) -> MbInfo {
        MbInfo {
            ref_frame: Some(ref_frame),
            mv,
            is_split: true,
            split_mvs: None,
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

    // ---- §16.4 SPLITMV: tables and tree shape -----------------------

    #[test]
    fn mv_partitions_matches_spec() {
        // §20.13 mv_partitions[4][16]: transcribed verbatim.
        assert_eq!(
            MV_PARTITIONS[0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(
            MV_PARTITIONS[1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        );
        assert_eq!(
            MV_PARTITIONS[2],
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
        );
        assert_eq!(
            MV_PARTITIONS[3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn mv_partition_tree_matches_spec() {
        // §20.13 split_mv_tree = {-3, 2, -2, 4, -0, -1}.
        assert_eq!(MV_PARTITION_TREE, [-3, 2, -2, 4, 0, -1]);
    }

    #[test]
    fn mv_partition_probs_matches_spec() {
        // §16.4 mvpartition_probs / §20.13 split_mv_probs = {110, 111, 150}.
        assert_eq!(MV_PARTITION_PROBS, [110, 111, 150]);
    }

    #[test]
    fn submv_ref_tree_matches_spec() {
        // §20.13 sub_mv_ref_tree = {-LEFT4X4, 2, -ABOVE4X4, 4, -ZERO4X4,
        // -NEW4X4}, leaves remapped to SubMvRefMode discriminant indices.
        assert_eq!(SUBMV_REF_TREE, [0, 2, -1, 4, -2, -3]);
    }

    #[test]
    fn submv_ref_probs_matches_spec() {
        // §20.13 submv_ref_probs2[5][3] — verbatim from the listing.
        assert_eq!(SUBMV_REF_PROBS[0], [147, 136, 18]);
        assert_eq!(SUBMV_REF_PROBS[1], [106, 145, 1]);
        assert_eq!(SUBMV_REF_PROBS[2], [179, 121, 1]);
        assert_eq!(SUBMV_REF_PROBS[3], [223, 1, 34]);
        assert_eq!(SUBMV_REF_PROBS[4], [208, 1, 1]);
    }

    #[test]
    fn submv_ref_context_picks_five_buckets() {
        let zero = Mv::default();
        let a = Mv { row: 4, col: 0 };
        let b = Mv { row: 0, col: 4 };
        // Both zero, equal → 4 (LEFT_ABOVE_ZED).
        assert_eq!(submv_ref_context(zero, zero), 4);
        // Equal, non-zero → 3 (LEFT_ABOVE_SAME).
        assert_eq!(submv_ref_context(a, a), 3);
        // Above zero, left non-zero, unequal → 2 (ABOVE_ZED).
        assert_eq!(submv_ref_context(a, zero), 2);
        // Left zero, above non-zero, unequal → 1 (LEFT_ZED).
        assert_eq!(submv_ref_context(zero, a), 1);
        // Both non-zero, unequal → 0 (NORMAL).
        assert_eq!(submv_ref_context(a, b), 0);
    }

    // ---- §16.4 SPLITMV: partition tree round-trip -------------------

    fn roundtrip_partition(p: MvPartition) -> MvPartition {
        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(p);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        read_mv_partition(&mut dec).unwrap()
    }

    #[test]
    fn every_partition_round_trips() {
        for p in [
            MvPartition::TopBottom,
            MvPartition::LeftRight,
            MvPartition::Quarters,
            MvPartition::Mv16,
        ] {
            assert_eq!(roundtrip_partition(p), p, "partition {:?}", p);
        }
    }

    // ---- §16.4 SPLITMV: submv_ref round-trip ------------------------

    fn roundtrip_submv_ref(left: Mv, above: Mv, mode: SubMvRefMode) -> SubMvRefMode {
        let mut enc = BoolEncoder::new();
        enc.write_submv_ref(left, above, mode);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        submv_ref(&mut dec, left, above).unwrap()
    }

    #[test]
    fn every_submv_ref_mode_round_trips() {
        // Use a "normal" context (left and above distinct non-zero).
        let left = Mv { row: 4, col: 0 };
        let above = Mv { row: 0, col: 4 };
        for mode in [
            SubMvRefMode::Left4x4,
            SubMvRefMode::Above4x4,
            SubMvRefMode::Zero4x4,
            SubMvRefMode::New4x4,
        ] {
            assert_eq!(roundtrip_submv_ref(left, above, mode), mode);
        }
    }

    #[test]
    fn submv_ref_round_trips_under_all_contexts() {
        // Exercise each of the five context rows by constructing l/a pairs
        // that fall into each bucket.
        let zero = Mv::default();
        let a = Mv { row: 4, col: 0 };
        let b = Mv { row: 0, col: 4 };
        let cases: &[(Mv, Mv)] = &[
            (a, b),       // 0 NORMAL
            (zero, a),    // 1 LEFT_ZED
            (a, zero),    // 2 ABOVE_ZED
            (a, a),       // 3 LEFT_ABOVE_SAME
            (zero, zero), // 4 LEFT_ABOVE_ZED
        ];
        for (left, above) in cases {
            for mode in [
                SubMvRefMode::Left4x4,
                SubMvRefMode::Above4x4,
                SubMvRefMode::Zero4x4,
                SubMvRefMode::New4x4,
            ] {
                assert_eq!(roundtrip_submv_ref(*left, *above, mode), mode);
            }
        }
    }

    // ---- §16.4 SPLITMV: neighbour MV lookups ------------------------

    #[test]
    fn above_block_mv_intra_neighbour_is_zero() {
        let split_mvs = [Mv::default(); 16];
        let intra = MbInfo::border();
        // Top row sub-blocks: b in 0..4 read from neighbour; intra → zero.
        for b in 0..4 {
            assert_eq!(above_block_mv(&split_mvs, &intra, b), Mv::default());
        }
    }

    #[test]
    fn above_block_mv_non_split_neighbour_uses_whole_mv() {
        let split_mvs = [Mv::default(); 16];
        let v = Mv { row: 8, col: -4 };
        let above = inter(RefFrame::Last, v);
        for b in 0..4 {
            assert_eq!(above_block_mv(&split_mvs, &above, b), v);
        }
    }

    #[test]
    fn above_block_mv_split_neighbour_uses_bottom_row() {
        let split_mvs = [Mv::default(); 16];
        // Build a SPLITMV above with a distinct vector per sub-block.
        let above_split: [Mv; 16] = core::array::from_fn(|k| Mv {
            row: k as i16,
            col: -(k as i16),
        });
        let above = MbInfo {
            ref_frame: Some(RefFrame::Last),
            mv: above_split[15],
            is_split: true,
            split_mvs: Some(above_split),
        };
        for b in 0..4 {
            // Above MB's bottom row is sub-blocks 12..15, mapped by b+12.
            assert_eq!(above_block_mv(&split_mvs, &above, b), above_split[b + 12]);
        }
    }

    #[test]
    fn above_block_mv_internal_uses_current_mb() {
        let split_mvs: [Mv; 16] = core::array::from_fn(|k| Mv {
            row: (k * 2) as i16,
            col: (k * 3) as i16,
        });
        let above = MbInfo::border();
        // b >= 4 reads from current MB's sub-block b - 4.
        for b in 4..16 {
            assert_eq!(above_block_mv(&split_mvs, &above, b), split_mvs[b - 4]);
        }
    }

    #[test]
    fn left_block_mv_intra_neighbour_is_zero() {
        let split_mvs = [Mv::default(); 16];
        let intra = MbInfo::border();
        // Left column sub-blocks: b ∈ {0, 4, 8, 12} read from neighbour.
        for &b in &[0usize, 4, 8, 12] {
            assert_eq!(left_block_mv(&split_mvs, &intra, b), Mv::default());
        }
    }

    #[test]
    fn left_block_mv_split_neighbour_uses_right_column() {
        let split_mvs = [Mv::default(); 16];
        let left_split: [Mv; 16] = core::array::from_fn(|k| Mv {
            row: -(k as i16),
            col: k as i16,
        });
        let left = MbInfo {
            ref_frame: Some(RefFrame::Last),
            mv: left_split[15],
            is_split: true,
            split_mvs: Some(left_split),
        };
        // Left MB's right column is sub-blocks 3, 7, 11, 15 → b + 3 for
        // b = 0, 4, 8, 12.
        for &b in &[0usize, 4, 8, 12] {
            assert_eq!(left_block_mv(&split_mvs, &left, b), left_split[b + 3]);
        }
    }

    #[test]
    fn left_block_mv_internal_uses_current_mb() {
        let split_mvs: [Mv; 16] = core::array::from_fn(|k| Mv {
            row: (k * 5) as i16,
            col: (k * 7) as i16,
        });
        let left = MbInfo::border();
        // b & 3 != 0 reads from current MB's sub-block b - 1.
        for b in 0..16 {
            if b & 3 == 0 {
                continue;
            }
            assert_eq!(left_block_mv(&split_mvs, &left, b), split_mvs[b - 1]);
        }
    }

    // ---- §16.4 SPLITMV: decode_split_mv shape coverage --------------

    /// All-ZERO4x4 SPLITMV: every partition picks ZERO so the result is
    /// sixteen zero vectors regardless of best/neighbours.
    fn decode_all_zero_split(partition: MvPartition, num_groups: usize) -> SplitMvResult {
        let above = MbInfo::border();
        let left = MbInfo::border();
        let best = Mv { row: 64, col: 64 }; // ignored by ZERO4x4
        let mv_ctx = default_contexts();

        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(partition);
        // For each partition group, the anchor's neighbours are zero (both
        // l and a come from the all-border neighbours and unfilled current
        // sub-blocks default-to-zero). So context = 4 (LEFT_ABOVE_ZED).
        for _ in 0..num_groups {
            enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::Zero4x4);
        }
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        decode_split_mv(&mut dec, &above, &left, best, &mv_ctx).unwrap()
    }

    #[test]
    fn decode_split_mv_top_bottom_all_zero() {
        let res = decode_all_zero_split(MvPartition::TopBottom, 2);
        assert_eq!(res.partition, MvPartition::TopBottom);
        for mv in res.split_mvs.iter() {
            assert_eq!(*mv, Mv::default());
        }
    }

    #[test]
    fn decode_split_mv_left_right_all_zero() {
        let res = decode_all_zero_split(MvPartition::LeftRight, 2);
        assert_eq!(res.partition, MvPartition::LeftRight);
        for mv in res.split_mvs.iter() {
            assert_eq!(*mv, Mv::default());
        }
    }

    #[test]
    fn decode_split_mv_quarters_all_zero() {
        let res = decode_all_zero_split(MvPartition::Quarters, 4);
        assert_eq!(res.partition, MvPartition::Quarters);
        for mv in res.split_mvs.iter() {
            assert_eq!(*mv, Mv::default());
        }
    }

    #[test]
    fn decode_split_mv_mv16_all_zero() {
        let res = decode_all_zero_split(MvPartition::Mv16, 16);
        assert_eq!(res.partition, MvPartition::Mv16);
        for mv in res.split_mvs.iter() {
            assert_eq!(*mv, Mv::default());
        }
    }

    // ---- §16.4 SPLITMV: per-mode semantics --------------------------

    #[test]
    fn decode_split_mv_top_bottom_zero_new() {
        // TopBottom: group 0 (top half, sub-blocks 0..7) picks ZERO; group 1
        // (bottom half, sub-blocks 8..15) picks NEW with a small diff.
        let above = MbInfo::border();
        let left = MbInfo::border();
        let best = Mv { row: 4, col: 8 };
        let diff = Mv { row: 4, col: -4 };
        let mv_ctx = default_contexts();

        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(MvPartition::TopBottom);
        // Group 0 (k=0): l=zero, a=zero, ctx=4 → ZERO4x4.
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::Zero4x4);
        // Group 1 (k=8): l=zero (border), a=sub-block 4 = zero (filled by
        // group 0). ctx=4 → NEW4x4 with diff.
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::New4x4);
        enc.write_mv(&mv_ctx, diff);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = decode_split_mv(&mut dec, &above, &left, best, &mv_ctx).unwrap();
        let expected_new = Mv {
            row: best.row + diff.row,
            col: best.col + diff.col,
        };
        // Top half zero, bottom half NEW.
        for b in 0..8 {
            assert_eq!(res.split_mvs[b], Mv::default(), "top sub-block {}", b);
        }
        for b in 8..16 {
            assert_eq!(res.split_mvs[b], expected_new, "bottom sub-block {}", b);
        }
    }

    #[test]
    fn decode_split_mv_top_bottom_above4x4() {
        // TopBottom: group 0 picks ABOVE4x4 from the above neighbour MB
        // (a non-split neighbour with a known whole-MB vector).
        let above_mv = Mv { row: 16, col: -8 };
        let above = inter(RefFrame::Last, above_mv);
        let left = MbInfo::border();
        let best = Mv { row: 4, col: 8 };
        let mv_ctx = default_contexts();

        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(MvPartition::TopBottom);
        // Group 0 (k=0): l=zero (border), a=above_mv (non-split neighbour).
        // Context: lez=true, aez=false, lea=false → ctx = 1 (LEFT_ZED).
        // Encoder emits "10" path under that context.
        enc.write_submv_ref(Mv::default(), above_mv, SubMvRefMode::Above4x4);
        // Group 1 (k=8): l=zero (border), a=sub-block 4 = above_mv (just
        // filled). ctx: lez=true, aez=false, lea=false → 1. Pick ZERO4x4.
        enc.write_submv_ref(Mv::default(), above_mv, SubMvRefMode::Zero4x4);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = decode_split_mv(&mut dec, &above, &left, best, &mv_ctx).unwrap();
        for b in 0..8 {
            assert_eq!(res.split_mvs[b], above_mv, "top sub-block {}", b);
        }
        for b in 8..16 {
            assert_eq!(res.split_mvs[b], Mv::default(), "bottom sub-block {}", b);
        }
    }

    #[test]
    fn decode_split_mv_left_right_left4x4() {
        // LeftRight: group 0 (left half) picks LEFT4x4 from the left
        // neighbour MB.
        let left_mv = Mv { row: -4, col: 16 };
        let left = inter(RefFrame::Last, left_mv);
        let above = MbInfo::border();
        let best = Mv { row: 4, col: 8 };
        let mv_ctx = default_contexts();

        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(MvPartition::LeftRight);
        // Group 0 (k=0): l=left_mv, a=zero (border).
        // Context: lez=false, aez=true → ctx=2 (ABOVE_ZED). LEFT4x4.
        enc.write_submv_ref(left_mv, Mv::default(), SubMvRefMode::Left4x4);
        // Group 1 (k=2): l=sub-block 1 (left_mv, just filled by group 0),
        // a=zero (border). Same ctx=2. Pick ZERO4x4.
        enc.write_submv_ref(left_mv, Mv::default(), SubMvRefMode::Zero4x4);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = decode_split_mv(&mut dec, &above, &left, best, &mv_ctx).unwrap();
        // Left half (cols 0..2 each row) = left_mv; right half (cols 2..4) = zero.
        let left_group: &[usize] = &[0, 1, 4, 5, 8, 9, 12, 13];
        let right_group: &[usize] = &[2, 3, 6, 7, 10, 11, 14, 15];
        for &b in left_group {
            assert_eq!(res.split_mvs[b], left_mv, "left sub-block {}", b);
        }
        for &b in right_group {
            assert_eq!(res.split_mvs[b], Mv::default(), "right sub-block {}", b);
        }
    }

    #[test]
    fn decode_split_mv_mv16_per_sub_block_new() {
        // MV_16: every sub-block carries its own NEW4x4 vector. Test that
        // anchor `k` for each group is the sub-block itself and the
        // decoded vectors land in raster order.
        let above = MbInfo::border();
        let left = MbInfo::border();
        let best = Mv { row: 0, col: 0 };
        let mv_ctx = default_contexts();

        // For deterministic ctx, encode neighbours that all start zero and
        // build progressively. Use NEW4x4 with a per-sub-block diff that
        // depends on the index so we can verify ordering.
        let diffs: [Mv; 16] = core::array::from_fn(|i| Mv {
            row: i as i16,
            col: -(i as i16),
        });

        let mut enc = BoolEncoder::new();
        enc.write_mv_partition(MvPartition::Mv16);
        // Walk anchor sub-blocks 0..16 in order; left/above neighbours are
        // the resolved per-sub-block MVs from earlier writes (the decoder
        // queries them as part of its `submv_ref` context derivation).
        let mut split_so_far = [Mv::default(); 16];
        for b in 0..16 {
            // Neighbours per left_block_mv / above_block_mv.
            let l_mv = if b & 3 == 0 {
                Mv::default() // left is border MB → zero
            } else {
                split_so_far[b - 1]
            };
            let a_mv = if b < 4 {
                Mv::default() // above is border MB → zero
            } else {
                split_so_far[b - 4]
            };
            enc.write_submv_ref(l_mv, a_mv, SubMvRefMode::New4x4);
            enc.write_mv(&mv_ctx, diffs[b]);
            split_so_far[b] = Mv {
                row: best.row + diffs[b].row,
                col: best.col + diffs[b].col,
            };
        }
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let res = decode_split_mv(&mut dec, &above, &left, best, &mv_ctx).unwrap();
        assert_eq!(res.partition, MvPartition::Mv16);
        for (b, diff) in diffs.iter().enumerate() {
            let expected = Mv {
                row: best.row + diff.row,
                col: best.col + diff.col,
            };
            assert_eq!(res.split_mvs[b], expected, "sub-block {}", b);
        }
    }

    // ---- §16.4 SPLITMV: end-to-end decode_split_mv_mb ---------------

    #[test]
    fn decode_split_mv_mb_zero_split_copies_colocated_block() {
        // A SPLITMV MB where every sub-block resolves to zero with
        // skip_coeff = true must reconstruct to the co-located reference
        // MB (the same property `decode_inter_mb_zeromv_copies_colocated`
        // tests for the whole-MB ZEROMV path).
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

        // Census neighbours: all intra (border). The CNT_SPLITMV bucket is
        // 0 and `find_near_mvs` returns slot 0 = (0,0); the inter-mode tree
        // still encodes a Split path against the resulting probabilities.
        let above = MbInfo::border();
        let left = MbInfo::border();
        let border = MbInfo::border();
        let cnt = find_near_mvs(&above, &left, &border, RefFrame::Last, SignBias::default()).cnt;
        let probs = mv_ref_probs(&cnt);
        let mv_ctx = default_contexts();

        // Encode: SPLITMV inter-mode + TopBottom partition + two ZERO4x4.
        // Both anchors see (left=zero, above=zero) → ctx 4 (LEFT_ABOVE_ZED).
        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Split);
        enc.write_mv_partition(MvPartition::TopBottom);
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::Zero4x4);
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::Zero4x4);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let (recon, split) = decode_split_mv_mb(
            &mut dec,
            &reference,
            1,
            1,
            &above,
            &left,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &mv_ctx,
            false,
            filters,
            true, // skip_coeff
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap();

        assert_eq!(split.partition, MvPartition::TopBottom);
        for mv in split.split_mvs.iter() {
            assert_eq!(*mv, Mv::default());
        }
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
    fn decode_split_mv_mb_top_bottom_distinct_halves() {
        // TopBottom SPLITMV with the top half ZERO (copies colocated) and
        // the bottom half NEW(diff) so the bottom-half luma sub-blocks come
        // from a shifted reference position. Use whole-pixel-after-doubling
        // vectors so we can compare exact byte values.
        // Best base from census = (0, 0) since neighbours are border, then
        // diff (4, 8) → final luma (4, 8) → doubled (8, 16) → luma offset
        // (1, 2), chroma offset (avg of 4 luma sub-block MVs).
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

        // Census: all-intra neighbours so best = (0, 0); the NEW4x4 diff
        // then equals the final per-sub-block vector.
        let above = MbInfo::border();
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
        let mv_ctx = default_contexts();

        // Encode SPLITMV / TopBottom / ZERO (top half) / NEW with diff.
        let diff = Mv { row: 4, col: 8 };
        let mut enc = BoolEncoder::new();
        enc.write_inter_mode(&probs, InterMode::Split);
        enc.write_mv_partition(MvPartition::TopBottom);
        // Group 0 (top, k=0): l=zero, a=zero → ctx=4. ZERO.
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::Zero4x4);
        // Group 1 (bottom, k=8): l=zero (border), a=sub-block 4 (zero,
        // just filled by top half). ctx=4. NEW.
        enc.write_submv_ref(Mv::default(), Mv::default(), SubMvRefMode::New4x4);
        enc.write_mv(&mv_ctx, diff);
        let bytes = enc.finish();

        let filters = crate::motion_comp::filter_set_for_version(0).taps();
        let coeffs = zero_coeffs();
        let mut dec = BoolDecoder::init(&bytes).unwrap();
        let (recon, split) = decode_split_mv_mb(
            &mut dec,
            &reference,
            1,
            1,
            &above,
            &border,
            &border,
            RefFrame::Last,
            SignBias::default(),
            &mv_ctx,
            false,
            filters,
            true, // skip_coeff
            &coeffs.y,
            &coeffs.u,
            &coeffs.v,
        )
        .unwrap();

        assert_eq!(split.partition, MvPartition::TopBottom);
        // Top half: zero vector, reference MB(1,1) verbatim. Rows 0..8.
        for r in 0..8 {
            for c in 0..16 {
                assert_eq!(
                    recon.y[r * 16 + c],
                    y[(16 + r) * 48 + (16 + c)],
                    "top y ({r},{c})"
                );
            }
        }
        // Bottom half: vector (4, 8) → doubled (8, 16) → whole-pixel offset
        // (1, 2). Rows 8..16 read from reference (16+1+r, 16+2+c).
        for r in 8..16 {
            for c in 0..16 {
                let ref_r = 16 + 1 + r;
                let ref_c = 16 + 2 + c;
                assert_eq!(
                    recon.y[r * 16 + c],
                    y[ref_r * 48 + ref_c],
                    "bot y ({r},{c})"
                );
            }
        }
        // Verify split_mvs[15] == diff (matches dixie `this->base.mv =
        // this->split.mvs[15]`).
        assert_eq!(split.split_mvs[15], diff);
    }

    // ---- §18.1 SPLITMV chroma averaging -----------------------------

    #[test]
    fn split_chroma_mvs_avg_matches_whole_mb_when_uniform() {
        use crate::motion_comp::{chroma_mv, split_chroma_mvs, stored_luma_mv};
        // With all sixteen luma vectors equal, §18.1 avg() across each
        // chroma slot must equal the whole-MB `chroma_mv` of the single
        // (stored-luma-doubled) vector.
        let mv = Mv { row: 10, col: -6 };
        let luma_split = [mv; 16];
        let split = split_chroma_mvs(&luma_split);
        let expected = chroma_mv(stored_luma_mv(mv));
        for (c, slot) in split.iter().enumerate() {
            assert_eq!(*slot, expected, "chroma slot {}", c);
        }
    }

    #[test]
    fn chroma_idx_for_luma_subblock_groups_match_spec() {
        // §18.1 enumeration: {0,1,4,5}→0, {2,3,6,7}→1, {8,9,12,13}→2,
        // {10,11,14,15}→3.
        use crate::motion_comp::chroma_idx_for_luma_subblock;
        let groups: [&[usize]; 4] = [
            &[0, 1, 4, 5],
            &[2, 3, 6, 7],
            &[8, 9, 12, 13],
            &[10, 11, 14, 15],
        ];
        for (slot, group) in groups.iter().enumerate() {
            for &b in *group {
                assert_eq!(
                    chroma_idx_for_luma_subblock(b),
                    slot,
                    "luma sub-block {} should map to chroma slot {}",
                    b,
                    slot
                );
            }
        }
    }
}
