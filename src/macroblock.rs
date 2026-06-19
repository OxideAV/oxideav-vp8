//! VP8 macroblock mode decoding for key frames (RFC 6386 §11) and the
//! intra-predicted branch of interframes (RFC 6386 §16.1).
//!
//! On a **key frame** this module wires up the structural mode-decision
//! layer that immediately follows the §19.2 frame header:
//!
//! 1. Optional `segment_id` (§10) gated by
//!    `segmentation_enabled && update_segmentation.update_mb_segmentation_map`.
//! 2. Optional `mb_skip_coeff` (§11.1) gated by `mb_no_skip_coeff`.
//! 3. Luma mode (§11.2) — `kf_ymode_tree` walk against the constant
//!    `KF_YMODE_PROB = {145, 156, 163, 128}` array.
//! 4. If the luma mode is [`IntraYMode::B`], sixteen sub-block modes
//!    (§11.3 / §11.5) — `bmode_tree` walks against the context-driven
//!    `KF_BMODE_PROB[above][left]` table.
//! 5. Chroma mode (§11.4) — `uv_mode_tree` walk against the constant
//!    `KF_UV_MODE_PROB = {142, 114, 183}` array.
//!
//! On an **interframe** the per-MB layout per §16.1 is structurally
//! analogous but the trees and probability tables differ:
//!
//! * The Y mode walks `ymode_tree` (root left-leaf is `DC_PRED`, not
//!   `B_PRED`) against the dynamic `IF_YMODE_PROB` whose defaults are
//!   `{112, 86, 140, 37}`; the frame header's
//!   `intra_y_mode_prob_update` (§9.10) may override all four entries.
//! * If the luma mode is [`IntraYMode::B`], the sixteen sub-block
//!   modes walk the same `bmode_tree` but against the fixed nine-entry
//!   `IF_BMODE_PROB = {120, 90, 79, 133, 87, 85, 80, 111, 151}` (no
//!   above / left context).
//! * The UV mode walks `uv_mode_tree` (same shape as on key frames)
//!   against the dynamic `IF_UV_MODE_PROB` whose defaults are
//!   `{162, 101, 204}`; the frame header's
//!   `intra_uv_mode_prob_update` may override all three entries.
//!
//! Per §16.1 final paragraph, the dynamic `IF_YMODE_PROB` and
//! `IF_UV_MODE_PROB` tables are **restored to their defaults whenever a
//! key frame is detected**; the
//! [`InterFrameIntraProbs::for_frame_header`] constructor handles that
//! reset together with the per-frame override.
//!
//! This round stays **structural**: it returns the decoded
//! [`MacroblockModes`] for every macroblock of the frame without
//! performing any pixel prediction (§12) or DCT-coefficient decode
//! (§13). The per-MB output is the input to the next round's
//! intra-prediction work.
//!
//! # Sub-block mode context
//!
//! Per §11.3, the probability array used to decode each 4×4 luma
//! sub-block's [`IntraBmode`] is indexed by the modes already coded
//! for the sub-block immediately above and immediately to the left of
//! the current one. Sub-blocks at the macroblock's top edge inherit
//! their "above" predictor from the bottom row of the macroblock
//! directly above; sub-blocks at the macroblock's left edge inherit
//! their "left" predictor from the right column of the macroblock
//! directly to the left.
//!
//! When the neighbouring macroblock used a 16×16 luma mode rather
//! than B_PRED, §11.3 item 4 defines a constant projection from that
//! 16×16 mode onto an [`IntraBmode`] (`DC→B_DC`, `V→B_VE`,
//! `H→B_HE`, `TM→B_TM`). On the very top row / left column of the
//! frame, missing predictors default to [`IntraBmode::Dc`]
//! (§11.3 item 3).
//!
//! # Reference
//!
//! RFC 6386 §11 in full (subsections 11.1 – 11.5). The KF Y / UV /
//! sub-block probability constants and the three coding trees are
//! transcribed verbatim from §11.2 (`kf_ymode_tree`,
//! `kf_ymode_prob`), §11.4 (`uv_mode_tree`, `kf_uv_mode_prob`), and
//! §11.5 (`kf_bmode_prob[10][10][9]`). The `bmode_tree` is from
//! §11.2's `bmode_tree` block, and `mb_segment_tree` is from §10.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::coded_header::Vp8CodedHeader;
use core::fmt;

/// 16×16 luma intra-prediction mode (RFC 6386 §11.2). The five
/// variants are listed in the order they take in the spec's
/// `intra_mbmode` enum (so [`IntraYMode::Dc`] is 0, [`IntraYMode::B`]
/// is 4); this matches the negative-leaf indices in `kf_ymode_tree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntraYMode {
    /// DC prediction (§12 `DC_PRED`).
    Dc,
    /// Vertical prediction (§12 `V_PRED`).
    V,
    /// Horizontal prediction (§12 `H_PRED`).
    H,
    /// "True Motion" prediction (§12 `TM_PRED`).
    Tm,
    /// Each Y sub-block is independently predicted (§11.3 `B_PRED`).
    B,
}

/// 8×8 chroma intra-prediction mode (RFC 6386 §11.4) — a subset of
/// [`IntraYMode`] with `B` removed. The four variants correspond to
/// `num_uv_modes = B_PRED`'s implicit enum range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntraUvMode {
    /// DC prediction.
    Dc,
    /// Vertical prediction.
    V,
    /// Horizontal prediction.
    H,
    /// "True Motion" prediction.
    Tm,
}

/// 4×4 luma sub-block intra-prediction mode (RFC 6386 §11.2). The
/// ten variants are listed in the order they take in the spec's
/// `intra_bmode` enum (so [`IntraBmode::Dc`] is 0). Used both as
/// the per-sub-block decoded mode and as the §11.3 context indexer
/// for the neighbouring sub-block predictors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntraBmode {
    /// Predict DC using row above and column to the left.
    Dc,
    /// "True Motion" — propagate second differences.
    Tm,
    /// Predict rows using row above.
    Ve,
    /// Predict columns using column to the left.
    He,
    /// Southwest 45° diagonal prediction.
    Ld,
    /// Southeast 45° diagonal prediction.
    Rd,
    /// SSE (vertical right) diagonal prediction.
    Vr,
    /// SSW (vertical left) diagonal prediction.
    Vl,
    /// ESE (horizontal down) diagonal prediction.
    Hd,
    /// ENE (horizontal up) diagonal prediction.
    Hu,
}

impl IntraUvMode {
    /// `uv_mode_tree` leaf code (RFC 6386 §11.4): the value
    /// `treed_read` recovers and the encoder writes via
    /// [`UV_MODE_TREE`]. `Dc` = 0 … `Tm` = 3.
    #[inline]
    pub(crate) fn leaf(self) -> u8 {
        match self {
            IntraUvMode::Dc => 0,
            IntraUvMode::V => 1,
            IntraUvMode::H => 2,
            IntraUvMode::Tm => 3,
        }
    }
}

impl IntraBmode {
    /// 0..=9 — the enum's spec-defined integer code. Used as the
    /// outer / middle index into `KF_BMODE_PROB` and as the
    /// `bmode_tree` leaf the encoder writes.
    #[inline]
    pub(crate) fn idx(self) -> usize {
        match self {
            IntraBmode::Dc => 0,
            IntraBmode::Tm => 1,
            IntraBmode::Ve => 2,
            IntraBmode::He => 3,
            IntraBmode::Ld => 4,
            IntraBmode::Rd => 5,
            IntraBmode::Vr => 6,
            IntraBmode::Vl => 7,
            IntraBmode::Hd => 8,
            IntraBmode::Hu => 9,
        }
    }
}

impl IntraYMode {
    /// `kf_ymode_tree` leaf code (RFC 6386 §11.2): the value
    /// `treed_read` recovers and the encoder writes via
    /// [`KF_YMODE_TREE`]. `Dc` = 0 … `B` = 4.
    #[inline]
    pub(crate) fn leaf(self) -> u8 {
        match self {
            IntraYMode::Dc => 0,
            IntraYMode::V => 1,
            IntraYMode::H => 2,
            IntraYMode::Tm => 3,
            IntraYMode::B => 4,
        }
    }

    /// §11.3 item 4 projection from a 16×16 luma mode onto the
    /// sub-block mode used as the context predictor for the
    /// neighbouring macroblock's edge sub-blocks. Note this is only
    /// invoked for non-`B_PRED` macroblocks; [`IntraYMode::B`]
    /// macroblocks store the actual sub-block modes themselves.
    #[inline]
    pub(crate) fn project_to_subblock_context(self) -> IntraBmode {
        match self {
            IntraYMode::Dc => IntraBmode::Dc,
            IntraYMode::V => IntraBmode::Ve,
            IntraYMode::H => IntraBmode::He,
            IntraYMode::Tm => IntraBmode::Tm,
            // §11.3 item 4 explicitly only lists the four 16×16
            // modes; B_PRED macroblocks never reach this branch
            // because their actual sub-block modes are recorded.
            IntraYMode::B => IntraBmode::Dc,
        }
    }
}

/// Decoded mode record for one macroblock on a key frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroblockModes {
    /// `segment_id` (§10). `None` when the frame did not enable
    /// `segmentation_enabled && update_mb_segmentation_map`.
    pub segment_id: Option<u8>,
    /// `mb_skip_coeff` (§11.1). `false` when `mb_no_skip_coeff` was
    /// `false` for the frame (in which case no bit is read; the
    /// value is forced to `false` per §11.1).
    pub mb_skip_coeff: bool,
    /// 16×16 luma mode (§11.2).
    pub y_mode: IntraYMode,
    /// 16 sub-block modes (§11.3 / §11.5), in raster order j=0..15.
    /// `Some(_)` if and only if `y_mode == IntraYMode::B`.
    pub subblock_modes: Option<[IntraBmode; 16]>,
    /// 8×8 chroma mode (§11.4).
    pub uv_mode: IntraUvMode,
}

/// Errors surfaced by [`parse_key_frame_macroblock_modes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroblockError {
    /// The bool decoder ran out of input partway through decoding
    /// the macroblock layer. Wraps the underlying [`BoolDecoderError`].
    BoolDecoder(BoolDecoderError),
    /// `parse_key_frame_macroblock_modes` was invoked on a header
    /// that has `key_frame == false`. This entry point only handles
    /// the §11 key-frame layout; interframes follow the §16 layout
    /// and will be wired up in a subsequent round.
    NotAKeyFrame,
}

impl fmt::Display for MacroblockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroblockError::BoolDecoder(inner) => {
                write!(f, "vp8 macroblock: {inner}")
            }
            MacroblockError::NotAKeyFrame => {
                write!(
                    f,
                    "vp8 macroblock: parse_key_frame_macroblock_modes invoked on an interframe"
                )
            }
        }
    }
}

impl std::error::Error for MacroblockError {}

impl From<BoolDecoderError> for MacroblockError {
    fn from(value: BoolDecoderError) -> Self {
        MacroblockError::BoolDecoder(value)
    }
}

/// `kf_ymode_tree` from RFC 6386 §11.2 (and §8 listing). Eight
/// signed entries — negative values are leaf-mode codes (negated to
/// recover the [`IntraYMode`] index) and positive values are
/// child-pair offsets to descend to.
pub(crate) const KF_YMODE_TREE: [i8; 8] = [
    // -B_PRED, 2 — root: B_PRED = "0", "1" subtree at [2..=3]
    -4, 2, // 4, 6 — "1" subtree has two descendant subtrees
    4, 6, // -DC_PRED, -V_PRED — "10" subtree
    0, -1, // -H_PRED, -TM_PRED — "11" subtree
    -2, -3,
];

/// `kf_ymode_prob` per §11.2.
pub(crate) const KF_YMODE_PROB: [u8; 4] = [145, 156, 163, 128];

/// `uv_mode_tree` from RFC 6386 §11.4. Six signed entries.
pub(crate) const UV_MODE_TREE: [i8; 6] = [
    // -DC_PRED, 2
    0, 2, // -V_PRED, 4
    -1, 4, // -H_PRED, -TM_PRED
    -2, -3,
];

/// `kf_uv_mode_prob` per §11.4.
pub(crate) const KF_UV_MODE_PROB: [u8; 3] = [142, 114, 183];

/// `bmode_tree` from RFC 6386 §11.2. Eighteen signed entries —
/// `2 * (num_intra_bmodes - 1) = 18`. The leaves cover all ten
/// [`IntraBmode`] values.
pub(crate) const BMODE_TREE: [i8; 18] = [
    // -B_DC_PRED, 2
    0, 2, // -B_TM_PRED, 4
    -1, 4, // -B_VE_PRED, 6
    -2, 6, // 8, 12 — two sub-subtrees
    8, 12, // -B_HE_PRED, 10
    -3, 10, // -B_RD_PRED, -B_VR_PRED
    -5, -6, // -B_LD_PRED, 14
    -4, 14, // -B_VL_PRED, 16
    -7, 16, // -B_HD_PRED, -B_HU_PRED
    -8, -9,
];

/// `mb_segment_tree` from RFC 6386 §10. Six entries — the segment
/// id is two bits walked through a balanced tree.
///
/// `pub(crate)` so the encoder's §11 mode-layer writer can drive
/// [`crate::encoder::BoolEncoder::write_treed`] against the same tree the
/// decoder walks in [`read_segment_id`], guaranteeing the per-MB
/// `segment_id` round-trips bit-for-bit.
pub(crate) const MB_SEGMENT_TREE: [i8; 6] = [
    // 2, 4
    2, 4, // -0, -1
    0, -1, // -2, -3
    -2, -3,
];

/// Walk `tree` using `dec`, reading at the probability supplied by
/// `prob_lookup(i)` where `i` is the current internal-node index
/// halved (matching the `p[i >> 1]` indexing in §8.1's
/// `treed_read`). Returns the leaf value (the negated final
/// `tree` entry).
fn treed_read<F>(
    dec: &mut BoolDecoder<'_>,
    tree: &[i8],
    prob_lookup: F,
) -> Result<u8, BoolDecoderError>
where
    F: Fn(usize) -> u8,
{
    let mut i: i8 = 0;
    loop {
        let prob = prob_lookup((i as usize) >> 1);
        let bit = dec.read_bool(prob)? as usize;
        let next = tree[i as usize + bit];
        if next <= 0 {
            return Ok((-next) as u8);
        }
        i = next;
    }
}

/// Decode the §11.2 key-frame Y mode against the constant
/// `KF_YMODE_PROB`. Returns the [`IntraYMode`] for the current MB.
fn read_kf_y_mode(dec: &mut BoolDecoder<'_>) -> Result<IntraYMode, BoolDecoderError> {
    let leaf = treed_read(dec, &KF_YMODE_TREE, |i| KF_YMODE_PROB[i])?;
    Ok(match leaf {
        0 => IntraYMode::Dc,
        1 => IntraYMode::V,
        2 => IntraYMode::H,
        3 => IntraYMode::Tm,
        4 => IntraYMode::B,
        _ => unreachable!("kf_ymode_tree leaf out of range"),
    })
}

/// Decode the §11.4 key-frame UV mode against the constant
/// `KF_UV_MODE_PROB`. Returns the [`IntraUvMode`].
fn read_kf_uv_mode(dec: &mut BoolDecoder<'_>) -> Result<IntraUvMode, BoolDecoderError> {
    let leaf = treed_read(dec, &UV_MODE_TREE, |i| KF_UV_MODE_PROB[i])?;
    Ok(match leaf {
        0 => IntraUvMode::Dc,
        1 => IntraUvMode::V,
        2 => IntraUvMode::H,
        3 => IntraUvMode::Tm,
        _ => unreachable!("uv_mode_tree leaf out of range"),
    })
}

/// Decode one §11.3 sub-block mode given the `above` / `left`
/// neighbouring sub-block modes. The 3-D `KF_BMODE_PROB` table is
/// indexed as `[above][left][i]`, where `i` is the node index of
/// the `bmode_tree` walk.
fn read_kf_b_mode(
    dec: &mut BoolDecoder<'_>,
    above: IntraBmode,
    left: IntraBmode,
) -> Result<IntraBmode, BoolDecoderError> {
    let row = &KF_BMODE_PROB[above.idx()][left.idx()];
    let leaf = treed_read(dec, &BMODE_TREE, |i| row[i])?;
    Ok(match leaf {
        0 => IntraBmode::Dc,
        1 => IntraBmode::Tm,
        2 => IntraBmode::Ve,
        3 => IntraBmode::He,
        4 => IntraBmode::Ld,
        5 => IntraBmode::Rd,
        6 => IntraBmode::Vr,
        7 => IntraBmode::Vl,
        8 => IntraBmode::Hd,
        9 => IntraBmode::Hu,
        _ => unreachable!("bmode_tree leaf out of range"),
    })
}

/// Decode the §10 segment id (a 2-bit code walked through a
/// 4-leaf balanced tree). `probs` is `mb_segment_tree_probs[3]`
/// off the frame's `update_segmentation` block.
fn read_segment_id(dec: &mut BoolDecoder<'_>, probs: &[u8; 3]) -> Result<u8, BoolDecoderError> {
    treed_read(dec, &MB_SEGMENT_TREE, |i| probs[i])
}

/// Decode the key-frame §11 macroblock layer for a complete frame
/// off `dec`. Returns one [`MacroblockModes`] entry per macroblock
/// in raster order (row-major; `(mb_rows * mb_cols)` entries
/// total).
///
/// `header` must be the [`Vp8CodedHeader`] returned by
/// [`Vp8CodedHeader::parse_with_decoder`] for the same partition
/// `dec` is reading — otherwise the bit positions would be
/// inconsistent.
pub fn parse_key_frame_macroblock_modes(
    dec: &mut BoolDecoder<'_>,
    header: &Vp8CodedHeader,
    mb_rows: usize,
    mb_cols: usize,
) -> Result<Vec<MacroblockModes>, MacroblockError> {
    // The Y mode for a key frame is in `color_space.is_some()` —
    // the header's key/inter framing is consistent across all the
    // Option-typed fields in §19.2. We rely on `color_space` as
    // the canonical key-vs-inter discriminator because it is the
    // first field §19.2 distinguishes on.
    if header.color_space.is_none() {
        return Err(MacroblockError::NotAKeyFrame);
    }

    let segmentation_active = header.segmentation_enabled
        && header
            .update_segmentation
            .as_ref()
            .is_some_and(|seg| seg.update_mb_segmentation_map);

    // `mb_segment_tree_probs[3]` defaults to 255 (i.e. "always read
    // a zero") when the corresponding `segment_prob_update_flag` was
    // 0 per §9.3 item 5. We materialise the resolved probabilities
    // up front so the per-MB loop doesn't re-resolve them.
    let segment_probs: [u8; 3] = if segmentation_active {
        let seg = header
            .update_segmentation
            .as_ref()
            .expect("segmentation_active implies update_segmentation is Some");
        let mut p = [255u8; 3];
        for (i, slot) in p.iter_mut().enumerate() {
            if let Some(v) = seg.segment_prob[i] {
                *slot = v;
            }
        }
        p
    } else {
        [255; 3]
    };

    let skip_active = header.mb_no_skip_coeff;
    let skip_prob = header.prob_skip_false.unwrap_or(0);

    // §11.3 item 3: "above" predictor row spans the entire frame
    // width in 4×4-sub-block units (4 sub-blocks per macroblock
    // column). "left" predictor column is 4 sub-blocks tall per
    // macroblock row but only one macroblock wide. Both buffers
    // are initialised to B_DC_PRED at the start of the frame.
    let mut above_subblock = vec![IntraBmode::Dc; mb_cols * 4];

    let mut out = Vec::with_capacity(mb_rows * mb_cols);
    for mb_row in 0..mb_rows {
        // §11.3 item 3 second sentence: "before decoding each row
        // of macroblocks, the four left predictors are also set to
        // B_DC_PRED."
        let mut left_subblock = [IntraBmode::Dc; 4];

        for mb_col in 0..mb_cols {
            // 1. segment_id (§10).
            let segment_id = if segmentation_active {
                Some(read_segment_id(dec, &segment_probs)?)
            } else {
                None
            };

            // 2. mb_skip_coeff (§11.1).
            let mb_skip_coeff = if skip_active {
                dec.read_bool(skip_prob)?
            } else {
                false
            };

            // 3. Luma mode (§11.2).
            let y_mode = read_kf_y_mode(dec)?;

            // 4. Sixteen sub-block modes (§11.3 / §11.5) only if
            //    the macroblock luma mode is B_PRED.
            let subblock_modes = if y_mode == IntraYMode::B {
                let mut modes = [IntraBmode::Dc; 16];
                for j in 0..16 {
                    let row = j >> 2;
                    let col = j & 3;

                    // §11.3 item 2: the sub-block immediately above
                    // sub-block (0, col) is bottom-edge sub-block
                    // (3, col) of the macroblock above. We've been
                    // keeping `above_subblock` updated to track
                    // that bottom row across the frame width.
                    let above = if row == 0 {
                        above_subblock[mb_col * 4 + col]
                    } else {
                        modes[(row - 1) * 4 + col]
                    };
                    // §11.3 item 2 (and 3): the sub-block to the
                    // left of sub-block (row, 0) is right-edge
                    // sub-block (row, 3) of the macroblock to the
                    // left; we've been keeping `left_subblock`
                    // updated to track that right column.
                    let left = if col == 0 {
                        left_subblock[row]
                    } else {
                        modes[row * 4 + (col - 1)]
                    };

                    modes[j] = read_kf_b_mode(dec, above, left)?;
                }
                Some(modes)
            } else {
                None
            };

            // 5. Chroma mode (§11.4).
            let uv_mode = read_kf_uv_mode(dec)?;

            // §11.3 item 3 / 4: update the cross-macroblock
            // predictor buffers in time for the next macroblock.
            // - The bottom four sub-block modes of the macroblock
            //   become `above_subblock` for the macroblock directly
            //   below us.
            // - The right four sub-block modes of the macroblock
            //   become `left_subblock` for the macroblock directly
            //   to our right.
            // For non-B_PRED macroblocks, every sub-block in the
            // macroblock projects to the same `IntraBmode` per
            // §11.3 item 4.
            match &subblock_modes {
                Some(modes) => {
                    above_subblock[mb_col * 4..mb_col * 4 + 4].copy_from_slice(&modes[12..16]);
                    for (row, slot) in left_subblock.iter_mut().enumerate() {
                        *slot = modes[row * 4 + 3];
                    }
                }
                None => {
                    let projected = y_mode.project_to_subblock_context();
                    for slot in &mut above_subblock[mb_col * 4..mb_col * 4 + 4] {
                        *slot = projected;
                    }
                    left_subblock.fill(projected);
                }
            }

            out.push(MacroblockModes {
                segment_id,
                mb_skip_coeff,
                y_mode,
                subblock_modes,
                uv_mode,
            });

            // Suppress unused-variable warning for the loop index
            // when no macroblock layer is decoded (e.g. the empty
            // 0-row degenerate case in tests). Not a real risk in
            // production; keeps clippy happy.
            let _ = mb_row;
        }
    }

    Ok(out)
}

/// `ymode_tree` from RFC 6386 §16.1 — the interframe variant of the
/// 16×16 luma mode tree. Same eight signed entries as
/// [`KF_YMODE_TREE`] but the root left-leaf is `DC_PRED` (the most
/// common mode on interframes) rather than `B_PRED`, so the tree
/// shape and the per-mode bitcost are different.
pub(crate) const IF_YMODE_TREE: [i8; 8] = [
    // -DC_PRED, 2 — root: DC_PRED = "0", "1" subtree at [2..=3]
    0, 2, // 4, 6 — "1" subtree has two descendant subtrees
    4, 6, // -V_PRED, -H_PRED — "10" subtree
    -1, -2, // -TM_PRED, -B_PRED — "11" subtree
    -3, -4,
];

/// `ymode_prob` from RFC 6386 §16.1 — the four default interframe
/// Y-mode probabilities. May be overridden every frame by the
/// `intra_y_mode_prob_update` field of [`Vp8CodedHeader`] and must
/// be **restored to these defaults whenever a key frame is detected**
/// (§16.1 final paragraph).
pub const IF_YMODE_PROB_DEFAULTS: [u8; 4] = [112, 86, 140, 37];

/// `bmode_prob` from RFC 6386 §16.1 — the fixed nine-entry sub-block
/// mode probability table used on interframes. On interframes the
/// sub-block mode walks the same [`BMODE_TREE`] as on key frames but
/// against this single context-free probability vector (whereas key
/// frames use the §11.5 `[above][left][9]` table).
pub const IF_BMODE_PROB: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];

/// `uv_mode_prob` from RFC 6386 §16.1 — the three default interframe
/// UV-mode probabilities. May be overridden every frame by the
/// `intra_uv_mode_prob_update` field of [`Vp8CodedHeader`] and must
/// be **restored to these defaults whenever a key frame is detected**
/// (§16.1 final paragraph).
pub const IF_UV_MODE_PROB_DEFAULTS: [u8; 3] = [162, 101, 204];

/// Resolved per-frame intra-mode probabilities for an interframe.
///
/// The dynamic Y- and UV-mode probability tables of §16.1 carry over
/// from one interframe to the next and are reset to
/// [`IF_YMODE_PROB_DEFAULTS`] / [`IF_UV_MODE_PROB_DEFAULTS`] on every
/// key frame. The frame's header may then overlay any of the four /
/// three replacement probabilities at once via the F-gated
/// `intra_y_mode_prob_update` / `intra_uv_mode_prob_update` fields of
/// the §9.10 header tail (already parsed into [`Vp8CodedHeader`]).
///
/// This struct is what the macroblock-layer decoder reads from. The
/// caller owns the across-frame state, computes the resolved struct
/// per frame using [`InterFrameIntraProbs::for_frame_header`], and
/// passes it in. The output of this round is per-MB; the across-frame
/// continuity is the caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterFrameIntraProbs {
    /// Resolved `ymode_prob[4]` — the four probabilities walked
    /// against [`IF_YMODE_TREE`] (root + the two depth-1 subtrees).
    pub y_mode_prob: [u8; 4],
    /// Resolved `uv_mode_prob[3]` — the three probabilities walked
    /// against [`UV_MODE_TREE`].
    pub uv_mode_prob: [u8; 3],
}

impl InterFrameIntraProbs {
    /// Default initial state for the very first frame of a stream
    /// (which is always a key frame, so the dynamic tables are at
    /// their §16.1 defaults).
    pub const fn defaults() -> Self {
        InterFrameIntraProbs {
            y_mode_prob: IF_YMODE_PROB_DEFAULTS,
            uv_mode_prob: IF_UV_MODE_PROB_DEFAULTS,
        }
    }

    /// Resolve the per-frame intra-mode probability state given the
    /// `previous` state at the end of the previous frame and `header`
    /// for the current frame.
    ///
    /// * If `header` is a key frame (`color_space.is_some()`), the
    ///   dynamic tables are reset to [`IF_YMODE_PROB_DEFAULTS`] /
    ///   [`IF_UV_MODE_PROB_DEFAULTS`] per §16.1 last paragraph,
    ///   ignoring `previous`. (Key-frame mode-decode uses the
    ///   constant `KF_*` probability tables instead — see
    ///   [`parse_key_frame_macroblock_modes`] — but the next
    ///   interframe must start from the defaults regardless.)
    /// * Otherwise the result is `previous` with each F-gated entry
    ///   from `header.intra_y_mode_prob_update` /
    ///   `header.intra_uv_mode_prob_update` overlaid in place. Per
    ///   §9.10, when the F gate is `false` the corresponding `Option`
    ///   is the whole-array `None`, in which case `previous` carries
    ///   forward unchanged.
    pub fn for_frame_header(previous: Self, header: &Vp8CodedHeader) -> Self {
        if header.color_space.is_some() {
            // Key frame: §16.1 last paragraph reset.
            return Self::defaults();
        }
        let y_mode_prob = match header.intra_y_mode_prob_update {
            Some(overrides) => overrides,
            None => previous.y_mode_prob,
        };
        let uv_mode_prob = match header.intra_uv_mode_prob_update {
            Some(overrides) => overrides,
            None => previous.uv_mode_prob,
        };
        InterFrameIntraProbs {
            y_mode_prob,
            uv_mode_prob,
        }
    }
}

impl Default for InterFrameIntraProbs {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Decode the §16.1 interframe Y mode against `probs` (the resolved
/// `IF_YMODE_PROB` for the current frame). Returns the
/// [`IntraYMode`] for the current MB.
fn read_if_y_mode(
    dec: &mut BoolDecoder<'_>,
    probs: &[u8; 4],
) -> Result<IntraYMode, BoolDecoderError> {
    let leaf = treed_read(dec, &IF_YMODE_TREE, |i| probs[i])?;
    Ok(match leaf {
        0 => IntraYMode::Dc,
        1 => IntraYMode::V,
        2 => IntraYMode::H,
        3 => IntraYMode::Tm,
        4 => IntraYMode::B,
        _ => unreachable!("if_ymode_tree leaf out of range"),
    })
}

/// Decode the §16.1 interframe UV mode against `probs` (the resolved
/// `IF_UV_MODE_PROB` for the current frame). The tree shape is the
/// same [`UV_MODE_TREE`] used on key frames; only the probability
/// vector differs.
fn read_if_uv_mode(
    dec: &mut BoolDecoder<'_>,
    probs: &[u8; 3],
) -> Result<IntraUvMode, BoolDecoderError> {
    let leaf = treed_read(dec, &UV_MODE_TREE, |i| probs[i])?;
    Ok(match leaf {
        0 => IntraUvMode::Dc,
        1 => IntraUvMode::V,
        2 => IntraUvMode::H,
        3 => IntraUvMode::Tm,
        _ => unreachable!("uv_mode_tree leaf out of range"),
    })
}

/// Decode one §16.1 interframe sub-block mode against the fixed
/// [`IF_BMODE_PROB`] vector. Unlike the key-frame variant
/// ([`read_kf_b_mode`]), there is **no above / left context**: the
/// same nine probabilities are used for every sub-block.
fn read_if_b_mode(dec: &mut BoolDecoder<'_>) -> Result<IntraBmode, BoolDecoderError> {
    let leaf = treed_read(dec, &BMODE_TREE, |i| IF_BMODE_PROB[i])?;
    Ok(match leaf {
        0 => IntraBmode::Dc,
        1 => IntraBmode::Tm,
        2 => IntraBmode::Ve,
        3 => IntraBmode::He,
        4 => IntraBmode::Ld,
        5 => IntraBmode::Rd,
        6 => IntraBmode::Vr,
        7 => IntraBmode::Vl,
        8 => IntraBmode::Hd,
        9 => IntraBmode::Hu,
        _ => unreachable!("bmode_tree leaf out of range"),
    })
}

/// Decode the §16.1 layer for one intra-predicted macroblock on an
/// interframe.
///
/// The caller has already established that the macroblock is
/// intra-predicted (i.e. the `intra-vs-inter` bit read against
/// `header.prob_intra` was `false` per §16). This function reads the
/// remaining §16.1 fields:
///
/// 1. The 16×16 Y mode (`ymode_tree` against `probs.y_mode_prob`).
/// 2. If the Y mode is [`IntraYMode::B`], the sixteen sub-block modes
///    (`bmode_tree` against the fixed [`IF_BMODE_PROB`], no context).
/// 3. The 8×8 chroma mode (`uv_mode_tree` against
///    `probs.uv_mode_prob`).
///
/// The optional `segment_id` (§10) and `mb_skip_coeff` (§11.1) bits
/// are **not** read here — on interframes they precede the
/// intra-vs-inter discriminator and are therefore consumed before the
/// caller reaches this entry point. To keep the returned
/// [`MacroblockModes`] uniform with the key-frame layout, pass them
/// in via `segment_id` and `mb_skip_coeff`; this function simply
/// forwards them to the output record.
pub fn parse_inter_frame_intra_macroblock_modes(
    dec: &mut BoolDecoder<'_>,
    probs: &InterFrameIntraProbs,
    segment_id: Option<u8>,
    mb_skip_coeff: bool,
) -> Result<MacroblockModes, MacroblockError> {
    let y_mode = read_if_y_mode(dec, &probs.y_mode_prob)?;

    let subblock_modes = if y_mode == IntraYMode::B {
        let mut modes = [IntraBmode::Dc; 16];
        for slot in modes.iter_mut() {
            *slot = read_if_b_mode(dec)?;
        }
        Some(modes)
    } else {
        None
    };

    let uv_mode = read_if_uv_mode(dec, &probs.uv_mode_prob)?;

    Ok(MacroblockModes {
        segment_id,
        mb_skip_coeff,
        y_mode,
        subblock_modes,
        uv_mode,
    })
}

/// `KF_BMODE_PROB[above][left]` — the fixed `[10][10][9]` table
/// from RFC 6386 §11.5. Outer index is the `IntraBmode` of the
/// sub-block above; middle index is the `IntraBmode` of the
/// sub-block to the left; inner array is the nine internal-node
/// probabilities for the `bmode_tree` walk (the leaves naturally
/// produce one of `num_intra_bmodes = 10` values).
pub(crate) const KF_BMODE_PROB: [[[u8; 9]; 10]; 10] = [
    // above = B_DC_PRED
    [
        [231, 120, 48, 89, 115, 113, 120, 152, 112],
        [152, 179, 64, 126, 170, 118, 46, 70, 95],
        [175, 69, 143, 80, 85, 82, 72, 155, 103],
        [56, 58, 10, 171, 218, 189, 17, 13, 152],
        [144, 71, 10, 38, 171, 213, 144, 34, 26],
        [114, 26, 17, 163, 44, 195, 21, 10, 173],
        [121, 24, 80, 195, 26, 62, 44, 64, 85],
        [170, 46, 55, 19, 136, 160, 33, 206, 71],
        [63, 20, 8, 114, 114, 208, 12, 9, 226],
        [81, 40, 11, 96, 182, 84, 29, 16, 36],
    ],
    // above = B_TM_PRED
    [
        [134, 183, 89, 137, 98, 101, 106, 165, 148],
        [72, 187, 100, 130, 157, 111, 32, 75, 80],
        [66, 102, 167, 99, 74, 62, 40, 234, 128],
        [41, 53, 9, 178, 241, 141, 26, 8, 107],
        [104, 79, 12, 27, 217, 255, 87, 17, 7],
        [74, 43, 26, 146, 73, 166, 49, 23, 157],
        [65, 38, 105, 160, 51, 52, 31, 115, 128],
        [87, 68, 71, 44, 114, 51, 15, 186, 23],
        [47, 41, 14, 110, 182, 183, 21, 17, 194],
        [66, 45, 25, 102, 197, 189, 23, 18, 22],
    ],
    // above = B_VE_PRED
    [
        [88, 88, 147, 150, 42, 46, 45, 196, 205],
        [43, 97, 183, 117, 85, 38, 35, 179, 61],
        [39, 53, 200, 87, 26, 21, 43, 232, 171],
        [56, 34, 51, 104, 114, 102, 29, 93, 77],
        [107, 54, 32, 26, 51, 1, 81, 43, 31],
        [39, 28, 85, 171, 58, 165, 90, 98, 64],
        [34, 22, 116, 206, 23, 34, 43, 166, 73],
        [68, 25, 106, 22, 64, 171, 36, 225, 114],
        [34, 19, 21, 102, 132, 188, 16, 76, 124],
        [62, 18, 78, 95, 85, 57, 50, 48, 51],
    ],
    // above = B_HE_PRED
    [
        [193, 101, 35, 159, 215, 111, 89, 46, 111],
        [60, 148, 31, 172, 219, 228, 21, 18, 111],
        [112, 113, 77, 85, 179, 255, 38, 120, 114],
        [40, 42, 1, 196, 245, 209, 10, 25, 109],
        [100, 80, 8, 43, 154, 1, 51, 26, 71],
        [88, 43, 29, 140, 166, 213, 37, 43, 154],
        [61, 63, 30, 155, 67, 45, 68, 1, 209],
        [142, 78, 78, 16, 255, 128, 34, 197, 171],
        [41, 40, 5, 102, 211, 183, 4, 1, 221],
        [51, 50, 17, 168, 209, 192, 23, 25, 82],
    ],
    // above = B_LD_PRED
    [
        [125, 98, 42, 88, 104, 85, 117, 175, 82],
        [95, 84, 53, 89, 128, 100, 113, 101, 45],
        [75, 79, 123, 47, 51, 128, 81, 171, 1],
        [57, 17, 5, 71, 102, 57, 53, 41, 49],
        [115, 21, 2, 10, 102, 255, 166, 23, 6],
        [38, 33, 13, 121, 57, 73, 26, 1, 85],
        [41, 10, 67, 138, 77, 110, 90, 47, 114],
        [101, 29, 16, 10, 85, 128, 101, 196, 26],
        [57, 18, 10, 102, 102, 213, 34, 20, 43],
        [117, 20, 15, 36, 163, 128, 68, 1, 26],
    ],
    // above = B_RD_PRED
    [
        [138, 31, 36, 171, 27, 166, 38, 44, 229],
        [67, 87, 58, 169, 82, 115, 26, 59, 179],
        [63, 59, 90, 180, 59, 166, 93, 73, 154],
        [40, 40, 21, 116, 143, 209, 34, 39, 175],
        [57, 46, 22, 24, 128, 1, 54, 17, 37],
        [47, 15, 16, 183, 34, 223, 49, 45, 183],
        [46, 17, 33, 183, 6, 98, 15, 32, 183],
        [65, 32, 73, 115, 28, 128, 23, 128, 205],
        [40, 3, 9, 115, 51, 192, 18, 6, 223],
        [87, 37, 9, 115, 59, 77, 64, 21, 47],
    ],
    // above = B_VR_PRED
    [
        [104, 55, 44, 218, 9, 54, 53, 130, 226],
        [64, 90, 70, 205, 40, 41, 23, 26, 57],
        [54, 57, 112, 184, 5, 41, 38, 166, 213],
        [30, 34, 26, 133, 152, 116, 10, 32, 134],
        [75, 32, 12, 51, 192, 255, 160, 43, 51],
        [39, 19, 53, 221, 26, 114, 32, 73, 255],
        [31, 9, 65, 234, 2, 15, 1, 118, 73],
        [88, 31, 35, 67, 102, 85, 55, 186, 85],
        [56, 21, 23, 111, 59, 205, 45, 37, 192],
        [55, 38, 70, 124, 73, 102, 1, 34, 98],
    ],
    // above = B_VL_PRED
    [
        [102, 61, 71, 37, 34, 53, 31, 243, 192],
        [69, 60, 71, 38, 73, 119, 28, 222, 37],
        [68, 45, 128, 34, 1, 47, 11, 245, 171],
        [62, 17, 19, 70, 146, 85, 55, 62, 70],
        [75, 15, 9, 9, 64, 255, 184, 119, 16],
        [37, 43, 37, 154, 100, 163, 85, 160, 1],
        [63, 9, 92, 136, 28, 64, 32, 201, 85],
        [86, 6, 28, 5, 64, 255, 25, 248, 1],
        [56, 8, 17, 132, 137, 255, 55, 116, 128],
        [58, 15, 20, 82, 135, 57, 26, 121, 40],
    ],
    // above = B_HD_PRED
    [
        [164, 50, 31, 137, 154, 133, 25, 35, 218],
        [51, 103, 44, 131, 131, 123, 31, 6, 158],
        [86, 40, 64, 135, 148, 224, 45, 183, 128],
        [22, 26, 17, 131, 240, 154, 14, 1, 209],
        [83, 12, 13, 54, 192, 255, 68, 47, 28],
        [45, 16, 21, 91, 64, 222, 7, 1, 197],
        [56, 21, 39, 155, 60, 138, 23, 102, 213],
        [85, 26, 85, 85, 128, 128, 32, 146, 171],
        [18, 11, 7, 63, 144, 171, 4, 4, 246],
        [35, 27, 10, 146, 174, 171, 12, 26, 128],
    ],
    // above = B_HU_PRED
    [
        [190, 80, 35, 99, 180, 80, 126, 54, 45],
        [85, 126, 47, 87, 176, 51, 41, 20, 32],
        [101, 75, 128, 139, 118, 146, 116, 128, 85],
        [56, 41, 15, 176, 236, 85, 37, 9, 62],
        [146, 36, 19, 30, 171, 255, 97, 27, 20],
        [71, 30, 17, 119, 118, 255, 17, 18, 138],
        [101, 38, 60, 138, 55, 70, 43, 26, 142],
        [138, 45, 61, 62, 219, 1, 81, 188, 64],
        [32, 41, 20, 117, 151, 142, 20, 21, 163],
        [112, 19, 12, 61, 195, 128, 48, 4, 24],
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    // In-test boolean encoder mirroring RFC 6386 §7.2 — same shape as
    // the helpers already used in `bool_decoder::tests` and
    // `coded_header::tests`. Kept module-local; the production crate
    // exports nothing that resembles an encoder.
    struct TestEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl TestEncoder {
        fn new() -> Self {
            TestEncoder {
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

        /// Write a tree-coded leaf by walking `tree` to find the path
        /// of bits that lands on `-leaf`. Each path bit is encoded
        /// against `prob_lookup(node_index >> 1)`, mirroring the
        /// decoder's `treed_read` walk.
        fn write_treed<F>(&mut self, tree: &[i8], prob_lookup: F, leaf: u8)
        where
            F: Fn(usize) -> u8,
        {
            // Recursive search for the path from node 0 to `-leaf`.
            fn find_path(tree: &[i8], i: i8, target: i8, path: &mut Vec<bool>) -> bool {
                for bit in 0..2 {
                    let next = tree[i as usize + bit];
                    path.push(bit == 1);
                    if next == target {
                        return true;
                    }
                    if next > 0 && find_path(tree, next, target, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            let mut path = Vec::new();
            let found = find_path(tree, 0, -(leaf as i8), &mut path);
            assert!(
                found,
                "test bug: leaf {} not reachable in tree {:?}",
                leaf, tree
            );
            // Walk the path against the actual tree to determine
            // the node index at every step (so we can call
            // prob_lookup with the same node-halved index the
            // decoder would).
            let mut i: i8 = 0;
            for bit in path {
                self.write_bool(prob_lookup((i as usize) >> 1), bit);
                i = tree[i as usize + bit as usize];
            }
        }

        fn finish(mut self) -> Vec<u8> {
            let c = self.bit_count;
            let mut v = self.bottom;
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
            self.out
        }
    }

    #[test]
    fn kf_ymode_tree_round_trips_all_five_modes() {
        for (leaf, mode) in [
            (0, IntraYMode::Dc),
            (1, IntraYMode::V),
            (2, IntraYMode::H),
            (3, IntraYMode::Tm),
            (4, IntraYMode::B),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&KF_YMODE_TREE, |i| KF_YMODE_PROB[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_kf_y_mode(&mut dec).unwrap();
            assert_eq!(decoded, mode, "leaf {} round-trip", leaf);
        }
    }

    #[test]
    fn uv_mode_tree_round_trips_all_four_modes() {
        for (leaf, mode) in [
            (0, IntraUvMode::Dc),
            (1, IntraUvMode::V),
            (2, IntraUvMode::H),
            (3, IntraUvMode::Tm),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&UV_MODE_TREE, |i| KF_UV_MODE_PROB[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_kf_uv_mode(&mut dec).unwrap();
            assert_eq!(decoded, mode, "uv leaf {} round-trip", leaf);
        }
    }

    #[test]
    fn bmode_tree_round_trips_all_ten_modes() {
        // Sub-block contexts: pick a representative (above, left)
        // pair so the inner KF_BMODE_PROB row is well-defined; the
        // round-trip is independent of which row we use.
        let above_idx = IntraBmode::Dc.idx();
        let left_idx = IntraBmode::Dc.idx();
        let row = &KF_BMODE_PROB[above_idx][left_idx];

        for (leaf, mode) in [
            (0, IntraBmode::Dc),
            (1, IntraBmode::Tm),
            (2, IntraBmode::Ve),
            (3, IntraBmode::He),
            (4, IntraBmode::Ld),
            (5, IntraBmode::Rd),
            (6, IntraBmode::Vr),
            (7, IntraBmode::Vl),
            (8, IntraBmode::Hd),
            (9, IntraBmode::Hu),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&BMODE_TREE, |i| row[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_kf_b_mode(&mut dec, IntraBmode::Dc, IntraBmode::Dc).unwrap();
            assert_eq!(decoded, mode, "bmode leaf {} round-trip", leaf);
        }
    }

    #[test]
    fn mb_segment_tree_round_trips_all_four_segments() {
        let probs = [200u8, 150, 100];
        for sid in 0..4 {
            let mut enc = TestEncoder::new();
            enc.write_treed(&MB_SEGMENT_TREE, |i| probs[i], sid);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_segment_id(&mut dec, &probs).unwrap();
            assert_eq!(decoded, sid, "segment id {} round-trip", sid);
        }
    }

    /// Minimal end-to-end check: a 2x1 macroblock frame, segmentation
    /// disabled, mb_no_skip_coeff disabled. The first MB picks
    /// `DC_PRED` plus UV `DC_PRED`; the second picks `B_PRED` plus
    /// UV `V_PRED` with sixteen sub-block modes (an all-Dc selection
    /// so the context-table lookup is well-defined).
    #[test]
    fn macroblock_decode_round_trips_minimal_two_mb_frame() {
        let mut enc = TestEncoder::new();
        // MB 0: DC_PRED, UV DC_PRED.
        enc.write_treed(&KF_YMODE_TREE, |i| KF_YMODE_PROB[i], 0);
        enc.write_treed(&UV_MODE_TREE, |i| KF_UV_MODE_PROB[i], 0);
        // MB 1: B_PRED, sixteen B_DC sub-block modes, UV V_PRED.
        enc.write_treed(&KF_YMODE_TREE, |i| KF_YMODE_PROB[i], 4);
        for j in 0..16 {
            // The above predictor for sub-block (0, col) comes from
            // above_subblock[1 * 4 + col]; for non-edge sub-blocks
            // we use the prior row. Both default to Dc on this
            // tiny frame (mb_row=0, mb above is non-existent).
            let row = j >> 2;
            let _col = j & 3;
            // Above defaults to Dc until we've populated the row.
            // Left defaults to Dc for col == 0 and for the row's
            // prior column also being Dc (since we're encoding all
            // Dcs).
            let _ = row;
            let row_probs = &KF_BMODE_PROB[IntraBmode::Dc.idx()][IntraBmode::Dc.idx()];
            enc.write_treed(&BMODE_TREE, |i| row_probs[i], 0);
        }
        enc.write_treed(&UV_MODE_TREE, |i| KF_UV_MODE_PROB[i], 1);
        let buf = enc.finish();

        // Build a synthetic header that says "key frame, no
        // segmentation, no skip". We can't easily generate one
        // off of bytes without round-tripping the whole §19.2 block,
        // so construct one by hand for the macroblock decode entry
        // point.
        let header = crate::coded_header::Vp8CodedHeader {
            color_space: Some(false),
            clamping_type: Some(false),
            segmentation_enabled: false,
            update_segmentation: None,
            filter_type: false,
            loop_filter_level: 0,
            sharpness_level: 0,
            mb_lf_adjustments: crate::coded_header::MbLfAdjustments {
                loop_filter_adj_enable: false,
                mode_ref_lf_delta_update: false,
                ref_frame_delta_update: [None; 4],
                mb_mode_delta_update: [None; 4],
            },
            log2_nbr_of_dct_partitions: 0,
            nbr_of_dct_partitions: 1,
            quant_indices: crate::coded_header::QuantIndices {
                y_ac_qi: 0,
                y_dc_delta: None,
                y2_dc_delta: None,
                y2_ac_delta: None,
                uv_dc_delta: None,
                uv_ac_delta: None,
            },
            refresh_entropy_probs: false,
            refresh_golden_frame: None,
            refresh_alternate_frame: None,
            copy_buffer_to_golden: None,
            copy_buffer_to_alternate: None,
            sign_bias_golden: None,
            sign_bias_alternate: None,
            refresh_last: None,
            token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
            mb_no_skip_coeff: false,
            prob_skip_false: None,
            prob_intra: None,
            prob_last: None,
            prob_gf: None,
            intra_y_mode_prob_update: None,
            intra_uv_mode_prob_update: None,
            mv_prob_update: None,
        };

        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mbs = parse_key_frame_macroblock_modes(&mut dec, &header, 1, 2).unwrap();

        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].y_mode, IntraYMode::Dc);
        assert_eq!(mbs[0].uv_mode, IntraUvMode::Dc);
        assert!(mbs[0].subblock_modes.is_none());
        assert_eq!(mbs[0].segment_id, None);
        assert!(!mbs[0].mb_skip_coeff);
        assert_eq!(mbs[1].y_mode, IntraYMode::B);
        assert_eq!(mbs[1].uv_mode, IntraUvMode::V);
        let sub = mbs[1].subblock_modes.expect("B_PRED carries 16 sub-modes");
        assert!(sub.iter().all(|m| *m == IntraBmode::Dc));
    }

    /// Segmentation enabled + mb_skip_coeff enabled — exercises the
    /// optional prefix bits of §10 and §11.1. One macroblock.
    #[test]
    fn macroblock_decode_handles_optional_segment_and_skip() {
        let mut enc = TestEncoder::new();
        // segment_id = 2 (path "10" against tree probs [200, 150, 100])
        let seg_probs = [200u8, 150, 100];
        enc.write_treed(&MB_SEGMENT_TREE, |i| seg_probs[i], 2);
        // mb_skip_coeff = true (read against prob_skip_false = 50).
        enc.write_bool(50, true);
        // Y mode = TM_PRED.
        enc.write_treed(&KF_YMODE_TREE, |i| KF_YMODE_PROB[i], 3);
        // UV mode = H_PRED.
        enc.write_treed(&UV_MODE_TREE, |i| KF_UV_MODE_PROB[i], 2);
        let buf = enc.finish();

        let header = crate::coded_header::Vp8CodedHeader {
            color_space: Some(false),
            clamping_type: Some(false),
            segmentation_enabled: true,
            update_segmentation: Some(crate::coded_header::UpdateSegmentation {
                update_mb_segmentation_map: true,
                update_segment_feature_data: false,
                segment_feature_mode_absolute: false,
                quantizer_update: [None; 4],
                loop_filter_update: [None; 4],
                segment_prob: [Some(200), Some(150), Some(100)],
            }),
            filter_type: false,
            loop_filter_level: 0,
            sharpness_level: 0,
            mb_lf_adjustments: crate::coded_header::MbLfAdjustments {
                loop_filter_adj_enable: false,
                mode_ref_lf_delta_update: false,
                ref_frame_delta_update: [None; 4],
                mb_mode_delta_update: [None; 4],
            },
            log2_nbr_of_dct_partitions: 0,
            nbr_of_dct_partitions: 1,
            quant_indices: crate::coded_header::QuantIndices {
                y_ac_qi: 0,
                y_dc_delta: None,
                y2_dc_delta: None,
                y2_ac_delta: None,
                uv_dc_delta: None,
                uv_ac_delta: None,
            },
            refresh_entropy_probs: false,
            refresh_golden_frame: None,
            refresh_alternate_frame: None,
            copy_buffer_to_golden: None,
            copy_buffer_to_alternate: None,
            sign_bias_golden: None,
            sign_bias_alternate: None,
            refresh_last: None,
            token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
            mb_no_skip_coeff: true,
            prob_skip_false: Some(50),
            prob_intra: None,
            prob_last: None,
            prob_gf: None,
            intra_y_mode_prob_update: None,
            intra_uv_mode_prob_update: None,
            mv_prob_update: None,
        };

        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mbs = parse_key_frame_macroblock_modes(&mut dec, &header, 1, 1).unwrap();
        assert_eq!(mbs.len(), 1);
        assert_eq!(mbs[0].segment_id, Some(2));
        assert!(mbs[0].mb_skip_coeff);
        assert_eq!(mbs[0].y_mode, IntraYMode::Tm);
        assert_eq!(mbs[0].uv_mode, IntraUvMode::H);
    }

    /// `parse_key_frame_macroblock_modes` must refuse to run on an
    /// interframe header (where `color_space` is `None`).
    #[test]
    fn macroblock_decode_rejects_interframe_header() {
        let header = crate::coded_header::Vp8CodedHeader {
            color_space: None,
            clamping_type: None,
            segmentation_enabled: false,
            update_segmentation: None,
            filter_type: false,
            loop_filter_level: 0,
            sharpness_level: 0,
            mb_lf_adjustments: crate::coded_header::MbLfAdjustments {
                loop_filter_adj_enable: false,
                mode_ref_lf_delta_update: false,
                ref_frame_delta_update: [None; 4],
                mb_mode_delta_update: [None; 4],
            },
            log2_nbr_of_dct_partitions: 0,
            nbr_of_dct_partitions: 1,
            quant_indices: crate::coded_header::QuantIndices {
                y_ac_qi: 0,
                y_dc_delta: None,
                y2_dc_delta: None,
                y2_ac_delta: None,
                uv_dc_delta: None,
                uv_ac_delta: None,
            },
            refresh_entropy_probs: false,
            refresh_golden_frame: Some(false),
            refresh_alternate_frame: Some(false),
            copy_buffer_to_golden: Some(0),
            copy_buffer_to_alternate: Some(0),
            sign_bias_golden: Some(false),
            sign_bias_alternate: Some(false),
            refresh_last: Some(false),
            token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
            mb_no_skip_coeff: false,
            prob_skip_false: None,
            prob_intra: Some(128),
            prob_last: Some(128),
            prob_gf: Some(128),
            intra_y_mode_prob_update: None,
            intra_uv_mode_prob_update: None,
            mv_prob_update: None,
        };
        // Two valid bytes for BoolDecoder::init; the macroblock
        // parser bails before consuming anything.
        let buf = [0u8, 0u8];
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let err = parse_key_frame_macroblock_modes(&mut dec, &header, 1, 1).unwrap_err();
        assert_eq!(err, MacroblockError::NotAKeyFrame);
    }

    /// Spec-listing transcription check on the KF Y / UV mode prob
    /// constants. Catches arithmetic typos at compile-time-of-test.
    #[test]
    fn kf_mode_probs_match_spec_listing() {
        assert_eq!(KF_YMODE_PROB, [145, 156, 163, 128]);
        assert_eq!(KF_UV_MODE_PROB, [142, 114, 183]);
    }

    /// `KF_BMODE_PROB` is 10×10×9 — spot-check a handful of values
    /// drawn from §11.5 to confirm transcription.
    #[test]
    fn kf_bmode_prob_spot_checks() {
        // First row of the [B_DC][B_DC] sub-table:
        assert_eq!(
            KF_BMODE_PROB[0][0],
            [231, 120, 48, 89, 115, 113, 120, 152, 112]
        );
        // Last entry: [B_HU][B_HU] = {112, 19, 12, 61, 195, 128, 48, 4, 24}
        assert_eq!(KF_BMODE_PROB[9][9], [112, 19, 12, 61, 195, 128, 48, 4, 24]);
        // Middle sample: [B_LD][B_RD] = {38, 33, 13, 121, 57, 73, 26, 1, 85}
        assert_eq!(KF_BMODE_PROB[4][5], [38, 33, 13, 121, 57, 73, 26, 1, 85]);
    }

    // ----- §16.1 interframe intra-MB mode tests -------------------------

    /// Spec-listing transcription check on the §16.1 default
    /// probability tables. Catches arithmetic typos at compile-time
    /// of test.
    #[test]
    fn if_mode_probs_match_spec_listing() {
        assert_eq!(IF_YMODE_PROB_DEFAULTS, [112, 86, 140, 37]);
        assert_eq!(IF_UV_MODE_PROB_DEFAULTS, [162, 101, 204]);
        assert_eq!(IF_BMODE_PROB, [120, 90, 79, 133, 87, 85, 80, 111, 151]);
    }

    /// `IF_YMODE_TREE` differs from `KF_YMODE_TREE`: on interframes
    /// the root left-leaf is `DC_PRED` ("0"), not `B_PRED`. The
    /// shape of the tree is the same eight signed entries but the
    /// leaves at the root differ.
    #[test]
    fn if_ymode_tree_shape_matches_spec() {
        // -DC_PRED, 2, 4, 6, -V_PRED, -H_PRED, -TM_PRED, -B_PRED.
        // DC=0, V=1, H=2, TM=3, B=4, so leaves are 0, -1, -2, -3, -4.
        assert_eq!(IF_YMODE_TREE, [0, 2, 4, 6, -1, -2, -3, -4]);
        // Tree[0] is the negative DC leaf (`-DC_PRED`); the KF tree
        // has the negative B_PRED leaf in the same slot. The two
        // trees disagree at exactly that position.
        assert_ne!(IF_YMODE_TREE[0], KF_YMODE_TREE[0]);
        assert_eq!(IF_YMODE_TREE[0], 0); // -DC_PRED
        assert_eq!(KF_YMODE_TREE[0], -4); // -B_PRED
    }

    #[test]
    fn if_ymode_tree_round_trips_all_five_modes() {
        for (leaf, mode) in [
            (0, IntraYMode::Dc),
            (1, IntraYMode::V),
            (2, IntraYMode::H),
            (3, IntraYMode::Tm),
            (4, IntraYMode::B),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&IF_YMODE_TREE, |i| IF_YMODE_PROB_DEFAULTS[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_if_y_mode(&mut dec, &IF_YMODE_PROB_DEFAULTS).unwrap();
            assert_eq!(decoded, mode, "if y leaf {} round-trip", leaf);
        }
    }

    #[test]
    fn if_uv_mode_tree_round_trips_all_four_modes() {
        for (leaf, mode) in [
            (0, IntraUvMode::Dc),
            (1, IntraUvMode::V),
            (2, IntraUvMode::H),
            (3, IntraUvMode::Tm),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&UV_MODE_TREE, |i| IF_UV_MODE_PROB_DEFAULTS[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_if_uv_mode(&mut dec, &IF_UV_MODE_PROB_DEFAULTS).unwrap();
            assert_eq!(decoded, mode, "if uv leaf {} round-trip", leaf);
        }
    }

    #[test]
    fn if_bmode_tree_round_trips_all_ten_modes() {
        for (leaf, mode) in [
            (0, IntraBmode::Dc),
            (1, IntraBmode::Tm),
            (2, IntraBmode::Ve),
            (3, IntraBmode::He),
            (4, IntraBmode::Ld),
            (5, IntraBmode::Rd),
            (6, IntraBmode::Vr),
            (7, IntraBmode::Vl),
            (8, IntraBmode::Hd),
            (9, IntraBmode::Hu),
        ] {
            let mut enc = TestEncoder::new();
            enc.write_treed(&BMODE_TREE, |i| IF_BMODE_PROB[i], leaf);
            let buf = enc.finish();

            let mut dec = BoolDecoder::init(&buf).unwrap();
            let decoded = read_if_b_mode(&mut dec).unwrap();
            assert_eq!(decoded, mode, "if bmode leaf {} round-trip", leaf);
        }
    }

    /// Round-trip a complete §16.1 intra MB with a non-`B_PRED` Y
    /// mode against the default probability tables.
    #[test]
    fn if_intra_mb_decode_non_bpred_round_trip() {
        let mut enc = TestEncoder::new();
        // Y mode = V_PRED (leaf 1).
        enc.write_treed(&IF_YMODE_TREE, |i| IF_YMODE_PROB_DEFAULTS[i], 1);
        // UV mode = H_PRED (leaf 2).
        enc.write_treed(&UV_MODE_TREE, |i| IF_UV_MODE_PROB_DEFAULTS[i], 2);
        let buf = enc.finish();

        let probs = InterFrameIntraProbs::defaults();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mb = parse_inter_frame_intra_macroblock_modes(&mut dec, &probs, None, false).unwrap();
        assert_eq!(mb.y_mode, IntraYMode::V);
        assert_eq!(mb.uv_mode, IntraUvMode::H);
        assert!(mb.subblock_modes.is_none());
        assert_eq!(mb.segment_id, None);
        assert!(!mb.mb_skip_coeff);
    }

    /// Round-trip a `B_PRED` Y mode with a mixed pattern of sub-block
    /// modes — proves the §16.1 fixed-prob `bmode_tree` walks
    /// produce the same leaf the encoder wrote.
    #[test]
    fn if_intra_mb_decode_bpred_with_sixteen_subblock_modes() {
        let leaves: [u8; 16] = [
            0, // Dc
            1, // Tm
            2, // Ve
            3, // He
            4, // Ld
            5, // Rd
            6, // Vr
            7, // Vl
            8, // Hd
            9, // Hu
            0, 1, 2, 3, 4, 5,
        ];
        let expected_modes = [
            IntraBmode::Dc,
            IntraBmode::Tm,
            IntraBmode::Ve,
            IntraBmode::He,
            IntraBmode::Ld,
            IntraBmode::Rd,
            IntraBmode::Vr,
            IntraBmode::Vl,
            IntraBmode::Hd,
            IntraBmode::Hu,
            IntraBmode::Dc,
            IntraBmode::Tm,
            IntraBmode::Ve,
            IntraBmode::He,
            IntraBmode::Ld,
            IntraBmode::Rd,
        ];

        let mut enc = TestEncoder::new();
        // Y mode = B_PRED (leaf 4).
        enc.write_treed(&IF_YMODE_TREE, |i| IF_YMODE_PROB_DEFAULTS[i], 4);
        for leaf in leaves {
            enc.write_treed(&BMODE_TREE, |i| IF_BMODE_PROB[i], leaf);
        }
        // UV mode = TM_PRED (leaf 3).
        enc.write_treed(&UV_MODE_TREE, |i| IF_UV_MODE_PROB_DEFAULTS[i], 3);
        let buf = enc.finish();

        let probs = InterFrameIntraProbs::defaults();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mb = parse_inter_frame_intra_macroblock_modes(&mut dec, &probs, Some(2), true).unwrap();
        assert_eq!(mb.y_mode, IntraYMode::B);
        assert_eq!(mb.uv_mode, IntraUvMode::Tm);
        // The optional segment id / skip flag pass through verbatim
        // from the caller, matching the §16.1 contract that those
        // fields precede the intra-vs-inter discriminator and are
        // therefore read before this entry point.
        assert_eq!(mb.segment_id, Some(2));
        assert!(mb.mb_skip_coeff);
        let sub = mb.subblock_modes.expect("B_PRED carries 16 sub-modes");
        assert_eq!(sub, expected_modes);
    }

    /// `InterFrameIntraProbs::for_frame_header` must reset both
    /// dynamic tables to their §16.1 defaults on a key frame,
    /// regardless of what the previous interframe state was.
    #[test]
    fn intra_probs_reset_on_key_frame() {
        let scratch = InterFrameIntraProbs {
            y_mode_prob: [10, 20, 30, 40],
            uv_mode_prob: [50, 60, 70],
        };
        let key_header = build_minimal_header_with_intra_overrides(true, None, None);
        let resolved = InterFrameIntraProbs::for_frame_header(scratch, &key_header);
        assert_eq!(resolved, InterFrameIntraProbs::defaults());
    }

    /// On an interframe without override blocks, the previous-frame
    /// dynamic state must carry forward unchanged.
    #[test]
    fn intra_probs_carry_forward_on_interframe_without_overrides() {
        let previous = InterFrameIntraProbs {
            y_mode_prob: [10, 20, 30, 40],
            uv_mode_prob: [50, 60, 70],
        };
        let inter_header = build_minimal_header_with_intra_overrides(false, None, None);
        let resolved = InterFrameIntraProbs::for_frame_header(previous, &inter_header);
        assert_eq!(resolved, previous);
    }

    /// On an interframe with override blocks, the override values
    /// replace the previous-frame entries wholesale (the §9.10 F
    /// gate is all-or-nothing on each of the Y / UV blocks).
    #[test]
    fn intra_probs_overlay_on_interframe_with_overrides() {
        let previous = InterFrameIntraProbs {
            y_mode_prob: [10, 20, 30, 40],
            uv_mode_prob: [50, 60, 70],
        };
        let inter_header = build_minimal_header_with_intra_overrides(
            false,
            Some([200, 150, 100, 50]),
            Some([180, 90, 1]),
        );
        let resolved = InterFrameIntraProbs::for_frame_header(previous, &inter_header);
        assert_eq!(resolved.y_mode_prob, [200, 150, 100, 50]);
        assert_eq!(resolved.uv_mode_prob, [180, 90, 1]);
    }

    /// Only one of the two override blocks present — the other one
    /// must carry forward.
    #[test]
    fn intra_probs_mixed_overlay_y_only() {
        let previous = InterFrameIntraProbs {
            y_mode_prob: [10, 20, 30, 40],
            uv_mode_prob: [50, 60, 70],
        };
        let inter_header =
            build_minimal_header_with_intra_overrides(false, Some([200, 150, 100, 50]), None);
        let resolved = InterFrameIntraProbs::for_frame_header(previous, &inter_header);
        assert_eq!(resolved.y_mode_prob, [200, 150, 100, 50]);
        assert_eq!(resolved.uv_mode_prob, [50, 60, 70]);
    }

    /// `Default` impl matches `defaults()`.
    #[test]
    fn intra_probs_default_matches_defaults_constructor() {
        assert_eq!(
            InterFrameIntraProbs::default(),
            InterFrameIntraProbs::defaults()
        );
    }

    /// Build a minimal [`Vp8CodedHeader`] suitable for the
    /// `InterFrameIntraProbs::for_frame_header` tests. `key_frame`
    /// controls `color_space`; `y_overrides` / `uv_overrides` carry
    /// straight into the two §9.10 fields.
    fn build_minimal_header_with_intra_overrides(
        key_frame: bool,
        y_overrides: Option<[u8; 4]>,
        uv_overrides: Option<[u8; 3]>,
    ) -> crate::coded_header::Vp8CodedHeader {
        crate::coded_header::Vp8CodedHeader {
            color_space: if key_frame { Some(false) } else { None },
            clamping_type: if key_frame { Some(false) } else { None },
            segmentation_enabled: false,
            update_segmentation: None,
            filter_type: false,
            loop_filter_level: 0,
            sharpness_level: 0,
            mb_lf_adjustments: crate::coded_header::MbLfAdjustments {
                loop_filter_adj_enable: false,
                mode_ref_lf_delta_update: false,
                ref_frame_delta_update: [None; 4],
                mb_mode_delta_update: [None; 4],
            },
            log2_nbr_of_dct_partitions: 0,
            nbr_of_dct_partitions: 1,
            quant_indices: crate::coded_header::QuantIndices {
                y_ac_qi: 0,
                y_dc_delta: None,
                y2_dc_delta: None,
                y2_ac_delta: None,
                uv_dc_delta: None,
                uv_ac_delta: None,
            },
            refresh_entropy_probs: false,
            refresh_golden_frame: if key_frame { None } else { Some(false) },
            refresh_alternate_frame: if key_frame { None } else { Some(false) },
            copy_buffer_to_golden: if key_frame { None } else { Some(0) },
            copy_buffer_to_alternate: if key_frame { None } else { Some(0) },
            sign_bias_golden: if key_frame { None } else { Some(false) },
            sign_bias_alternate: if key_frame { None } else { Some(false) },
            refresh_last: if key_frame { None } else { Some(false) },
            token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
            mb_no_skip_coeff: false,
            prob_skip_false: None,
            prob_intra: if key_frame { None } else { Some(128) },
            prob_last: if key_frame { None } else { Some(128) },
            prob_gf: if key_frame { None } else { Some(128) },
            intra_y_mode_prob_update: y_overrides,
            intra_uv_mode_prob_update: uv_overrides,
            mv_prob_update: None,
        }
    }
}
