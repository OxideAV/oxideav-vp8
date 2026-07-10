//! VP8 DCT-coefficient token decoding per RFC 6386 §13.
//!
//! This module walks the `coeff_tree` of §13.2 against the
//! `coeff_probs[4][8][3][11]` probability table to recover the sixteen
//! quantised DCT coefficients of a single 4×4 sub-block. Encoded
//! coefficients are written into a caller-supplied `[i16; 16]` in the
//! spec's "implicit position" order — i.e. positions `firstCoeff ..
//! firstCoeff + n` of the (already-zig-zagged) sub-block array. After
//! a `dct_eob` token the remaining positions are left at zero.
//!
//! **What is in this module:**
//!
//! * The §13.2 token-tree walk, with the §13.2 "skip the dct_eob
//!   branch when the previous coefficient was a DCT_0" optimisation.
//!   The tree is fixed (`COEFF_TREE` transcribes the §13.2 listing),
//!   so the production descent in `decode_block_core` is written out
//!   branch-by-branch — each internal node reads `probs[node >> 1]`
//!   exactly as the generic §8.1 `treed_read` walk would, but without
//!   the per-step tree-table loads, and each leaf flows straight into
//!   its §13.2 consequence (zero / small value / `DCTextra` category /
//!   end-of-block). The "skip dct_eob" rule becomes the inner
//!   zero-run loop re-entering at the DCT_0 node. Bit-equivalence
//!   against a generic table-driven walker over the same `COEFF_TREE`
//!   is enforced by `fused_descent_matches_generic_tree_walk`.
//! * The §13.2 `DCTextra` extra-bits decoder over the six fixed
//!   probability tables `Pcat1..Pcat6`.
//! * The §13.2 sign bit (read at probability 128).
//! * The §13.3 `coeff_bands[16]` position-to-band mapping that selects
//!   the second probability-table index.
//! * The §13.3 "previous-token class" `ctx3` rollover that selects the
//!   third probability-table index (0 if previous was zero, 1 if its
//!   absolute value was 1, 2 if its absolute value was > 1).
//! * The §13.3 plane-type discriminator: 0 = Y after Y2, 1 = Y2,
//!   2 = U or V, 3 = Y in the absence of Y2.
//! * The §13.5 default token-probability table (transcribed verbatim).
//! * A `merge_default_token_probs` helper that overlays a
//!   `TokenProbUpdates` (from `coded_header::Vp8CodedHeader`) onto the
//!   default table, producing a complete `coeff_probs[4][8][3][11]`
//!   ready for `decode_block` calls.
//! * The §13.3 per-macroblock token walk (`decode_mb_coeffs`): it
//!   aggregates the 24/25 blocks of one macroblock — the §14.2 Y2 (WHT)
//!   block first when present, then the sixteen Y DCT blocks, then the
//!   four U and four V chroma blocks — threading the above / left
//!   non-zero predictor context through the §20.16
//!   `left_context_index` / `above_context_index` slot tables, honouring
//!   the §13.1 `mb_skip_coeff` short-circuit (with the §20.16
//!   `reset_mb_context` Y2-preserving rule), selecting the
//!   `YAfterY2` / `YNoY2` plane per `has_y2`, and reordering each block
//!   into raster order via the §20.16 `zigzag[16]` table. It produces a
//!   [`crate::frame::MbCoeffs`].
//!
//! **What is NOT in this module yet:**
//!
//! * Dequantisation — multiplying the recovered `i16` coefficients by
//!   the §14.1 `dc_qlookup` / `ac_qlookup` table values. `decode_mb_coeffs`
//!   emits the **raw quantized** token coefficients (in raster order);
//!   the §14.1 Y2 / chroma dequant scaling remains a documented spec gap
//!   (§14.1 page 77 defers it to `dixie.c` §20.4).
//! * The inverse WHT / IDCT (§14.2 / §14.3).
//!
//! # Spec-pseudocode caveat
//!
//! RFC 6386 §13.3 page 67's pseudocode ends each iteration with the
//! literal statement `prevCoeffWasZero = true;` — i.e. *unconditionally
//! true*. That is plainly a transcription error: the field is named
//! `prevCoeffWasZero` and is used at the next iteration's tree entry
//! to decide whether to skip the `dct_eob` branch (which is illegal
//! after a zero). Setting it unconditionally true would mean **every
//! coefficient after the first** allows EOB-after-non-zero, which
//! contradicts §13.2's "if the preceding coefficient is a DCT_0,
//! decoding will skip the first branch" statement. We treat the
//! correct semantics as `prevCoeffWasZero = (token == DCT_0)`. The
//! `prev_token_skips_eob_branch` test exercises both directions to
//! prove this branch decision matters.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::coded_header::TokenProbUpdates;

/// Twelve DCT-token alphabet symbols per RFC 6386 §13.2. Order is
/// fixed to match the spec; the integer discriminants are also the
/// `dct_token` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctToken {
    /// Coefficient value 0.
    Dct0 = 0,
    /// Coefficient value 1.
    Dct1 = 1,
    /// Coefficient value 2.
    Dct2 = 2,
    /// Coefficient value 3.
    Dct3 = 3,
    /// Coefficient value 4.
    Dct4 = 4,
    /// Range `5..=6` (size 2; 1 extra bit).
    Cat1 = 5,
    /// Range `7..=10` (size 4; 2 extra bits).
    Cat2 = 6,
    /// Range `11..=18` (size 8; 3 extra bits).
    Cat3 = 7,
    /// Range `19..=34` (size 16; 4 extra bits).
    Cat4 = 8,
    /// Range `35..=66` (size 32; 5 extra bits).
    Cat5 = 9,
    /// Range `67..=2048` (size 1982; 11 extra bits).
    Cat6 = 10,
    /// End of block — the remaining coefficients of this sub-block
    /// are zero.
    Eob = 11,
}

/// The §13.3 "plane type" — the outermost index into
/// `coeff_probs[4][8][3][11]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// Y luma block when the macroblock also has a Y2 block — i.e.
    /// the DC slot is part of Y2 and Y DCT starts at coefficient 1.
    YAfterY2 = 0,
    /// The Y2 block itself (one per macroblock, holds the 16 Y DCs).
    Y2 = 1,
    /// U or V chroma block.
    UV = 2,
    /// Y luma block in a macroblock without Y2 (B_PRED / SPLITMV) —
    /// Y DCT starts at coefficient 0.
    YNoY2 = 3,
}

impl BlockType {
    /// The §13.3 `firstCoeff` index — the first coefficient position
    /// the token loop visits. `1` for [`BlockType::YAfterY2`], `0` for
    /// every other plane.
    pub const fn first_coeff(self) -> usize {
        match self {
            BlockType::YAfterY2 => 1,
            BlockType::Y2 | BlockType::UV | BlockType::YNoY2 => 0,
        }
    }

    /// The numeric `coeff_probs` outermost index per §13.3.
    pub const fn plane_index(self) -> usize {
        self as usize
    }
}

/// Out-of-band conditions surfaced by [`decode_block`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctTokenError {
    /// Underlying bool-decoder ran out of bytes mid-token.
    BoolDecoder(BoolDecoderError),
    /// Historical variant: a generic tree walk stepped to a leaf index
    /// outside `0..=11`. The production descent is now written out
    /// branch-by-branch over the fixed §13.2 tree, so this can no
    /// longer occur; the variant is retained for API compatibility
    /// (it is part of the public error surface).
    InvalidTokenIndex,
}

impl From<BoolDecoderError> for DctTokenError {
    fn from(e: BoolDecoderError) -> Self {
        DctTokenError::BoolDecoder(e)
    }
}

impl core::fmt::Display for DctTokenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DctTokenError::BoolDecoder(e) => write!(f, "vp8 dct token: bool decoder: {e}"),
            DctTokenError::InvalidTokenIndex => {
                write!(f, "vp8 dct token: token tree leaf index out of range")
            }
        }
    }
}

impl std::error::Error for DctTokenError {}

/// §13.2 `coeff_tree` — the eleven-internal-node, twelve-leaf token
/// tree. Entry `2*i` is the left child of internal node `i`; entry
/// `2*i + 1` is the right child. Non-negative entries point to the
/// next internal node; non-positive entries `-t` denote the leaf
/// `DctToken::from(t as u8)`.
///
/// Encoded verbatim from the §13.2 listing. The leading `-Eob` entry
/// is at index 0; when the previous coefficient was zero the walker
/// starts at index 2 (skipping the EOB branch) — see [`decode_block`].
///
/// The production descent (`decode_block_core`) writes this fixed tree
/// out branch-by-branch instead of walking the table, so the constant
/// is no longer loaded on the hot path. It stays as the §13.2
/// transcription anchor: the test-side token encoder and the generic
/// reference walker (`fused_descent_matches_generic_tree_walk`) both
/// consume it to prove the fused descent agrees with the listing.
#[allow(dead_code)] // production reads are fused; tests consume the table.
const COEFF_TREE: [i8; 22] = [
    -(DctToken::Eob as i8),
    2, // eob = "0"
    -(DctToken::Dct0 as i8),
    4, // 0   = "10"
    -(DctToken::Dct1 as i8),
    6, // 1   = "110"
    8,
    12,
    -(DctToken::Dct2 as i8),
    10, // 2   = "11100"
    -(DctToken::Dct3 as i8),
    -(DctToken::Dct4 as i8),
    14,
    16,
    -(DctToken::Cat1 as i8),
    -(DctToken::Cat2 as i8),
    18,
    20,
    -(DctToken::Cat3 as i8),
    -(DctToken::Cat4 as i8),
    -(DctToken::Cat5 as i8),
    -(DctToken::Cat6 as i8),
];

/// §13.2 `Pcat1` extra-bits probability list (terminator removed).
const PCAT1: &[u8] = &[159];
/// §13.2 `Pcat2`.
const PCAT2: &[u8] = &[165, 145];
/// §13.2 `Pcat3`.
const PCAT3: &[u8] = &[173, 148, 140];
/// §13.2 `Pcat4`.
const PCAT4: &[u8] = &[176, 155, 140, 135];
/// §13.2 `Pcat5`.
const PCAT5: &[u8] = &[180, 157, 141, 134, 130];
/// §13.2 `Pcat6` — eleven probabilities for the wide cat6 range.
const PCAT6: &[u8] = &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

/// §13.2 base values for cat1..cat6 (`categoryBase[6]`).
const CAT_BASE: [u16; 6] = [5, 7, 11, 19, 35, 67];

/// §13.3 `coeff_bands[16]` — position-to-band lookup for the second
/// probability-table dimension.
pub const COEFF_BANDS: [usize; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];

/// Complete `coeff_probs[plane][band][prev_ctx][position]` table.
/// The four outer dimensions match RFC 6386 §13.3's `Prob
/// coeff_probs[4][8][3][num_dct_tokens - 1]` declaration.
pub type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// §13.5 default token-probability table (`default_coeff_probs[4][8][3][11]`),
/// transcribed verbatim from RFC 6386.
pub const DEFAULT_COEFF_PROBS: CoeffProbs = [
    // plane 0 — Y after Y2 (DCT starts at coefficient 1)
    [
        [
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
        [
            [253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128],
            [189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128],
            [106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128],
        ],
        [
            [1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128],
            [181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128],
            [78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128],
        ],
        [
            [1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128],
            [184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128],
            [77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128],
        ],
        [
            [1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128],
            [170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128],
            [37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128],
        ],
        [
            [1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128],
            [207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128],
            [102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128],
        ],
        [
            [1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128],
            [177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128],
            [80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128],
        ],
        [
            [1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
    ],
    // plane 1 — Y2
    [
        [
            [198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62],
            [131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1],
            [68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128],
        ],
        [
            [1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128],
            [184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128],
            [81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128],
        ],
        [
            [1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128],
            [99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128],
            [23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128],
        ],
        [
            [1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128],
            [109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128],
            [44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128],
        ],
        [
            [1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128],
            [94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128],
            [22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128],
        ],
        [
            [1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128],
            [124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128],
            [35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128],
        ],
        [
            [1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128],
            [121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128],
            [45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128],
        ],
        [
            [1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128],
            [203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128],
            [137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128],
        ],
    ],
    // plane 2 — U or V
    [
        [
            [253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128],
            [175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128],
            [73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128],
        ],
        [
            [1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128],
            [239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128],
            [155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128],
        ],
        [
            [1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128],
            [201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128],
            [69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128],
        ],
        [
            [1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128],
            [223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128],
            [141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128],
        ],
        [
            [1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128],
            [190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128],
            [149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
        [
            [1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
        [
            [1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128],
            [213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128],
            [55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
        [
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
    ],
    // plane 3 — Y in the absence of Y2 (DCT starts at coefficient 0)
    [
        [
            [202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255],
            [126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128],
            [61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128],
        ],
        [
            [1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128],
            [166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128],
            [39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128],
        ],
        [
            [1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128],
            [124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128],
            [24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128],
        ],
        [
            [1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128],
            [149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128],
            [28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128],
        ],
        [
            [1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128],
            [123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128],
            [20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128],
        ],
        [
            [1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128],
            [168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128],
            [47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128],
        ],
        [
            [1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128],
            [141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128],
            [42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128],
        ],
        [
            [1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
            [238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
        ],
    ],
];

/// Overlay an in-frame `token_prob_update()` block onto the §13.5
/// default token-probability table. The output is the resolved
/// `coeff_probs[4][8][3][11]` used by the per-frame DCT decode loop.
///
/// Each entry of `updates` that is `Some(p)` replaces the
/// corresponding default position; `None` leaves the default in
/// place. The on-the-wire ordering of `updates` matches RFC 6386
/// §19.2's `token_prob_update()` four-nested-loop sweep, so the
/// `[plane][band][prev_ctx][pos]` indexing on the type alias is
/// already aligned.
pub fn merge_default_token_probs(updates: &TokenProbUpdates) -> CoeffProbs {
    let mut out = DEFAULT_COEFF_PROBS;
    for (plane_idx, plane) in updates.iter().enumerate() {
        for (band_idx, band) in plane.iter().enumerate() {
            for (prev_idx, ctx) in band.iter().enumerate() {
                for (pos_idx, slot) in ctx.iter().enumerate() {
                    if let Some(p) = *slot {
                        out[plane_idx][band_idx][prev_idx][pos_idx] = p;
                    }
                }
            }
        }
    }
    out
}

/// Decode the variable-length "extra bits" trailing a `Cat1..Cat6`
/// token (RFC 6386 §13.2 `DCTextra`). `cat_probs` is the cat's
/// fixed probability list (`Pcat1..Pcat6`, terminator omitted).
#[inline]
fn read_extra_bits(dec: &mut BoolDecoder<'_>, cat_probs: &[u8]) -> Result<u16, BoolDecoderError> {
    let mut v: u16 = 0;
    for &p in cat_probs {
        let bit = dec.read_bool(p)? as u16;
        v = (v << 1) | bit;
    }
    Ok(v)
}

/// Identity write-order table — [`decode_block`]'s public contract is
/// scan-order output, so its core invocation writes position `c` to
/// slot `c`. The per-macroblock walk passes [`ZIGZAG`] instead and
/// gets raster-order output without a separate reorder pass.
const SCAN_IDENTITY: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The §13.3 token loop with the §13.2 `coeff_tree` descent written
/// out branch-by-branch.
///
/// Each `read_bool` below sits at one internal node of the §13.2
/// `coeff_tree` and reads `probs[node >> 1]`, exactly as the generic
/// §8.1 `treed_read` walk over [`COEFF_TREE`] would — the node-by-node
/// mapping is: `probs[0]` eob-vs-rest (node 0), `probs[1]` DCT_0
/// (node 2), `probs[2]` DCT_1 (node 4), `probs[3]` small-vs-category
/// (node 6), `probs[4]`/`probs[5]` DCT_2/3/4 (nodes 8/10),
/// `probs[6]`..`probs[10]` the cat1..cat6 fan-out (nodes 12..20).
/// Flattening the walk removes the per-step tree-table loads and lets
/// every leaf flow directly into its consequence — the §13.2 "skip
/// the dct_eob branch after a DCT_0" rule becomes the inner zero-run
/// loop re-entering at the DCT_0 node (with the §13.3 zero-class
/// `ctx3 = 0` row), so the `prevCoeffWasZero` flag disappears
/// entirely. The bit-read sequence is identical to the generic walk:
/// tree bits, then `DCTextra` bits, then the sign bit at probability
/// 128 (`fused_descent_matches_generic_tree_walk` proves it).
///
/// `plane_probs` is the resolved per-plane `[band][prev_ctx][11]`
/// slice of [`CoeffProbs`]; `seed_ctx3` the §13.3 neighbour-count
/// context for the first coefficient; `write_order[c]` the output
/// slot for scan position `c` ([`SCAN_IDENTITY`] or [`ZIGZAG`]).
/// Returns the number of non-zero coefficients written.
///
/// When the `DQ` const parameter is `true` the §14.1 dequant multiply
/// is fused into the coefficient write: each non-zero value is scaled
/// by `dc_factor` (raster slot 0) or `ac_factor` (raster slots 1..=15)
/// — the identical `i32` product / `i16` truncation
/// [`crate::dequant`]'s per-block apply performs — at the moment it is
/// produced, so the all-lanes second pass over the macroblock (400
/// occupancy-independent multiplies plus an 800-byte re-walk) never
/// runs. Zero coefficients need no scaling (`0 × factor` truncates to
/// `0`, exactly what the second pass stored), which is what makes the
/// fusion bit-identical by construction; the equivalence is pinned
/// stream-for-stream by `fused_dequant_walk_matches_decode_then_dequantize`.
/// With `DQ = false` the factors are ignored and the generated code is
/// the raw-coefficient walk, unchanged.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_block_core<const DQ: bool>(
    dec: &mut BoolDecoder<'_>,
    plane_probs: &[[[u8; 11]; 3]; 8],
    first_coeff: usize,
    seed_ctx3: usize,
    write_order: &[usize; 16],
    coeffs: &mut [i16; 16],
    dc_factor: i16,
    ac_factor: i16,
) -> Result<usize, DctTokenError> {
    let mut ctx3 = seed_ctx3;
    let mut non_zero_count = 0usize;
    let mut i = first_coeff;
    'position: while i < 16 {
        let mut probs = &plane_probs[COEFF_BANDS[i]][ctx3];
        // Node 0 — dct_eob vs everything else. Only reachable when the
        // previous token was not a DCT_0 (or at the first coefficient).
        if !dec.read_bool(probs[0])? {
            break;
        }
        // Node 2 — DCT_0 vs non-zero. A DCT_0 forbids dct_eob at the
        // next position (§13.2), so a run of zeros loops here directly,
        // each subsequent position using the §13.3 zero-class row.
        while !dec.read_bool(probs[1])? {
            coeffs[write_order[i]] = 0;
            i += 1;
            if i == 16 {
                break 'position;
            }
            probs = &plane_probs[COEFF_BANDS[i]][0];
        }
        // Non-zero magnitude. §13.3: the next position's context class
        // is 1 for |value| == 1, 2 for |value| > 1.
        let abs_value: u16 = if !dec.read_bool(probs[2])? {
            // Node 4 — DCT_1.
            ctx3 = 1;
            1
        } else {
            ctx3 = 2;
            if !dec.read_bool(probs[3])? {
                // Node 6 left — DCT_2 / DCT_3 / DCT_4 (nodes 8, 10).
                if !dec.read_bool(probs[4])? {
                    2
                } else if !dec.read_bool(probs[5])? {
                    3
                } else {
                    4
                }
            } else if !dec.read_bool(probs[6])? {
                // Node 12 left — cat1 / cat2 (node 14).
                if !dec.read_bool(probs[7])? {
                    CAT_BASE[0] + read_extra_bits(dec, PCAT1)?
                } else {
                    CAT_BASE[1] + read_extra_bits(dec, PCAT2)?
                }
            } else if !dec.read_bool(probs[8])? {
                // Node 16 left — cat3 / cat4 (node 18).
                if !dec.read_bool(probs[9])? {
                    CAT_BASE[2] + read_extra_bits(dec, PCAT3)?
                } else {
                    CAT_BASE[3] + read_extra_bits(dec, PCAT4)?
                }
            } else if !dec.read_bool(probs[10])? {
                // Node 20 — cat5 / cat6.
                CAT_BASE[4] + read_extra_bits(dec, PCAT5)?
            } else {
                CAT_BASE[5] + read_extra_bits(dec, PCAT6)?
            }
        };
        // Non-zero coefficients carry a sign bit at fixed probability
        // 128 (§13.2 page 62). §13.2 cat6 max value is 67 + (2^11 - 1)
        // = 2114, which still fits in i16 with sign.
        let sign = dec.read_bool(128)?;
        let signed = abs_value as i16;
        let slot = write_order[i];
        let mut value = if sign { -signed } else { signed };
        if DQ {
            // Fused §14.1 apply: DC factor for raster slot 0, AC for
            // slots 1..=15, product in i32 truncated to i16 — the exact
            // arithmetic of the per-block second pass this replaces.
            let factor = if slot == 0 { dc_factor } else { ac_factor };
            value = ((value as i32) * (factor as i32)) as i16;
        }
        coeffs[slot] = value;
        non_zero_count += 1;
        i += 1;
    }
    Ok(non_zero_count)
}

/// Decode one DCT-token sub-block per RFC 6386 §13.3 and return the
/// number of non-zero coefficients written. The recovered values are
/// written into `coeffs[firstCoeff..16]` (positions in the
/// already-zig-zagged sub-block order); positions past the implicit
/// `dct_eob` are left untouched (caller is expected to pre-zero
/// `coeffs`).
///
/// * `block_type` selects the §13.3 plane index and the
///   `firstCoeff` value.
/// * `coeff_probs` is the resolved `coeff_probs[4][8][3][11]`
///   produced by [`merge_default_token_probs`].
/// * `above_has_nonzero` / `left_has_nonzero` are the §13.3
///   "neighbour block has at least one non-zero coefficient"
///   predictors for the *first* coefficient only. Off-frame
///   neighbours are passed as `false` per §13.3 page 64.
/// * Returns `Ok(non_zero_count)` — the number of non-zero
///   coefficients decoded. `coeffs` is updated in place. A return
///   value of 0 means the block was all-zero (either an immediate
///   EOB at `firstCoeff` or a sequence of `DCT_0` tokens followed
///   by EOB).
pub fn decode_block(
    dec: &mut BoolDecoder<'_>,
    block_type: BlockType,
    coeff_probs: &CoeffProbs,
    above_has_nonzero: bool,
    left_has_nonzero: bool,
    coeffs: &mut [i16; 16],
) -> Result<usize, DctTokenError> {
    // §13.3 page 65: the third context-index (`ctx3`) is seeded by
    // the count of non-zero neighbours for the very first coefficient
    // only. After the first coefficient the field rolls over to the
    // class of the last decoded coefficient (handled inside the core).
    //
    // §13.3 page 67 calls the "skip dct_eob branch" flag
    // `prevCoeffWasZero`. It is `false` for the first coefficient
    // because there is no previous coefficient at all (i.e. we DO
    // permit an immediate dct_eob). After that, the spec-typo-corrected
    // semantics (see the module-level doc comment) — `prevCoeffWasZero
    // = (token == DCT_0)`, NOT the literal `true` the spec listing
    // types — are realised structurally by `decode_block_core`'s inner
    // zero-run loop, which re-enters the tree at the DCT_0 node.
    let ctx3: usize = (above_has_nonzero as usize) + (left_has_nonzero as usize);
    decode_block_core::<false>(
        dec,
        &coeff_probs[block_type.plane_index()],
        block_type.first_coeff(),
        ctx3,
        &SCAN_IDENTITY,
        coeffs,
        0,
        0,
    )
}

// ===========================================================================
// §13.3 per-macroblock token walk
// ===========================================================================

/// §13 page 60 zig-zag scan order, transcribed from the §20.16 (tokens.c)
/// reference annex `zigzag[16]`. `ZIGZAG[c]` is the raster (natural-order)
/// position of the coefficient at scan position `c`. The per-block token
/// loop fills coefficients in scan order; this table places each into its
/// raster slot for the §14 raster-order inverse transforms.
pub const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Number of per-plane non-zero predictor slots a macroblock maintains in
/// each direction (above / left): four Y columns/rows, two U, two V, and
/// one Y2 — the `token_entropy_ctx_t = int[4 + 2 + 2 + 1]` of the §20.16
/// reference annex.
pub const MB_ENTROPY_CTX_LEN: usize = 4 + 2 + 2 + 1;

/// The §20.16 `left_context_index[25]` — maps a macroblock's 25 residual
/// blocks (16 Y, then 4 U, 4 V, then the Y2 block as block 24) to a slot
/// in the nine-entry "left" non-zero predictor vector. Y subblocks share a
/// left slot per subblock *row* (slots 0..3); U and V each occupy two slots
/// (4/5 and 6/7); Y2 is slot 8.
const LEFT_CONTEXT_INDEX: [usize; 25] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, // 16 Y
    4, 4, 5, 5, // 4 U
    6, 6, 7, 7, // 4 V
    8, // Y2
];

/// The §20.16 `above_context_index[25]` — the companion mapping for the
/// "above" predictor vector. Y subblocks share an above slot per subblock
/// *column* (slots 0..3); U/V/Y2 mirror [`LEFT_CONTEXT_INDEX`].
const ABOVE_CONTEXT_INDEX: [usize; 25] = [
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, // 16 Y
    4, 5, 4, 5, // 4 U
    6, 7, 6, 7, // 4 V
    8, // Y2
];

/// One macroblock's worth of non-zero coefficient predictors for a single
/// direction (above *or* left), per the §13.3 "aggregate coefficient
/// predictor" description: a single Y2 predictor, two each for U and V, and
/// four for Y. A decoder maintains one [`MbEntropyCtx`] per macroblock
/// column for the "above" row and a single rolling "left" instance.
///
/// Each entry is `true` if the most-recently-decoded block mapping to that
/// slot had at least one non-zero coefficient. Off-frame predictors are
/// `false` (the §13.3 "non-existent predictors ... taken to be empty"
/// rule); a freshly [`Default`]-constructed context satisfies that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MbEntropyCtx {
    /// The nine non-zero flags, indexed by [`LEFT_CONTEXT_INDEX`] /
    /// [`ABOVE_CONTEXT_INDEX`]: slots 0..3 = Y, 4..5 = U, 6..7 = V,
    /// 8 = Y2.
    pub nonzero: [bool; MB_ENTROPY_CTX_LEN],
}

impl MbEntropyCtx {
    /// Apply the §20.16 `reset_mb_context` rule for a skipped macroblock.
    ///
    /// The eight Y/U/V slots are always cleared (a skipped macroblock has
    /// no residue, so every same-plane neighbour predictor it contributes
    /// is empty). The Y2 slot (8) is cleared **only** when this macroblock
    /// would have carried a Y2 block (i.e. a non-`B_PRED` / non-`SPLITMV`
    /// mode); otherwise it is preserved so that the most-recent Y2-bearing
    /// macroblock in this column / row still drives the Y2 predictor, per
    /// the §13.3 "most recent macroblock ... that has a Y2 block" rule.
    fn reset_for_skip(&mut self, has_y2: bool) {
        for slot in self.nonzero.iter_mut().take(8) {
            *slot = false;
        }
        if has_y2 {
            self.nonzero[8] = false;
        }
    }
}

/// Out-of-band conditions surfaced by [`decode_mb_coeffs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbCoeffError {
    /// A per-block token decode failed (bool-decoder ran dry, or a
    /// corrupted tree table). Carries the residual block index
    /// (`0..=24`, dixie order: 0..15 Y, 16..19 U, 20..23 V, 24 Y2) and
    /// the underlying error.
    Block {
        /// The §20.16 residual block index that failed.
        index: usize,
        /// The per-block decode error.
        source: DctTokenError,
    },
}

impl core::fmt::Display for MbCoeffError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MbCoeffError::Block { index, source } => {
                write!(f, "vp8 mb coeffs: residual block {index}: {source}")
            }
        }
    }
}

impl std::error::Error for MbCoeffError {}

/// Reorder a scan-order coefficient block into raster (natural) order via
/// the §20.16 zig-zag table. `scan[c]` (scan position `c`) is placed at
/// raster position `ZIGZAG[c]`.
///
/// The production per-macroblock walk no longer needs this pass — it
/// hands [`ZIGZAG`] to `decode_block_core` as the write-order table so
/// each coefficient lands in its raster slot as it is decoded. Kept as
/// the test-side reference permutation.
#[cfg(test)]
fn scan_to_raster(scan: &[i16; 16]) -> [i16; 16] {
    let mut raster = [0i16; 16];
    for (c, &v) in scan.iter().enumerate() {
        raster[ZIGZAG[c]] = v;
    }
    raster
}

/// Decode one macroblock's worth of DCT/WHT coefficient tokens per RFC
/// 6386 §13.3, threading the above / left non-zero predictor context the
/// §13.3 third-dimension probability index requires.
///
/// This is the integration layer over [`decode_block`]: it walks all the
/// residual blocks of one macroblock — the §14.2 Y2 (WHT) block first when
/// present, then the sixteen Y 4×4 DCT blocks, then the four U and four V
/// chroma blocks, in the §13 "residual_data()" order — selecting the
/// correct [`BlockType`] plane and the correct above / left predictor slot
/// per the §20.16 `left_context_index` / `above_context_index` tables, and
/// updating those slots with each block's non-zero status for the
/// macroblocks below and to the right (§13.3 "after each block is decoded,
/// the two predictors referenced by the block are replaced").
///
/// # Inputs
///
/// * `dec` — the §13.2 boolean decoder positioned at this macroblock's
///   residual record in the (second) DCT partition.
/// * `has_y2` — whether this macroblock carries a Y2 block. `true` for the
///   four 16×16 luma modes (and inter modes other than `SPLITMV`); `false`
///   for `B_PRED` / `SPLITMV`, whose Y blocks instead carry their own DC.
///   The caller derives this from the decoded [`crate::macroblock::IntraYMode`]
///   (`!= B`).
/// * `mb_skip_coeff` — the §11.1 / §13.1 skip flag. When `true` no tokens
///   are read; the returned coefficients are all zero and the predictor
///   context is updated per the §20.16 `reset_mb_context` rule (clear all
///   Y/U/V slots; clear Y2 only when `has_y2`).
/// * `coeff_probs` — the resolved `coeff_probs[4][8][3][11]` from
///   [`merge_default_token_probs`].
/// * `above` / `left` — the macroblock's incoming non-zero predictor
///   contexts. Both are read for the *first* coefficient of each block and
///   updated in place as blocks are decoded. The caller maintains a row of
///   `above` contexts (one per macroblock column) and a single rolling
///   `left` context, clearing both at the appropriate frame / row edges.
///
/// # Output
///
/// The macroblock's coefficients as an [`crate::frame::MbCoeffs`], with
/// each 4×4 block placed in **raster (natural) order** (the layout the §14
/// inverse transforms consume). The coefficients are the **raw quantized
/// token values** — the §14.1 Y2 / chroma dequant scaling is a documented
/// spec gap (§14.1 page 77 defers it to `dixie.c` §20.4), so this function
/// does **not** multiply by any dequant factor. The caller must apply
/// §14.1 dequantization before the inverse transforms once that gap is
/// filled.
pub fn decode_mb_coeffs(
    dec: &mut BoolDecoder<'_>,
    has_y2: bool,
    mb_skip_coeff: bool,
    coeff_probs: &CoeffProbs,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<crate::frame::MbCoeffs, MbCoeffError> {
    // Raw walk: the factor pairs are ignored under `DQ = false`.
    decode_mb_coeffs_inner::<false>(
        dec,
        has_y2,
        mb_skip_coeff,
        coeff_probs,
        &MbDequantPairs {
            y2: (0, 0),
            y1: (0, 0),
            uv: (0, 0),
        },
        above,
        left,
    )
}

/// Per-plane §14.1 `(DC, AC)` dequant factor pairs for the fused
/// decode→dequant walk ([`decode_mb_coeffs_dequant`]). A plain
/// destructuring of [`crate::dequant::MbDequantFactors`] into the three
/// plane pairs the §13 residual order selects between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MbDequantPairs {
    /// Y2 (the 25th, WHT) block factors.
    pub y2: (i16, i16),
    /// Y1 (the sixteen luma DCT sub-blocks) factors.
    pub y1: (i16, i16),
    /// Chroma (the four U and four V sub-blocks) factors.
    pub uv: (i16, i16),
}

/// The fused §13.3 decode → §14.1 dequant walk: identical to
/// [`decode_mb_coeffs`] except that every non-zero coefficient is
/// scaled by its plane's DC/AC factor at the moment it is written, so
/// the caller receives the **dequantized** [`crate::frame::MbCoeffs`]
/// directly and the per-macroblock second pass
/// ([`crate::dequant::MbDequantFactors::dequantize`] — 400
/// occupancy-independent multiplies over all twenty-five blocks) is
/// never needed. Bit-identical to decode-then-dequantize by
/// construction (untouched / zero-run lanes hold `0`, and `0 × factor`
/// truncates to `0`); pinned stream-for-stream by
/// `fused_dequant_walk_matches_decode_then_dequantize`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_mb_coeffs_dequant(
    dec: &mut BoolDecoder<'_>,
    has_y2: bool,
    mb_skip_coeff: bool,
    coeff_probs: &CoeffProbs,
    pairs: &MbDequantPairs,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<crate::frame::MbCoeffs, MbCoeffError> {
    decode_mb_coeffs_inner::<true>(dec, has_y2, mb_skip_coeff, coeff_probs, pairs, above, left)
}

/// Shared body of [`decode_mb_coeffs`] / [`decode_mb_coeffs_dequant`]:
/// the §13 / §14.2 residual-order walk, monomorphised on whether the
/// §14.1 apply is fused into the coefficient writes.
#[allow(clippy::too_many_arguments)]
fn decode_mb_coeffs_inner<const DQ: bool>(
    dec: &mut BoolDecoder<'_>,
    has_y2: bool,
    mb_skip_coeff: bool,
    coeff_probs: &CoeffProbs,
    pairs: &MbDequantPairs,
    above: &mut MbEntropyCtx,
    left: &mut MbEntropyCtx,
) -> Result<crate::frame::MbCoeffs, MbCoeffError> {
    let mut out = crate::frame::MbCoeffs::default();

    // §13.1: a skip macroblock reads no tokens. Update the predictor
    // context per the §20.16 `reset_mb_context` rule and return zeros
    // (which dequantize to zeros, so the fused walk returns them as-is).
    if mb_skip_coeff {
        above.reset_for_skip(has_y2);
        left.reset_for_skip(has_y2);
        return Ok(out);
    }

    // Decode one residual block: select the predictor slots, run the
    // per-coefficient token loop (writing each coefficient straight
    // into its raster slot via the §20.16 `ZIGZAG` write order — no
    // separate reorder pass), and write back the non-zero status into
    // both predictor vectors. `block_index` is the §20.16 residual
    // block index (0..=24); `(dc_factor, ac_factor)` the plane's §14.1
    // pair (ignored under `DQ = false`).
    let decode_one = |dec: &mut BoolDecoder<'_>,
                      block_index: usize,
                      block_type: BlockType,
                      (dc_factor, ac_factor): (i16, i16),
                      above: &mut MbEntropyCtx,
                      left: &mut MbEntropyCtx,
                      raster: &mut [i16; 16]|
     -> Result<(), MbCoeffError> {
        let a_slot = ABOVE_CONTEXT_INDEX[block_index];
        let l_slot = LEFT_CONTEXT_INDEX[block_index];
        // §13.3 page 65: ctx3 for the first coefficient is the count
        // of non-zero neighbours.
        let ctx3 = (above.nonzero[a_slot] as usize) + (left.nonzero[l_slot] as usize);
        let nz = decode_block_core::<DQ>(
            dec,
            &coeff_probs[block_type.plane_index()],
            block_type.first_coeff(),
            ctx3,
            &ZIGZAG,
            raster,
            dc_factor,
            ac_factor,
        )
        .map_err(|source| MbCoeffError::Block {
            index: block_index,
            source,
        })?;
        // §13.3: replace the two referenced predictors with this
        // block's (non-)empty state for blocks below and to the right.
        let has_coeffs = nz != 0;
        above.nonzero[a_slot] = has_coeffs;
        left.nonzero[l_slot] = has_coeffs;
        Ok(())
    };

    // §13 / §14.2 residual order: Y2 (when present) → 16 Y → 4 U → 4 V.
    // The Y luma plane type is `YAfterY2` (DCT starts at coefficient 1)
    // when a Y2 block carries the DCs, else `YNoY2` (DCT starts at 0).
    if has_y2 {
        // Y2 is residual block 24 in the §20.16 index tables.
        decode_one(dec, 24, BlockType::Y2, pairs.y2, above, left, &mut out.y2)?;
    }

    let y_plane = if has_y2 {
        BlockType::YAfterY2
    } else {
        BlockType::YNoY2
    };
    for (i, y_block) in out.y.iter_mut().enumerate() {
        decode_one(dec, i, y_plane, pairs.y1, above, left, y_block)?;
    }

    // U occupies residual blocks 16..=19, V occupies 20..=23.
    for (i, u_block) in out.u.iter_mut().enumerate() {
        decode_one(dec, 16 + i, BlockType::UV, pairs.uv, above, left, u_block)?;
    }
    for (i, v_block) in out.v.iter_mut().enumerate() {
        decode_one(dec, 20 + i, BlockType::UV, pairs.uv, above, left, v_block)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test-side bool encoder (mirrors the one inside
    // `bool_decoder::tests`). Lives only in tests; never exported.
    struct TestEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl TestEncoder {
        fn new() -> Self {
            Self {
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
            self.out
        }
    }

    /// Encode a single DctToken against the given probability vector by
    /// walking the same tree the decoder will walk. Returns the (start
    /// internal node, list of (prob_index, bit) pairs to write).
    ///
    /// `start_at_index` is `0` if we may emit eob, `2` if not (i.e.
    /// when the previous coefficient was a DCT_0).
    fn encode_token(token: DctToken, start_at_index: i8) -> Vec<(usize, bool)> {
        // We replay the tree-walk and at each step record (i>>1, bit)
        // that the decoder would have read. We do this by knowing the
        // tree statically.
        let target = token as i8;
        let mut out = Vec::new();
        // Helper: at internal node `i`, the two children are
        // COEFF_TREE[i] (left, bit=0) and COEFF_TREE[i+1] (right, bit=1).
        fn descend(i: i8, target: i8, path: &mut Vec<(usize, bool)>) -> bool {
            for &bit in &[false, true] {
                let child = COEFF_TREE[i as usize + bit as usize];
                if child <= 0 {
                    if -child == target {
                        path.push(((i as usize) >> 1, bit));
                        return true;
                    }
                } else {
                    path.push(((i as usize) >> 1, bit));
                    if descend(child, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            false
        }
        assert!(descend(start_at_index, target, &mut out));
        out
    }

    /// Encode a full block of coefficients against the resolved coeff
    /// probability table and return the bool-encoder byte stream that
    /// `decode_block` should consume.
    fn encode_block(
        coeffs: &[i16; 16],
        block_type: BlockType,
        coeff_probs: &CoeffProbs,
        above_has_nonzero: bool,
        left_has_nonzero: bool,
    ) -> Vec<u8> {
        let mut enc = TestEncoder::new();
        let plane = block_type.plane_index();
        let first_coeff = block_type.first_coeff();
        let mut ctx3 = (above_has_nonzero as usize) + (left_has_nonzero as usize);
        let mut prev_was_zero = false;

        // Find the position of the last non-zero coefficient — we will
        // emit an EOB token immediately after that position (or at
        // first_coeff if the block is empty).
        let mut last_non_zero: i32 = -1;
        for (idx, c) in coeffs.iter().enumerate().take(16) {
            if idx >= first_coeff && *c != 0 {
                last_non_zero = idx as i32;
            }
        }

        let mut i = first_coeff as i32;
        while i < 16 {
            let band = COEFF_BANDS[i as usize];
            let probs = &coeff_probs[plane][band][ctx3];

            let emit_eob = i > last_non_zero;
            let abs_value: u16;
            let sign: bool;
            let token: DctToken;
            if emit_eob {
                token = DctToken::Eob;
                abs_value = 0;
                sign = false;
            } else {
                let v = coeffs[i as usize];
                sign = v < 0;
                abs_value = v.unsigned_abs();
                token = if abs_value == 0 {
                    DctToken::Dct0
                } else if abs_value == 1 {
                    DctToken::Dct1
                } else if abs_value == 2 {
                    DctToken::Dct2
                } else if abs_value == 3 {
                    DctToken::Dct3
                } else if abs_value == 4 {
                    DctToken::Dct4
                } else if abs_value <= 6 {
                    DctToken::Cat1
                } else if abs_value <= 10 {
                    DctToken::Cat2
                } else if abs_value <= 18 {
                    DctToken::Cat3
                } else if abs_value <= 34 {
                    DctToken::Cat4
                } else if abs_value <= 66 {
                    DctToken::Cat5
                } else {
                    DctToken::Cat6
                };
            }

            let start = if prev_was_zero { 2i8 } else { 0i8 };
            for (i_half, bit) in encode_token(token, start) {
                enc.write_bool(probs[i_half], bit);
            }
            if token == DctToken::Eob {
                break;
            }

            if let Some((base, plist)) = match token {
                DctToken::Cat1 => Some((CAT_BASE[0], PCAT1)),
                DctToken::Cat2 => Some((CAT_BASE[1], PCAT2)),
                DctToken::Cat3 => Some((CAT_BASE[2], PCAT3)),
                DctToken::Cat4 => Some((CAT_BASE[3], PCAT4)),
                DctToken::Cat5 => Some((CAT_BASE[4], PCAT5)),
                DctToken::Cat6 => Some((CAT_BASE[5], PCAT6)),
                _ => None,
            } {
                let extra = abs_value - base;
                let n = plist.len();
                for (j, &p) in plist.iter().enumerate() {
                    let bit = ((extra >> (n - 1 - j)) & 1) == 1;
                    enc.write_bool(p, bit);
                }
            }

            if abs_value != 0 {
                enc.write_bool(128, sign);
            }

            ctx3 = if abs_value == 0 {
                0
            } else if abs_value == 1 {
                1
            } else {
                2
            };
            prev_was_zero = token == DctToken::Dct0;
            i += 1;
        }

        enc.finish()
    }

    #[test]
    fn default_probs_table_shape() {
        // Cheap sanity check on the §13.5 transcription: every entry
        // is a u8 in the legal range and the table has 4*8*3*11=1056
        // probabilities.
        let mut count = 0;
        for plane in &DEFAULT_COEFF_PROBS {
            for band in plane {
                for ctx in band {
                    for &_p in ctx {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 4 * 8 * 3 * 11);
    }

    #[test]
    fn default_probs_first_row_is_uniform_128() {
        // Plane 0 (Y after Y2) band 0 is all-128 in the RFC listing —
        // unused since plane 0 starts at coefficient 1, so band 0
        // never reads. Spot-check the first row.
        for ctx in &DEFAULT_COEFF_PROBS[0][0] {
            for &p in ctx {
                assert_eq!(p, 128);
            }
        }
    }

    #[test]
    fn default_probs_known_values() {
        // A handful of specific cells from §13.5 — would catch a
        // transcription typo on any of the four touched planes.
        assert_eq!(DEFAULT_COEFF_PROBS[0][1][0][0], 253);
        assert_eq!(DEFAULT_COEFF_PROBS[1][0][0][10], 62);
        assert_eq!(DEFAULT_COEFF_PROBS[2][7][2][0], 128); // chroma plane band 7 all-128
        assert_eq!(DEFAULT_COEFF_PROBS[3][0][0][0], 202);
        assert_eq!(DEFAULT_COEFF_PROBS[3][7][2][1], 1); // last block: { 238, 1, 255, ... }
    }

    #[test]
    fn coeff_bands_is_spec_listing() {
        assert_eq!(
            COEFF_BANDS,
            [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7]
        );
    }

    #[test]
    fn block_type_first_coeff_and_plane_index() {
        assert_eq!(BlockType::YAfterY2.first_coeff(), 1);
        assert_eq!(BlockType::Y2.first_coeff(), 0);
        assert_eq!(BlockType::UV.first_coeff(), 0);
        assert_eq!(BlockType::YNoY2.first_coeff(), 0);
        assert_eq!(BlockType::YAfterY2.plane_index(), 0);
        assert_eq!(BlockType::Y2.plane_index(), 1);
        assert_eq!(BlockType::UV.plane_index(), 2);
        assert_eq!(BlockType::YNoY2.plane_index(), 3);
    }

    #[test]
    fn merge_no_updates_is_identity() {
        let updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        let merged = merge_default_token_probs(&updates);
        assert_eq!(merged, DEFAULT_COEFF_PROBS);
    }

    #[test]
    fn merge_overlays_present_entries() {
        let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        updates[2][3][1][5] = Some(42);
        updates[0][0][0][0] = Some(7);
        let merged = merge_default_token_probs(&updates);
        assert_eq!(merged[2][3][1][5], 42);
        assert_eq!(merged[0][0][0][0], 7);
        // Untouched cells still match defaults.
        assert_eq!(merged[3][5][2][1], DEFAULT_COEFF_PROBS[3][5][2][1]);
    }

    #[test]
    fn decode_immediate_eob_yields_all_zero_block() {
        // Block: 16 zeros; first token is dct_eob.
        let probs = &DEFAULT_COEFF_PROBS;
        let coeffs = [0i16; 16];
        let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 0);
        assert_eq!(out, [0i16; 16]);
    }

    #[test]
    fn decode_roundtrip_single_dct1_at_position_0() {
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;
        let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 1);
        assert_eq!(out[0], 1);
        for &v in out.iter().skip(1) {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn decode_roundtrip_negative_value() {
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[0] = -3;
        coeffs[1] = -2;
        let buf = encode_block(&coeffs, BlockType::Y2, probs, true, true);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, true, true, &mut out).unwrap();
        assert_eq!(nz, 2);
        assert_eq!(out[0], -3);
        assert_eq!(out[1], -2);
    }

    #[test]
    fn decode_roundtrip_each_cat_range() {
        // One value from inside each cat1..cat6 range. Verifies the
        // base + extra-bits arithmetic at the cat boundary, halfway
        // inside the range, and at the cat ceiling.
        for &val in &[5i16, 6, 7, 10, 11, 18, 19, 34, 35, 66, 67, 100, 2048] {
            let probs = &DEFAULT_COEFF_PROBS;
            let mut coeffs = [0i16; 16];
            coeffs[0] = val;
            let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
            let mut dec = BoolDecoder::init(&buf).unwrap();
            let mut out = [0i16; 16];
            let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
            assert_eq!(nz, 1, "val={val}");
            assert_eq!(out[0], val, "val={val}");
        }
    }

    #[test]
    fn decode_y_after_y2_skips_dc_position() {
        // Plane 0 (`YAfterY2`) starts at coefficient 1; position 0
        // must stay zero even if the encoded bits would otherwise
        // ascribe a token there. Verify by encoding a value at
        // position 1.
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[1] = 7; // cat2
        let buf = encode_block(&coeffs, BlockType::YAfterY2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz =
            decode_block(&mut dec, BlockType::YAfterY2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 1);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 7);
    }

    #[test]
    fn decode_dense_block_zigzag_positions() {
        // A non-trivial block with values at multiple positions
        // including the rollover from DCT_0 -> non-zero (covering
        // the §13.3 `prev_was_zero` / `ctx3` rollover).
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 3;
        coeffs[1] = 0;
        coeffs[2] = -1;
        coeffs[3] = 2;
        coeffs[5] = 15; // cat3
        coeffs[8] = -25; // cat4
        let buf = encode_block(&coeffs, BlockType::UV, probs, true, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::UV, probs, true, false, &mut out).unwrap();
        assert_eq!(nz, 5);
        assert_eq!(out, coeffs);
    }

    #[test]
    fn decode_under_overlaid_probability_updates() {
        // Use a TokenProbUpdates that overrides a handful of entries
        // and verify decode still round-trips. Catches a bug where
        // `decode_block` accidentally hits `DEFAULT_COEFF_PROBS`
        // instead of the caller-supplied table.
        let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        updates[2][1][0][0] = Some(200);
        updates[2][1][0][1] = Some(50);
        updates[2][2][1][2] = Some(99);
        let probs = merge_default_token_probs(&updates);

        let mut coeffs = [0i16; 16];
        coeffs[0] = 4;
        coeffs[1] = -1;
        coeffs[3] = 12; // cat3
        let buf = encode_block(&coeffs, BlockType::UV, &probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::UV, &probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 3);
        assert_eq!(out, coeffs);
    }

    #[test]
    fn prev_token_skips_eob_branch() {
        // Encode a block where the second coefficient is reached
        // through the "previous was zero" path. The decoder must
        // start the second token's tree-walk at index 2 (skipping
        // the dct_eob branch). To prove this matters: emit a long
        // run of zeros that would, under the buggy spec literal
        // `prev_was_zero = true`, be interpreted as eob.
        //
        // We rely on the encode helper applying the same "start at
        // 2 when prev was zero" rule; if `decode_block` did the
        // wrong thing, the round-trip would fail.
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 0;
        coeffs[1] = 0;
        coeffs[2] = 0;
        coeffs[3] = 5; // cat1
        let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 1);
        assert_eq!(out, coeffs);
    }

    #[test]
    fn neighbour_predictor_count_seeds_ctx3() {
        // The `ctx3` for the very first coefficient is the count
        // (0/1/2) of non-zero neighbours. Encoding with all four
        // cases (00, 01, 10, 11) and round-tripping makes sure the
        // decoder picks the same row.
        let probs = &DEFAULT_COEFF_PROBS;
        for (above, left) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut coeffs = [0i16; 16];
            coeffs[0] = -7; // cat2
            coeffs[2] = 1;
            let buf = encode_block(&coeffs, BlockType::UV, probs, above, left);
            let mut dec = BoolDecoder::init(&buf).unwrap();
            let mut out = [0i16; 16];
            let nz = decode_block(&mut dec, BlockType::UV, probs, above, left, &mut out).unwrap();
            assert_eq!(nz, 2, "above={above} left={left}");
            assert_eq!(out, coeffs, "above={above} left={left}");
        }
    }

    #[test]
    fn block_full_to_position_15_no_eob_emitted() {
        // A block that uses all 16 positions emits no EOB at all —
        // the §13.3 loop runs to completion with i = 16. Verify
        // the decoder handles that termination correctly.
        let probs = &DEFAULT_COEFF_PROBS;
        let coeffs = [1i16; 16];
        let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 16);
        assert_eq!(out, coeffs);
    }

    #[test]
    fn cat6_high_magnitude_roundtrip() {
        // cat6 with the largest legal magnitude (67 + 2047 = 2114).
        // Exercises the 11-extra-bit DCTextra path.
        let probs = &DEFAULT_COEFF_PROBS;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 2114;
        let buf = encode_block(&coeffs, BlockType::Y2, probs, false, false);
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut out = [0i16; 16];
        let nz = decode_block(&mut dec, BlockType::Y2, probs, false, false, &mut out).unwrap();
        assert_eq!(nz, 1);
        assert_eq!(out[0], 2114);
    }

    /// Reference §8.1 `treed_read` over [`COEFF_TREE`]: the generic
    /// table-driven walk the fused production descent replaced. Ground
    /// truth for `fused_descent_matches_generic_tree_walk`.
    fn treed_read_coef_reference(
        dec: &mut BoolDecoder<'_>,
        probs: &[u8; 11],
        start_index: i8,
    ) -> Result<DctToken, DctTokenError> {
        let mut i: i8 = start_index;
        loop {
            let prob = probs[(i as usize) >> 1];
            let bit = dec.read_bool(prob)? as usize;
            let next = COEFF_TREE[i as usize + bit];
            if next <= 0 {
                let leaf = -next as u8;
                return Ok(match leaf {
                    0 => DctToken::Dct0,
                    1 => DctToken::Dct1,
                    2 => DctToken::Dct2,
                    3 => DctToken::Dct3,
                    4 => DctToken::Dct4,
                    5 => DctToken::Cat1,
                    6 => DctToken::Cat2,
                    7 => DctToken::Cat3,
                    8 => DctToken::Cat4,
                    9 => DctToken::Cat5,
                    10 => DctToken::Cat6,
                    11 => DctToken::Eob,
                    _ => return Err(DctTokenError::InvalidTokenIndex),
                });
            }
            i = next;
        }
    }

    /// Reference §13.3 token loop built on the generic tree walk — the
    /// pre-fusion `decode_block` body, kept verbatim as ground truth.
    fn decode_block_reference(
        dec: &mut BoolDecoder<'_>,
        block_type: BlockType,
        coeff_probs: &CoeffProbs,
        above_has_nonzero: bool,
        left_has_nonzero: bool,
        coeffs: &mut [i16; 16],
    ) -> Result<usize, DctTokenError> {
        let mut ctx3: usize = (above_has_nonzero as usize) + (left_has_nonzero as usize);
        let plane = block_type.plane_index();
        let first_coeff = block_type.first_coeff();
        let mut prev_was_zero = false;
        let mut non_zero_count = 0usize;
        let mut i = first_coeff;
        while i < 16 {
            let band = COEFF_BANDS[i];
            let probs = &coeff_probs[plane][band][ctx3];
            let start = if prev_was_zero { 2i8 } else { 0i8 };
            let token = treed_read_coef_reference(dec, probs, start)?;
            if token == DctToken::Eob {
                break;
            }
            let abs_value: u16 = match token {
                DctToken::Dct0 => 0,
                DctToken::Dct1 => 1,
                DctToken::Dct2 => 2,
                DctToken::Dct3 => 3,
                DctToken::Dct4 => 4,
                DctToken::Cat1 => CAT_BASE[0] + read_extra_bits(dec, PCAT1)?,
                DctToken::Cat2 => CAT_BASE[1] + read_extra_bits(dec, PCAT2)?,
                DctToken::Cat3 => CAT_BASE[2] + read_extra_bits(dec, PCAT3)?,
                DctToken::Cat4 => CAT_BASE[3] + read_extra_bits(dec, PCAT4)?,
                DctToken::Cat5 => CAT_BASE[4] + read_extra_bits(dec, PCAT5)?,
                DctToken::Cat6 => CAT_BASE[5] + read_extra_bits(dec, PCAT6)?,
                DctToken::Eob => unreachable!("eob handled above"),
            };
            if abs_value != 0 {
                let sign = dec.read_bool(128)?;
                let signed = abs_value as i16;
                coeffs[i] = if sign { -signed } else { signed };
                non_zero_count += 1;
            } else {
                coeffs[i] = 0;
            }
            ctx3 = if abs_value == 0 {
                0
            } else if abs_value == 1 {
                1
            } else {
                2
            };
            prev_was_zero = token == DctToken::Dct0;
            i += 1;
        }
        Ok(non_zero_count)
    }

    #[test]
    fn fused_descent_matches_generic_tree_walk() {
        // Drive the fused production `decode_block` and the generic
        // table-driven reference over a deterministic randomised corpus
        // covering every plane type, every neighbour-context seed, and
        // a sparsity/magnitude mix that reaches every tree leaf
        // (zero runs, DCT_1..4, all six DCTextra categories, immediate
        // EOB, and full-16 blocks). Outputs, non-zero counts, and the
        // complete bool-decoder end state must agree exactly — proving
        // the fused walk consumes the identical bit sequence.
        let mut seed = 0x9e3779b9u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let block_types = [
            BlockType::YAfterY2,
            BlockType::Y2,
            BlockType::UV,
            BlockType::YNoY2,
        ];
        // A second probability table with overlays, so the corpus is
        // not tied to the defaults.
        let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        updates[0][1][0][0] = Some(90);
        updates[1][0][1][3] = Some(17);
        updates[3][6][2][1] = Some(201);
        let overlaid = merge_default_token_probs(&updates);
        let tables = [&DEFAULT_COEFF_PROBS, &overlaid];

        for trial in 0..400 {
            let block_type = block_types[(rng() % 4) as usize];
            let probs = tables[(trial % 2) as usize];
            let above = rng() & 1 == 1;
            let left = rng() & 1 == 1;
            // Sparsity dial: 0 = empty block, up to dense full blocks.
            let density = rng() % 17;
            let mut coeffs = [0i16; 16];
            for slot in coeffs.iter_mut().skip(block_type.first_coeff()) {
                if rng() % 16 < density {
                    // Magnitude classes hitting every leaf: 1..4 small,
                    // 5..2114 across cat1..cat6.
                    let m = match rng() % 8 {
                        0 => 1,
                        1 => 2,
                        2 => 3,
                        3 => 4,
                        4 => 5 + (rng() % 6) as i16,     // cat1/cat2
                        5 => 11 + (rng() % 24) as i16,   // cat3/cat4
                        6 => 35 + (rng() % 32) as i16,   // cat5
                        _ => 67 + (rng() % 2048) as i16, // cat6
                    };
                    *slot = if rng() & 1 == 1 { -m } else { m };
                }
            }
            let buf = encode_block(&coeffs, block_type, probs, above, left);

            let mut dec_fused = BoolDecoder::init(&buf).unwrap();
            let mut out_fused = [0i16; 16];
            let nz_fused = decode_block(
                &mut dec_fused,
                block_type,
                probs,
                above,
                left,
                &mut out_fused,
            )
            .unwrap();

            let mut dec_ref = BoolDecoder::init(&buf).unwrap();
            let mut out_ref = [0i16; 16];
            let nz_ref =
                decode_block_reference(&mut dec_ref, block_type, probs, above, left, &mut out_ref)
                    .unwrap();

            assert_eq!(nz_fused, nz_ref, "trial {trial}: non-zero count");
            assert_eq!(out_fused, out_ref, "trial {trial}: coefficients");
            assert_eq!(out_fused, coeffs, "trial {trial}: roundtrip");
            assert_eq!(
                dec_fused.range(),
                dec_ref.range(),
                "trial {trial}: range state"
            );
            assert_eq!(
                dec_fused.value(),
                dec_ref.value(),
                "trial {trial}: value state"
            );
            assert_eq!(
                dec_fused.remaining_input(),
                dec_ref.remaining_input(),
                "trial {trial}: input cursor"
            );
        }
    }

    // -----------------------------------------------------------------
    // §13.3 per-macroblock walk tests
    // -----------------------------------------------------------------

    /// Reorder a raster-order block back into scan order (the inverse of
    /// `scan_to_raster`) so a test can specify the expected raster layout
    /// and feed `encode_block` (which works in scan order).
    fn raster_to_scan(raster: &[i16; 16]) -> [i16; 16] {
        let mut scan = [0i16; 16];
        for (c, slot) in scan.iter_mut().enumerate() {
            *slot = raster[ZIGZAG[c]];
        }
        scan
    }

    /// A single shared bool-encoder that emits the whole macroblock
    /// residual record contiguously (the per-block `TestEncoder` cannot be
    /// concatenated). Mirrors `decode_mb_coeffs` exactly: block order,
    /// plane selection, zig-zag, and predictor threading.
    struct MbEncoder {
        enc: TestEncoder,
    }

    impl MbEncoder {
        fn new() -> Self {
            Self {
                enc: TestEncoder::new(),
            }
        }

        fn write_block(
            &mut self,
            block_index: usize,
            block_type: BlockType,
            raster: &[i16; 16],
            coeff_probs: &CoeffProbs,
            above: &mut MbEntropyCtx,
            left: &mut MbEntropyCtx,
        ) {
            let a_slot = ABOVE_CONTEXT_INDEX[block_index];
            let l_slot = LEFT_CONTEXT_INDEX[block_index];
            let above_nz = above.nonzero[a_slot];
            let left_nz = left.nonzero[l_slot];
            let scan = raster_to_scan(raster);

            // Replay the §13.3 token loop, writing bits into the shared
            // encoder (this is `encode_block`'s body, inlined to share the
            // encoder state across blocks).
            let plane = block_type.plane_index();
            let first_coeff = block_type.first_coeff();
            let mut ctx3 = (above_nz as usize) + (left_nz as usize);
            let mut prev_was_zero = false;
            let mut last_non_zero: i32 = -1;
            for (idx, c) in scan.iter().enumerate().take(16) {
                if idx >= first_coeff && *c != 0 {
                    last_non_zero = idx as i32;
                }
            }
            let mut nonzero_count = 0usize;
            let mut i = first_coeff as i32;
            while i < 16 {
                let band = COEFF_BANDS[i as usize];
                let probs = &coeff_probs[plane][band][ctx3];
                let emit_eob = i > last_non_zero;
                let abs_value: u16;
                let sign: bool;
                let token: DctToken;
                if emit_eob {
                    token = DctToken::Eob;
                    abs_value = 0;
                    sign = false;
                } else {
                    let v = scan[i as usize];
                    sign = v < 0;
                    abs_value = v.unsigned_abs();
                    token = if abs_value == 0 {
                        DctToken::Dct0
                    } else if abs_value == 1 {
                        DctToken::Dct1
                    } else if abs_value == 2 {
                        DctToken::Dct2
                    } else if abs_value == 3 {
                        DctToken::Dct3
                    } else if abs_value == 4 {
                        DctToken::Dct4
                    } else if abs_value <= 6 {
                        DctToken::Cat1
                    } else if abs_value <= 10 {
                        DctToken::Cat2
                    } else if abs_value <= 18 {
                        DctToken::Cat3
                    } else if abs_value <= 34 {
                        DctToken::Cat4
                    } else if abs_value <= 66 {
                        DctToken::Cat5
                    } else {
                        DctToken::Cat6
                    };
                }
                let start = if prev_was_zero { 2i8 } else { 0i8 };
                for (i_half, bit) in encode_token(token, start) {
                    self.enc.write_bool(probs[i_half], bit);
                }
                if token == DctToken::Eob {
                    break;
                }
                if let Some((base, plist)) = match token {
                    DctToken::Cat1 => Some((CAT_BASE[0], PCAT1)),
                    DctToken::Cat2 => Some((CAT_BASE[1], PCAT2)),
                    DctToken::Cat3 => Some((CAT_BASE[2], PCAT3)),
                    DctToken::Cat4 => Some((CAT_BASE[3], PCAT4)),
                    DctToken::Cat5 => Some((CAT_BASE[4], PCAT5)),
                    DctToken::Cat6 => Some((CAT_BASE[5], PCAT6)),
                    _ => None,
                } {
                    let extra = abs_value - base;
                    let n = plist.len();
                    for (j, &p) in plist.iter().enumerate() {
                        let bit = ((extra >> (n - 1 - j)) & 1) == 1;
                        self.enc.write_bool(p, bit);
                    }
                }
                if abs_value != 0 {
                    self.enc.write_bool(128, sign);
                    nonzero_count += 1;
                }
                ctx3 = if abs_value == 0 {
                    0
                } else if abs_value == 1 {
                    1
                } else {
                    2
                };
                prev_was_zero = token == DctToken::Dct0;
                i += 1;
            }
            // Mirror the decoder's predictor update.
            let has_coeffs = nonzero_count != 0;
            above.nonzero[a_slot] = has_coeffs;
            left.nonzero[l_slot] = has_coeffs;
        }

        fn finish(self) -> Vec<u8> {
            self.enc.finish()
        }
    }

    /// Encode a whole macroblock contiguously and return the byte stream.
    /// `above` / `left` are threaded and left holding the post-MB context.
    #[allow(clippy::too_many_arguments)]
    fn encode_mb_shared(
        has_y2: bool,
        coeff_probs: &CoeffProbs,
        above: &mut MbEntropyCtx,
        left: &mut MbEntropyCtx,
        y2_raster: &[i16; 16],
        y_raster: &[[i16; 16]; 16],
        u_raster: &[[i16; 16]; 4],
        v_raster: &[[i16; 16]; 4],
    ) -> Vec<u8> {
        let mut mb = MbEncoder::new();
        if has_y2 {
            mb.write_block(24, BlockType::Y2, y2_raster, coeff_probs, above, left);
        }
        let y_plane = if has_y2 {
            BlockType::YAfterY2
        } else {
            BlockType::YNoY2
        };
        for (i, b) in y_raster.iter().enumerate() {
            mb.write_block(i, y_plane, b, coeff_probs, above, left);
        }
        for (i, b) in u_raster.iter().enumerate() {
            mb.write_block(16 + i, BlockType::UV, b, coeff_probs, above, left);
        }
        for (i, b) in v_raster.iter().enumerate() {
            mb.write_block(20 + i, BlockType::UV, b, coeff_probs, above, left);
        }
        mb.finish()
    }

    #[test]
    fn zigzag_is_a_permutation_of_0_15() {
        // §20.16 zigzag[16] must be a bijection on 0..16 so
        // scan_to_raster / raster_to_scan round-trip.
        let mut seen = [false; 16];
        for &r in &ZIGZAG {
            assert!(!seen[r], "duplicate raster index {r}");
            seen[r] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // Round-trip a distinct-valued block.
        let raster: [i16; 16] = core::array::from_fn(|i| (i as i16) - 8);
        assert_eq!(scan_to_raster(&raster_to_scan(&raster)), raster);
    }

    #[test]
    fn context_index_tables_match_section_20_16() {
        // Spot-check the §20.16 left/above slot tables against the RFC
        // annex listing, including the Y2 (block 24) slot.
        assert_eq!(LEFT_CONTEXT_INDEX[0], 0);
        assert_eq!(LEFT_CONTEXT_INDEX[4], 1);
        assert_eq!(LEFT_CONTEXT_INDEX[15], 3);
        assert_eq!(LEFT_CONTEXT_INDEX[16], 4); // first U
        assert_eq!(LEFT_CONTEXT_INDEX[20], 6); // first V
        assert_eq!(LEFT_CONTEXT_INDEX[24], 8); // Y2
        assert_eq!(ABOVE_CONTEXT_INDEX[0], 0);
        assert_eq!(ABOVE_CONTEXT_INDEX[4], 0);
        assert_eq!(ABOVE_CONTEXT_INDEX[5], 1);
        assert_eq!(ABOVE_CONTEXT_INDEX[16], 4);
        assert_eq!(ABOVE_CONTEXT_INDEX[24], 8);
    }

    #[test]
    fn mb_skip_yields_zero_coeffs_and_resets_context() {
        // A skip MB reads no tokens and zeroes its Y/U/V predictor slots.
        // With has_y2 = true the Y2 slot is also cleared.
        let probs = &DEFAULT_COEFF_PROBS;
        let mut above = MbEntropyCtx {
            nonzero: [true; MB_ENTROPY_CTX_LEN],
        };
        let mut left = MbEntropyCtx {
            nonzero: [true; MB_ENTROPY_CTX_LEN],
        };
        // No bytes are consumed; a trivial buffer suffices.
        let buf = [0u8; 8];
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mb = decode_mb_coeffs(&mut dec, true, true, probs, &mut above, &mut left).unwrap();
        assert_eq!(mb.y2, [0i16; 16]);
        assert_eq!(mb.y, [[0i16; 16]; 16]);
        assert_eq!(mb.u, [[0i16; 16]; 4]);
        assert_eq!(mb.v, [[0i16; 16]; 4]);
        assert_eq!(above.nonzero, [false; MB_ENTROPY_CTX_LEN]);
        assert_eq!(left.nonzero, [false; MB_ENTROPY_CTX_LEN]);
    }

    #[test]
    fn mb_skip_bpred_preserves_y2_context() {
        // §20.16 reset_mb_context: a skipped B_PRED MB (no Y2) clears the
        // eight Y/U/V slots but preserves the Y2 slot (8).
        let probs = &DEFAULT_COEFF_PROBS;
        let mut above = MbEntropyCtx {
            nonzero: [true; MB_ENTROPY_CTX_LEN],
        };
        let mut left = MbEntropyCtx {
            nonzero: [true; MB_ENTROPY_CTX_LEN],
        };
        let buf = [0u8; 8];
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let _ = decode_mb_coeffs(&mut dec, false, true, probs, &mut above, &mut left).unwrap();
        for slot in 0..8 {
            assert!(!above.nonzero[slot], "above slot {slot} not cleared");
            assert!(!left.nonzero[slot], "left slot {slot} not cleared");
        }
        assert!(above.nonzero[8], "above Y2 slot must be preserved");
        assert!(left.nonzero[8], "left Y2 slot must be preserved");
    }

    #[test]
    fn synthetic_mb_decodes_to_expected_per_block_layout() {
        // Build a macroblock with distinctive coefficients in each plane,
        // encode it contiguously, and verify decode_mb_coeffs recovers the
        // exact per-block raster layout — including the Y2 block, the
        // YAfterY2 first-coefficient-1 luma plane, and chroma.
        let probs = &DEFAULT_COEFF_PROBS;

        let mut y2 = [0i16; 16];
        y2[0] = 9;
        y2[3] = -4;
        y2[15] = 2;

        // Each Y block: a recognisable value at raster position 1 (AC,
        // since the YAfterY2 plane starts decoding at scan coefficient 1).
        // Raster position 0 (the DC) must stay 0 — it is owned by Y2 and
        // never decoded for the YAfterY2 plane.
        let mut y = [[0i16; 16]; 16];
        for (k, blk) in y.iter_mut().enumerate() {
            blk[1] = (k as i16) - 8; // distinct per block, some negative
            if k % 3 == 0 {
                blk[5] = 3;
            }
        }

        let mut u = [[0i16; 16]; 4];
        let mut v = [[0i16; 16]; 4];
        for (k, blk) in u.iter_mut().enumerate() {
            blk[0] = (k as i16) + 1; // DC present (UV plane starts at 0)
        }
        for (k, blk) in v.iter_mut().enumerate() {
            blk[0] = -((k as i16) + 1);
            blk[2] = 7;
        }

        let mut enc_above = MbEntropyCtx::default();
        let mut enc_left = MbEntropyCtx::default();
        let buf = encode_mb_shared(true, probs, &mut enc_above, &mut enc_left, &y2, &y, &u, &v);

        let mut dec_above = MbEntropyCtx::default();
        let mut dec_left = MbEntropyCtx::default();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mb =
            decode_mb_coeffs(&mut dec, true, false, probs, &mut dec_above, &mut dec_left).unwrap();

        assert_eq!(mb.y2, y2, "Y2 block mismatch");
        assert_eq!(mb.y, y, "Y blocks mismatch");
        assert_eq!(mb.u, u, "U blocks mismatch");
        assert_eq!(mb.v, v, "V blocks mismatch");

        // The decoder's post-MB context must equal the encoder's.
        assert_eq!(dec_above, enc_above, "above context drift");
        assert_eq!(dec_left, enc_left, "left context drift");

        // Sanity on the resulting context: Y2 had coeffs (slot 8 set);
        // every Y column/row slot set (each Y block has a nonzero AC);
        // U/V slots set.
        assert!(dec_above.nonzero[8], "Y2 above predictor should be set");
        for slot in 0..8 {
            assert!(dec_above.nonzero[slot], "above slot {slot} should be set");
        }
    }

    #[test]
    fn empty_block_clears_its_predictor_slot() {
        // A block that decodes to all-zero (immediate EOB) must clear its
        // predictor slot even if it was set on entry, so the slot reflects
        // the LAST block mapped to it. We encode a Y MB where Y block 0
        // (slots above=0, left=0) is empty while Y block 1 (above=1,
        // left=0) is non-empty, then assert the resulting context.
        let probs = &DEFAULT_COEFF_PROBS;
        let y2 = [0i16; 16];
        let mut y = [[0i16; 16]; 16];
        // Y block 0 empty; Y block 1 has an AC coeff. Leave the rest empty.
        y[1][1] = 5;
        let u = [[0i16; 16]; 4];
        let v = [[0i16; 16]; 4];

        let mut enc_above = MbEntropyCtx::default();
        let mut enc_left = MbEntropyCtx::default();
        let buf = encode_mb_shared(true, probs, &mut enc_above, &mut enc_left, &y2, &y, &u, &v);
        let mut dec_above = MbEntropyCtx::default();
        let mut dec_left = MbEntropyCtx::default();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mb =
            decode_mb_coeffs(&mut dec, true, false, probs, &mut dec_above, &mut dec_left).unwrap();
        assert_eq!(mb.y, y);
        // Above slot for Y column 0 (blocks 0,4,8,12): the last decoded
        // block touching above-slot 0 is Y block 12 (empty) → false.
        assert!(
            !dec_above.nonzero[0],
            "above slot 0 should be cleared by later empty block"
        );
        // Above slot for Y column 1 (blocks 1,5,9,13): last is block 13
        // (empty) → false; but block 1 was non-empty earlier — confirm the
        // slot reflects the *last* block, not the earlier non-empty one.
        assert!(
            !dec_above.nonzero[1],
            "above slot 1 must reflect last (empty) block"
        );
        assert_eq!(dec_above, enc_above);
        assert_eq!(dec_left, enc_left);
    }

    #[test]
    fn two_adjacent_mbs_propagate_left_context() {
        // §13.3: the left predictor of MB(col=1) is the right column of
        // MB(col=0). Decode two horizontally-adjacent MBs sharing a single
        // rolling `left` context (and independent `above` rows), and prove
        // the second MB's first-coefficient context depends on the first
        // MB's right-edge blocks.
        //
        // We construct MB0 so that its right-column Y blocks (3,7,11,15 ->
        // left slot 3) and right-column U/V are non-empty, then verify the
        // identical MB1 token bytes decode correctly only when the carried
        // `left` context is used. The strongest proof: decode MB1 with the
        // propagated context vs a fresh (all-false) context and show the
        // recovered coefficients differ (a context mismatch desyncs the
        // bool decoder, corrupting the stream).
        let probs = &DEFAULT_COEFF_PROBS;

        // MB0: give every block a non-empty residue so all left slots end
        // up set (especially the right-column-driven slot 3, and U/V).
        let y2_0 = {
            let mut b = [0i16; 16];
            b[0] = 2;
            b
        };
        let mut y0 = [[0i16; 16]; 16];
        for blk in y0.iter_mut() {
            blk[1] = 1;
        }
        let mut u0 = [[0i16; 16]; 4];
        let mut v0 = [[0i16; 16]; 4];
        for blk in u0.iter_mut() {
            blk[0] = 1;
        }
        for blk in v0.iter_mut() {
            blk[0] = 1;
        }

        // MB1: distinctive coefficients so we can assert exact recovery.
        let y2_1 = {
            let mut b = [0i16; 16];
            b[0] = -5;
            b[1] = 2;
            b
        };
        let mut y1 = [[0i16; 16]; 16];
        for (k, blk) in y1.iter_mut().enumerate() {
            blk[1] = (k as i16) - 5;
        }
        let mut u1 = [[0i16; 16]; 4];
        let mut v1 = [[0i16; 16]; 4];
        for (k, blk) in u1.iter_mut().enumerate() {
            blk[0] = (k as i16) + 2;
        }
        for (k, blk) in v1.iter_mut().enumerate() {
            blk[0] = -((k as i16) + 2);
        }

        // Encode both MBs with the shared rolling `left` and a per-column
        // `above` row of two contexts (mirroring a 2-wide MB row).
        let mut enc_left = MbEntropyCtx::default();
        let mut enc_above0 = MbEntropyCtx::default();
        let mut enc_above1 = MbEntropyCtx::default();
        let buf0 = encode_mb_shared(
            true,
            probs,
            &mut enc_above0,
            &mut enc_left,
            &y2_0,
            &y0,
            &u0,
            &v0,
        );
        // After MB0, `enc_left` holds MB0's right-edge context — exactly
        // what MB1 must consume. Capture it for the negative control.
        let left_after_mb0 = enc_left;
        let buf1 = encode_mb_shared(
            true,
            probs,
            &mut enc_above1,
            &mut enc_left,
            &y2_1,
            &y1,
            &u1,
            &v1,
        );

        // Decode MB0, then MB1 with the propagated `left`.
        let mut dec_left = MbEntropyCtx::default();
        let mut dec_above0 = MbEntropyCtx::default();
        let mut dec_above1 = MbEntropyCtx::default();
        let mut dec0 = BoolDecoder::init(&buf0).unwrap();
        let mb0 = decode_mb_coeffs(
            &mut dec0,
            true,
            false,
            probs,
            &mut dec_above0,
            &mut dec_left,
        )
        .unwrap();
        assert_eq!(mb0.y, y0);
        assert_eq!(
            dec_left, left_after_mb0,
            "MB0 must yield MB1's left context"
        );

        let mut dec1 = BoolDecoder::init(&buf1).unwrap();
        let mb1 = decode_mb_coeffs(
            &mut dec1,
            true,
            false,
            probs,
            &mut dec_above1,
            &mut dec_left,
        )
        .unwrap();
        assert_eq!(mb1.y2, y2_1, "MB1 Y2 with propagated left context");
        assert_eq!(mb1.y, y1, "MB1 Y with propagated left context");
        assert_eq!(mb1.u, u1, "MB1 U with propagated left context");
        assert_eq!(mb1.v, v1, "MB1 V with propagated left context");

        // Negative control: decoding MB1 with a FRESH (all-false) left
        // context picks the wrong probability rows on the first
        // coefficient of every block, desyncing the range decoder. The
        // recovered coefficients must NOT match the encoded MB1 — proving
        // the propagation is load-bearing, not incidental.
        let mut wrong_left = MbEntropyCtx::default();
        let mut wrong_above = MbEntropyCtx::default();
        let mut dec_wrong = BoolDecoder::init(&buf1).unwrap();
        let mb_wrong = decode_mb_coeffs(
            &mut dec_wrong,
            true,
            false,
            probs,
            &mut wrong_above,
            &mut wrong_left,
        );
        // It may decode (to garbage) or error; either way it must not
        // reproduce the correct MB1 coefficients.
        if let Ok(garbage) = mb_wrong {
            assert!(
                garbage.y != y1 || garbage.y2 != y2_1 || garbage.u != u1 || garbage.v != v1,
                "fresh-left decode unexpectedly matched — propagation not load-bearing?"
            );
        }
    }

    #[test]
    fn fused_dequant_walk_matches_decode_then_dequantize() {
        // The fused §13.3 decode → §14.1 dequant walk
        // (`decode_mb_coeffs_dequant`, the `DQ = true` monomorphisation)
        // must be stream-for-stream identical to the two-pass form —
        // `decode_mb_coeffs` followed by
        // `MbDequantFactors::dequantize` — in every §13 shape:
        //
        // * a dense MB (every lane of every block non-zero, including
        //   cat6 magnitudes whose product with a large factor overflows
        //   i16 and must truncate identically);
        // * a sparse MB (DC + a few low-frequency AC, long zero runs —
        //   the common shape, exercising the DCT_0 re-entry loop and
        //   the untouched-lane invariant);
        // * a no-Y2 (B_PRED-shaped) MB where luma starts at coefficient
        //   0 and carries its own DC;
        // * a skip MB (no tokens; both paths return zeros and apply the
        //   same context reset).
        //
        // The two decoders walk physically separate copies of the same
        // byte stream with independently threaded above/left contexts;
        // afterwards the coefficients, both contexts, and the number of
        // consumed input bits must all agree (the trailing literal read
        // proves bit-position lockstep).
        use crate::coded_header::QuantIndices;
        use crate::dequant::MbDequantFactors;

        let probs = &DEFAULT_COEFF_PROBS;

        // Large factors (high qi + worst-case deltas) so the cat6
        // products genuinely overflow i16 and pin the truncation.
        let factors = MbDequantFactors::from_quant_indices(&QuantIndices {
            y_ac_qi: 120,
            y_dc_delta: Some(7),
            y2_dc_delta: Some(15),
            y2_ac_delta: Some(15),
            uv_dc_delta: Some(3),
            uv_ac_delta: Some(9),
        });
        let pairs = MbDequantPairs {
            y2: (factors.y2_dc, factors.y2_ac),
            y1: (factors.y1_dc, factors.y1_ac),
            uv: (factors.uv_dc, factors.uv_ac),
        };

        // Dense MB: every lane non-zero, cat6 values in every plane.
        let dense_block: [i16; 16] = [
            900, -3, 1, 70, //
            2, -80, 5, 1, //
            -1, 4, -2114, 2, //
            3, -1, 1, -6,
        ];
        let dense_y2 = dense_block;
        let dense_y = [dense_block; 16];
        let dense_uv = [dense_block; 4];

        // Sparse MB: DC + two low-frequency ACs, long zero runs.
        let sparse_block: [i16; 16] = [
            7, -1, 0, 0, //
            1, 0, 0, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0,
        ];
        let sparse_y2 = sparse_block;
        let sparse_y = [sparse_block; 16];
        let sparse_uv = [sparse_block; 4];

        // (has_y2, skip, y2, y, u, v) shapes.
        let zero_y2 = [0i16; 16];
        #[allow(clippy::type_complexity)]
        let shapes: [(bool, bool, [i16; 16], [[i16; 16]; 16], [[i16; 16]; 4]); 4] = [
            (true, false, dense_y2, dense_y, dense_uv),
            (true, false, sparse_y2, sparse_y, sparse_uv),
            (false, false, zero_y2, sparse_y, dense_uv),
            (true, true, zero_y2, [[0i16; 16]; 16], [[0i16; 16]; 4]),
        ];

        for (case, &(has_y2, skip, y2, y, uv)) in shapes.iter().enumerate() {
            // Encode the MB (skip MBs contribute no tokens; encode an
            // empty stream and let both paths take the skip branch).
            let mut bytes = if skip {
                vec![0u8; 8]
            } else {
                let mut enc_above = MbEntropyCtx::default();
                let mut enc_left = MbEntropyCtx::default();
                encode_mb_shared(
                    has_y2,
                    probs,
                    &mut enc_above,
                    &mut enc_left,
                    &y2,
                    &y,
                    &uv,
                    &uv,
                )
            };
            // Slack for the trailing lockstep literal read below.
            bytes.extend_from_slice(&[0xA5; 8]);

            // Path A: two-pass — decode raw, then dequantize.
            let mut dec_a = BoolDecoder::init(&bytes).unwrap();
            let mut above_a = MbEntropyCtx::default();
            let mut left_a = MbEntropyCtx::default();
            let mut two_pass =
                decode_mb_coeffs(&mut dec_a, has_y2, skip, probs, &mut above_a, &mut left_a)
                    .unwrap();
            factors.dequantize(&mut two_pass);

            // Path B: the fused walk.
            let mut dec_b = BoolDecoder::init(&bytes).unwrap();
            let mut above_b = MbEntropyCtx::default();
            let mut left_b = MbEntropyCtx::default();
            let fused = decode_mb_coeffs_dequant(
                &mut dec_b,
                has_y2,
                skip,
                probs,
                &pairs,
                &mut above_b,
                &mut left_b,
            )
            .unwrap();

            assert_eq!(fused, two_pass, "case {case}: coefficients diverge");
            assert_eq!(above_a, above_b, "case {case}: above ctx diverges");
            assert_eq!(left_a, left_b, "case {case}: left ctx diverges");
            // Bit-position lockstep: the next reads must agree.
            assert_eq!(
                dec_a.read_literal(16).unwrap(),
                dec_b.read_literal(16).unwrap(),
                "case {case}: consumed bit counts diverge"
            );
        }
    }
}
