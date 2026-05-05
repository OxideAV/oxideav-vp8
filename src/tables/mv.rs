//! Motion vector probabilities and sub-pel filter taps — RFC 6386 §17 / §18.
//!
//! The MV component probabilities come from the reference implementation in
//! RFC 6386 §17.1 (`default_mv_context`). Each component has 19 entries:
//!   0    — IS_SHORT (short vs long)
//!   1    — SIGN
//!   2..=8  short-magnitude tree (8 probs, 9 branches of a short tree)
//!   9..=18 long-magnitude bits (one per bit position, probs for bits 0..9 with
//!          bit 3 handled last per RFC)
//!
//! We represent the 19 entries directly.
//!
//! The sub-pel filter taps come from RFC 6386 §18 `bilinear_filters` (2-tap)
//! and `sixtap_filters` (6-tap) tables.

/// MV probability context — 19 entries per component (x then y).
pub type MvContext = [u8; 19];

/// Indices into [`MvContext`].
pub mod mv_ctx {
    pub const IS_SHORT: usize = 0;
    pub const SIGN: usize = 1;
    pub const SHORT_TREE: usize = 2; // 7 entries: [2..9]
    pub const BITS: usize = 9; // 10 entries: [9..19]
    pub const LONG_WIDTH: usize = 10; // bit positions 0..=9, bit 3 is last
}

/// Default MV probability context from RFC 6386 §17.1 `vp8_default_mv_context`
/// (page 110-111). Indexed as `[component][entry]` where component 0 is row
/// (y) and 1 is col (x). Each component is 19 entries:
///
/// * `[0]` IS_SHORT, `[1]` SIGN, `[2..=8]` 7-entry short tree, `[9..=18]`
///   10-entry long-bit probabilities for bits 0..9 (bit 3 handled last).
///
/// Earlier versions of this table had the trailing three long-bit entries
/// for bits 7/8/9 (`[16]`/`[17]`/`[18]`) populated with low values
/// (`145/162/163` for row, `166/172/182` for col). The RFC values are
/// `239/254/254` (row) and `236/254/254` (col); the low values caused
/// inter MV components in the long-magnitude path to receive incorrect
/// high-bit probabilities (P(bit=0) ≈ 57-64% instead of 93-99%), so
/// any MV with magnitude ≥ 128 (long-path) sub-decoded its top bits as
/// near-50/50, producing wildly wrong magnitudes that landed exactly on
/// the §16.3 `vp8_clamp_mv` lower bound. Visible in
/// `small-roi-segmentation/trace.txt` MB(4,0) frame 1: our pre-fix
/// decode produced mv=(0, -640), the row's `mb_to_left_edge - MV_BORDER`
/// clamp boundary; with the corrected probs the same MB decodes a
/// short-path zero-magnitude MV.
pub const DEFAULT_MV_CONTEXT: [MvContext; 2] = [
    // Row / y component.
    [
        162, 128, 225, 146, 172, 147, 214, 39, 156, 128, 129, 132, 75, 145, 178, 206, 239, 254, 254,
    ],
    // Col / x component.
    [
        164, 128, 204, 170, 119, 235, 140, 230, 228, 128, 130, 130, 74, 148, 180, 203, 236, 254,
        254,
    ],
];

/// Update probabilities for MV probabilities in the frame header — RFC 6386
/// §17.2 `vp8_mv_update_probs`.
pub const MV_UPDATE_PROBS: [MvContext; 2] = [
    [
        237, 246, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 250, 250, 252, 254,
        254,
    ],
    [
        231, 243, 245, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 251, 251, 254, 254,
        254,
    ],
];

/// MV short-magnitude tree — RFC 6386 §17.1 `vp8_small_mvtree`. Decodes
/// a 3-bit magnitude in range 0..=7.
pub const MV_SHORT_TREE: [i8; 14] = [
    2, 8, //
    4, 6, //
    -0, -1, //
    -2, -3, //
    10, 12, //
    -4, -5, //
    -6, -7, //
];

/// Sub-pixel filter taps — RFC 6386 §18.3 `sixtap_filters`. Used when
/// reconstructing inter-predicted luma. Taps are [−2, −1, 0, +1, +2, +3]
/// relative to the integer sample position; each row is indexed by the
/// fractional offset in eighth-pel units (0..=7).
pub const SIXTAP_FILTERS: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, -6, 123, 12, -1, 0],
    [2, -11, 108, 36, -8, 1],
    [0, -9, 93, 50, -6, 0],
    [3, -16, 77, 77, -16, 3],
    [0, -6, 50, 93, -9, 0],
    [1, -8, 36, 108, -11, 2],
    [0, -1, 12, 123, -6, 0],
];

/// Bilinear sub-pel taps (2-tap) — used for chroma and as a low-complexity
/// fallback. RFC 6386 §18.3 `bilinear_filters`.
pub const BILINEAR_FILTERS: [[i32; 2]; 8] = [
    [128, 0],
    [112, 16],
    [96, 32],
    [80, 48],
    [64, 64],
    [48, 80],
    [32, 96],
    [16, 112],
];
