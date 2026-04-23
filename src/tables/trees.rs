//! Tree-coded helpers and intra-prediction-mode trees.
//!
//! VP8 encodes most non-coefficient symbols using the "tree" form of its
//! boolean decoder (RFC 6386 §8). A tree is a flat `&[i8]` where each pair
//! `(L, R)` describes a branch. Negative values are literal symbol values
//! (negated), positive values are byte offsets to the next pair within the
//! same tree.
//!
//! `decode_tree` walks the tree starting from index `start_index` (almost
//! always `0`), reading one boolean per branch using the corresponding
//! probability from `probs`.

use crate::bool_decoder::BoolDecoder;

pub fn decode_tree(d: &mut BoolDecoder<'_>, tree: &[i8], probs: &[u8]) -> i32 {
    let mut idx = 0usize;
    loop {
        let prob = probs[idx >> 1] as u32;
        let bit = d.read_bool(prob) as usize;
        let v = tree[idx + bit];
        if v <= 0 {
            return -(v as i32);
        }
        idx = v as usize;
    }
}

// --- Intra prediction mode constants -------------------------------------

// Luma intra-16×16 modes (§16.1).
pub const DC_PRED: i32 = 0;
pub const V_PRED: i32 = 1;
pub const H_PRED: i32 = 2;
pub const TM_PRED: i32 = 3;
pub const B_PRED: i32 = 4;

// Intra 4x4 sub-block modes (§16.2).
pub const B_DC_PRED: i32 = 0;
pub const B_TM_PRED: i32 = 1;
pub const B_VE_PRED: i32 = 2;
pub const B_HE_PRED: i32 = 3;
pub const B_LD_PRED: i32 = 4;
pub const B_RD_PRED: i32 = 5;
pub const B_VR_PRED: i32 = 6;
pub const B_VL_PRED: i32 = 7;
pub const B_HD_PRED: i32 = 8;
pub const B_HU_PRED: i32 = 9;

// Tree for keyframe luma intra-16×16 modes (RFC 6386 §11.2 kf_ymode_tree).
pub const KF_YMODE_TREE: [i8; 8] = [
    -B_PRED as i8,
    2,
    4,
    6,
    -DC_PRED as i8,
    -V_PRED as i8,
    -H_PRED as i8,
    -TM_PRED as i8,
];

// Probabilities used by KF_YMODE_TREE for keyframes (§11.2).
pub const KF_YMODE_PROBS: [u8; 4] = [145, 156, 163, 128];

// Tree for keyframe chroma intra-8×8 modes (§11.2 kf_uv_mode_tree).
pub const KF_UV_MODE_TREE: [i8; 6] = [
    -DC_PRED as i8,
    2,
    -V_PRED as i8,
    4,
    -H_PRED as i8,
    -TM_PRED as i8,
];
pub const KF_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];

// --- Inter-MB prediction modes (§16.3) ----------------------------------

// Inter-MB Y prediction mode values. These occupy a separate enum from the
// intra modes and drive MV derivation.
pub const NEAREST_MV: i32 = 10;
pub const NEAR_MV: i32 = 11;
pub const ZERO_MV: i32 = 12;
pub const NEW_MV: i32 = 13;
pub const SPLIT_MV: i32 = 14;

// Sub-MB split-mode selector (§16.3, `mb_split_tree`).
pub const MB_SPLIT_16X8: i32 = 0;
pub const MB_SPLIT_8X16: i32 = 1;
pub const MB_SPLIT_QUARTERS: i32 = 2;
pub const MB_SPLIT_4X4: i32 = 3;

// Sub-MV modes inside a split MB (§16.3, `sub_mv_ref_tree`).
pub const LEFT_4X4: i32 = 0;
pub const ABOVE_4X4: i32 = 1;
pub const ZERO_4X4: i32 = 2;
pub const NEW_4X4: i32 = 3;

/// Inter MB Y-mode tree (RFC §16.3). Leaf 0/1/2/3 corresponds to intra modes
/// decoded via a second tree below (DC/V/H/TM/B), leaves 4/5/6/7/8 are the
/// inter modes NEAREST/NEAR/ZERO/NEW/SPLIT.
pub const YMODE_TREE: [i8; 8] = [
    -DC_PRED as i8,
    2,
    4,
    6,
    -V_PRED as i8,
    -H_PRED as i8,
    -TM_PRED as i8,
    -B_PRED as i8,
];

/// Default (inter) luma Y mode probabilities. Will be updated from the
/// frame header (vp8_kf_default_ymode_probs replacement).
pub const DEFAULT_YMODE_PROBS: [u8; 4] = [112, 86, 140, 37];

pub const UV_MODE_TREE: [i8; 6] = [
    -DC_PRED as i8,
    2,
    -V_PRED as i8,
    4,
    -H_PRED as i8,
    -TM_PRED as i8,
];
pub const DEFAULT_UV_MODE_PROBS: [u8; 3] = [162, 101, 204];

/// MV-reference (inter-MB macro) tree — RFC 6386 §16.3 `vp8_mv_ref_tree`.
pub const MV_REF_TREE: [i8; 8] = [
    -(ZERO_MV - 10) as i8, // leaf 0 → ZERO_MV
    2,
    -(NEAREST_MV - 10) as i8, // leaf 1 → NEAREST_MV
    4,
    -(NEAR_MV - 10) as i8, // leaf 2 → NEAR_MV
    6,
    -(NEW_MV - 10) as i8,   // leaf 3 → NEW_MV
    -(SPLIT_MV - 10) as i8, // leaf 4 → SPLIT_MV
];

/// MV split mode tree (§16.3 `mb_split_tree`).
pub const MB_SPLIT_TREE: [i8; 6] = [
    -MB_SPLIT_4X4 as i8,
    2,
    -MB_SPLIT_16X8 as i8,
    4,
    -MB_SPLIT_8X16 as i8,
    -MB_SPLIT_QUARTERS as i8,
];

/// Sub-MV reference tree — RFC 6386 §16.3 `sub_mv_ref_tree`.
pub const SUB_MV_REF_TREE: [i8; 6] = [
    -LEFT_4X4 as i8,
    2,
    -ABOVE_4X4 as i8,
    4,
    -ZERO_4X4 as i8,
    -NEW_4X4 as i8,
];

/// MV ref context probabilities. 4 probabilities × 6 contexts — RFC 6386 §16.3
/// `mv_counts_to_probs`. Each row is indexed by the corresponding cnt[i],
/// NOT by cnt[0] — `probs[i] = mv_counts_to_probs[cnt[i]][i]`.
pub const MV_COUNTS_TO_PROBS: [[u8; 4]; 6] = [
    [7, 1, 1, 143],
    [14, 18, 14, 107],
    [135, 64, 57, 68],
    [60, 56, 128, 65],
    [159, 134, 128, 34],
    [234, 188, 128, 28],
];

/// MB split probabilities (RFC 6386 §16.3 `split_mv_probs`).
pub const MBSPLIT_PROBS: [u8; 3] = [110, 111, 150];

/// Sub-MV reference probabilities given neighbour MVs.
///   Row 0: left == above == 0      → probs for [LEFT_4X4, ABOVE_4X4, ZERO_4X4]
///   Row 1: left != 0, above == 0
///   Row 2: left == 0, above != 0
///   Row 3: left != above (both non-zero)
///   Row 4: left == above (both non-zero)
pub const SUB_MV_REF_PROBS: [[u8; 3]; 5] = [
    [147, 136, 18],
    [106, 145, 1],
    [179, 121, 1],
    [223, 1, 34],
    [208, 1, 1],
];

/// Splitting patterns for each SPLIT MV mode. Index 0..=15 maps each
/// 4×4 Y sub-block to a partition number (0..=3).
pub const MB_SPLITS: [[u8; 16]; 4] = [
    // 16x8 (two 16×8 pieces — top/bottom)
    [
        0, 0, 0, 0, //
        0, 0, 0, 0, //
        1, 1, 1, 1, //
        1, 1, 1, 1,
    ],
    // 8x16 (two 8×16 pieces — left/right)
    [
        0, 0, 1, 1, //
        0, 0, 1, 1, //
        0, 0, 1, 1, //
        0, 0, 1, 1,
    ],
    // Quarters — four 8×8.
    [
        0, 0, 1, 1, //
        0, 0, 1, 1, //
        2, 2, 3, 3, //
        2, 2, 3, 3,
    ],
    // 16 quarters (every 4×4 its own partition).
    [
        0, 1, 2, 3, //
        4, 5, 6, 7, //
        8, 9, 10, 11, //
        12, 13, 14, 15,
    ],
];

/// Number of partitions for each split mode.
pub const MB_SPLIT_COUNT: [u8; 4] = [2, 2, 4, 16];

// Tree for intra-4×4 mode (§11.5 b_mode_tree).
pub const BMODE_TREE: [i8; 18] = [
    -B_DC_PRED as i8,
    2,
    -B_TM_PRED as i8,
    4,
    -B_VE_PRED as i8,
    6,
    8,
    12,
    -B_HE_PRED as i8,
    10,
    -B_RD_PRED as i8,
    -B_VR_PRED as i8,
    -B_LD_PRED as i8,
    14,
    -B_VL_PRED as i8,
    16,
    -B_HD_PRED as i8,
    -B_HU_PRED as i8,
];

// `kf_bmode_prob[A][L][i]` from RFC 6386 §11.5 — keyframe context-sensitive
// probabilities for the b_mode tree, indexed by [above mode][left mode][branch].
pub const KF_BMODE_PROB: [[[u8; 9]; 10]; 10] = [
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
    #[test]
    fn tree_lengths_sane() {
        assert_eq!(KF_YMODE_TREE.len(), 8);
        assert_eq!(KF_UV_MODE_TREE.len(), 6);
        assert_eq!(BMODE_TREE.len(), 18);
    }
}
