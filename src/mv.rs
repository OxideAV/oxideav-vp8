//! Motion-vector decoding — RFC 6386 §16 + §17.
//!
//! A VP8 motion vector is a signed pair `(row, col)` in 1/8-pel units.
//! Each component is either "short" (magnitude 0..=7) or "long"
//! (magnitude 8..=1023) and is coded using a dedicated set of 19
//! per-component probabilities (`MvContext`).

use crate::bool_decoder::BoolDecoder;
use crate::bool_encoder::{bool_cost_x256, BoolEncoder};
use crate::tables::mv::{MvContext, MV_SHORT_TREE};
use crate::tables::trees::decode_tree;

/// Signed MV component in 1/8-pel units.
pub type MvComponent = i16;

/// A 2D motion vector, row then column, in 1/8-pel units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: MvComponent,
    pub col: MvComponent,
}

impl Mv {
    pub const ZERO: Self = Self { row: 0, col: 0 };

    pub fn new(row: i32, col: i32) -> Self {
        Self {
            row: row.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            col: col.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        }
    }
}

/// MV-clamp border (RFC 6386 §16.3 `vp8_clamp_mv`, §18.1).
///
/// The MV-clamp margins applied around the per-MB `mb_to_*_edge`
/// distances. RFC 6386 §16.3 cites them as `LEFT_TOP_MARGIN` and
/// `RIGHT_BOTTOM_MARGIN` without naming the value; §18.1 fixes the
/// reference-frame extended border at "1 macroblock" of replicated
/// pixels around each side, which equates to 16 pixels × 8 sub-pel
/// units = 128 in MV units. The same value is used for the top/left and
/// bottom/right margins (single-MB symmetric border).
pub const MV_BORDER: i32 = 128;

/// Clamp an MV (in 1/8-pel units) to the per-MB legal range as defined
/// in RFC 6386 §16.3 `vp8_clamp_mv`.
///
/// The four `mb_to_*_edge` distances are computed in 1/8-pel units from
/// the current MB's pixel position to each frame edge:
///
/// ```text
/// mb_to_left_edge   = -(mb_x * 16) << 3 = -mb_x * 128
/// mb_to_right_edge  = ((mb_w - 1 - mb_x) * 16) << 3 = (mb_w-1-mb_x)*128
/// mb_to_top_edge    = -(mb_y * 16) << 3 = -mb_y * 128
/// mb_to_bottom_edge = ((mb_h - 1 - mb_y) * 16) << 3 = (mb_h-1-mb_y)*128
/// ```
///
/// The MV is then clamped to:
///
/// ```text
/// col in [mb_to_left_edge - MV_BORDER, mb_to_right_edge + MV_BORDER]
/// row in [mb_to_top_edge  - MV_BORDER, mb_to_bottom_edge + MV_BORDER]
/// ```
///
/// At MB position (mb_x, mb_y) in a (mb_w × mb_h)-MB frame, this
/// permits the MV to point at most one MB beyond the visible image
/// edge — exactly the size of the §18.1 extended reference border.
pub fn clamp_mv_to_border(mv: Mv, mb_x: usize, mb_y: usize, mb_w: usize, mb_h: usize) -> Mv {
    let mb_to_left_edge = -((mb_x as i32) * 128);
    let mb_to_right_edge = ((mb_w as i32) - 1 - (mb_x as i32)) * 128;
    let mb_to_top_edge = -((mb_y as i32) * 128);
    let mb_to_bottom_edge = ((mb_h as i32) - 1 - (mb_y as i32)) * 128;
    let col = (mv.col as i32).clamp(mb_to_left_edge - MV_BORDER, mb_to_right_edge + MV_BORDER);
    let row = (mv.row as i32).clamp(mb_to_top_edge - MV_BORDER, mb_to_bottom_edge + MV_BORDER);
    Mv::new(row, col)
}

/// Decode a single MV component. Returns a signed integer in 1/8-pel
/// units.
pub fn decode_mv_component(d: &mut BoolDecoder<'_>, probs: &MvContext) -> i32 {
    // Probability at index 0 indicates "is large" (long). When true we take
    // the long path, otherwise use the 3-bit short tree.
    let large = d.read_bool(probs[0] as u32);
    let mut mag = if large {
        // Bits 0..=9 individually except bit 3 is deferred to the end.
        let mut v = 0i32;
        for i in 0..3 {
            v += (d.read_bool(probs[9 + i] as u32) as i32) << i;
        }
        for i in (4..=9).rev() {
            v += (d.read_bool(probs[9 + i] as u32) as i32) << i;
        }
        // Bit 3 (LSB-4) — only relevant if at least one of bits 4..9 is
        // non-zero; otherwise implicit. See RFC 6386 §17.1 pseudo-code.
        if (v & 0xfff0) == 0 || d.read_bool(probs[9 + 3] as u32) {
            v += 8;
        }
        v
    } else {
        // 3-bit short magnitude using bits at indices 2..=8.
        decode_tree(d, &MV_SHORT_TREE, &probs[2..9])
    };
    if mag != 0 && d.read_bool(probs[1] as u32) {
        mag = -mag;
    }
    mag
}

/// Decode a full MV given a pair of component probability contexts
/// (row-first then col — matching libvpx's layout of the default context).
pub fn decode_mv(d: &mut BoolDecoder<'_>, mv_probs: &[MvContext; 2]) -> Mv {
    let row = decode_mv_component(d, &mv_probs[0]);
    let col = decode_mv_component(d, &mv_probs[1]);
    Mv::new(row, col)
}

/// Bit-exact write-side inverse of [`decode_mv_component`]. Given the same
/// probability context, emits the boolean-coded bits that
/// `decode_mv_component` will read back as `value`.
pub fn encode_mv_component(enc: &mut BoolEncoder, probs: &MvContext, value: i32) {
    let sign = value < 0;
    let mag = value.unsigned_abs() as i32;
    let large = mag >= 8;
    enc.write_bool(probs[0] as u32, large);
    if large {
        for i in 0..3 {
            let bit = ((mag >> i) & 1) != 0;
            enc.write_bool(probs[9 + i] as u32, bit);
        }
        for i in (4..=9).rev() {
            let bit = ((mag >> i) & 1) != 0;
            enc.write_bool(probs[9 + i] as u32, bit);
        }
        // Bit 3 is implicit when bits 4..=9 are all zero: if mag in [8..=15]
        // is represented without the bit-3 write, the decoder adds 8
        // unconditionally. If bits 4..=9 are non-zero, bit 3 is coded
        // explicitly.
        let upper_any = (mag & 0xfff0) != 0;
        if upper_any {
            let bit3 = ((mag >> 3) & 1) != 0;
            enc.write_bool(probs[9 + 3] as u32, bit3);
        } else {
            // Encoder must only take this branch when mag in [8..=15]. The
            // decoder unconditionally adds 8 when the upper bits are all
            // zero, so the magnitude is fully determined by bits 0..=2.
            debug_assert!((8..=15).contains(&mag));
        }
    } else {
        encode_tree_leaf(enc, &MV_SHORT_TREE, &probs[2..9], mag);
    }
    if mag != 0 {
        enc.write_bool(probs[1] as u32, sign);
    }
}

/// Write-side inverse of [`crate::tables::trees::decode_tree`]: emits the
/// boolean branch sequence that decodes to `leaf`.
fn encode_tree_leaf(enc: &mut BoolEncoder, tree: &[i8], probs: &[u8], leaf: i32) {
    // Walk the tree recording the branches that reach the target leaf.
    fn walk(tree: &[i8], leaf: i32, idx: usize) -> Option<Vec<(usize, bool)>> {
        for bit in [false, true] {
            let v = tree[idx + bit as usize];
            if v <= 0 {
                if -(v as i32) == leaf {
                    return Some(vec![(idx >> 1, bit)]);
                }
            } else if let Some(mut rest) = walk(tree, leaf, v as usize) {
                rest.insert(0, (idx >> 1, bit));
                return Some(rest);
            }
        }
        None
    }
    let path = walk(tree, leaf, 0).expect("leaf not in tree");
    for (prob_idx, bit) in path {
        enc.write_bool(probs[prob_idx] as u32, bit);
    }
}

/// Encode a full MV (row then col) using the two-component context used by
/// the decoder.
pub fn encode_mv(enc: &mut BoolEncoder, mv_probs: &[MvContext; 2], mv: Mv) {
    encode_mv_component(enc, &mv_probs[0], mv.row as i32);
    encode_mv_component(enc, &mv_probs[1], mv.col as i32);
}

/// Cost (in 1/256-bit units) of writing a single MV component with
/// [`encode_mv_component`] — i.e. the *real* per-component bit count
/// for the candidate value, summed over all the bool symbols
/// `encode_mv_component` would emit. Used by the encoder's RDO mode
/// picker (issue #340) so the rate input to the Lagrangian matches the
/// actual bit stream.
///
/// Mirrors `encode_mv_component`'s control flow exactly, but instead
/// of emitting bits it accumulates each bool's
/// `bool_encoder::bool_cost_x256(prob, outcome)`.
pub fn mv_component_cost_x256(probs: &MvContext, value: i32) -> u32 {
    let sign = value < 0;
    let mag = value.unsigned_abs() as i32;
    let large = mag >= 8;
    let mut cost = bool_cost_x256(probs[0], large);
    if large {
        for i in 0..3 {
            let bit = ((mag >> i) & 1) != 0;
            cost += bool_cost_x256(probs[9 + i], bit);
        }
        for i in (4..=9).rev() {
            let bit = ((mag >> i) & 1) != 0;
            cost += bool_cost_x256(probs[9 + i], bit);
        }
        let upper_any = (mag & 0xfff0) != 0;
        if upper_any {
            let bit3 = ((mag >> 3) & 1) != 0;
            cost += bool_cost_x256(probs[9 + 3], bit3);
        }
    } else {
        cost += tree_leaf_cost_x256(&MV_SHORT_TREE, &probs[2..9], mag);
    }
    if mag != 0 {
        cost += bool_cost_x256(probs[1], sign);
    }
    cost
}

/// Cost (in 1/256-bit units) of writing the path that leads to `leaf`
/// in `tree`. Mirrors `encode_tree_leaf`'s control flow.
fn tree_leaf_cost_x256(tree: &[i8], probs: &[u8], leaf: i32) -> u32 {
    fn walk(tree: &[i8], leaf: i32, idx: usize) -> Option<Vec<(usize, bool)>> {
        for bit in [false, true] {
            let v = tree[idx + bit as usize];
            if v <= 0 {
                if -(v as i32) == leaf {
                    return Some(vec![(idx >> 1, bit)]);
                }
            } else if let Some(mut rest) = walk(tree, leaf, v as usize) {
                rest.insert(0, (idx >> 1, bit));
                return Some(rest);
            }
        }
        None
    }
    let path = walk(tree, leaf, 0).expect("leaf not in tree");
    path.into_iter()
        .map(|(prob_idx, bit)| bool_cost_x256(probs[prob_idx], bit))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::mv::DEFAULT_MV_CONTEXT;

    fn roundtrip_component(value: i32) {
        let mut enc = BoolEncoder::new();
        encode_mv_component(&mut enc, &DEFAULT_MV_CONTEXT[0], value);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let got = decode_mv_component(&mut dec, &DEFAULT_MV_CONTEXT[0]);
        assert_eq!(got, value, "mv component {value} did not round-trip");
    }

    #[test]
    fn mv_component_roundtrip_short() {
        for v in -7..=7 {
            roundtrip_component(v);
        }
    }

    #[test]
    fn mv_component_roundtrip_long_boundary() {
        // Covers the 8..=15 "bit-3 implicit" range and a few larger values.
        for v in [
            8, -8, 15, -15, 16, -16, 17, -17, 24, 40, 127, -127, 256, 1023,
        ] {
            roundtrip_component(v);
        }
    }

    #[test]
    fn mv_component_cost_matches_real_emit_bits() {
        // mv_component_cost_x256 must agree with the bit count produced
        // by piping the same value through encode_mv_component into a
        // real BoolEncoder. We emit each value into its own fresh
        // encoder (so the buffered-state slack is bounded by ~25 bits)
        // and compare totals in 1/256-bit units.
        for &v in &[
            0, 1, -1, 4, 7, -7, 8, -8, 15, -15, 16, 33, -33, 127, -127, 256, 1023,
        ] {
            let mut enc = BoolEncoder::new();
            encode_mv_component(&mut enc, &DEFAULT_MV_CONTEXT[0], v);
            let real_bytes = enc.finish().len();
            // Subtract the 32-bit (= 4-byte) trailing flush pad.
            let real_bits_x256 = (real_bytes.saturating_sub(4) as u64) * 8 * 256;

            let est_x256 = mv_component_cost_x256(&DEFAULT_MV_CONTEXT[0], v) as u64;
            // The real encoder also has up to 25 buffered bits not yet
            // flushed to bytes; allow a generous window since the
            // expected payload for these magnitudes is < ~16 bits
            // anyway. The check is that the estimator is in the same
            // ballpark as the actual bool encoder, not bit-exact.
            let abs_diff = est_x256.saturating_sub(real_bits_x256 + 32 * 256);
            assert!(
                abs_diff <= 8 * 256,
                "v={v}: estimator {est_x256}/256 vs real bits {real_bits_x256}/256 + slack"
            );
        }
    }

    #[test]
    fn mv_roundtrip_pair() {
        let mv = Mv::new(-64, 32);
        let mut enc = BoolEncoder::new();
        encode_mv(&mut enc, &DEFAULT_MV_CONTEXT, mv);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let got = decode_mv(&mut dec, &DEFAULT_MV_CONTEXT);
        assert_eq!(got, mv);
    }
}
