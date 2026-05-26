//! VP8 motion-vector component decoding — RFC 6386 §17.
//!
//! Motion vectors appear in two places in the VP8 datastream: applied to
//! whole macroblocks in `NEWMV` mode and applied to subsets of
//! macroblocks in `NEW4x4` mode (RFC 6386 §17, page 108). The format of
//! the vectors is identical in both cases, so this module provides the
//! shared primitive both call sites consume.
//!
//! Each vector has two pieces: a vertical component (row) followed by a
//! horizontal component (column). The row and column use separate coding
//! probabilities (the two [`MvContext`]s, row then column) but are
//! otherwise represented identically (§17, page 108).
//!
//! Each component is a signed integer `V` representing a vertical or
//! horizontal luma displacement of `V` quarter-pixels (and a chroma
//! displacement of `V` eighth-pixels). The absolute value of `V`, if
//! non-zero, is followed by a boolean sign. `V` may take any value
//! between -1023 and +1023, inclusive (§17.1, page 108).
//!
//! The absolute value `A` is coded in one of two ways according to its
//! size. For `0 <= A <= 7`, `A` is tree-coded against [`SMALL_MVTREE`];
//! for `8 <= A <= 1023`, the bits in the binary expansion of `A` are
//! coded using independent boolean probabilities. The coding of `A`
//! begins with a bool specifying which range is in effect (§17.1).
//!
//! ## Probability layout (§17.1 `MVPindices`)
//!
//! Each component is decoded with a 19-position probability table whose
//! offsets are:
//!
//! | index   | name          | role                                       |
//! | ------- | ------------- | ------------------------------------------ |
//! | 0       | `mvpis_short` | short (`<= 7`) vs long (`>= 8`) selector   |
//! | 1       | `MVPsign`     | sign bit for a non-zero value              |
//! | 2..=8   | `MVPshort`    | 7-position tree for the 8 short values     |
//! | 9..=18  | `MVPbits`     | 10 long-value bit probabilities (bits 0–9) |
//!
//! (`MVPshort = 2`, `MVPbits = MVPshort + 7 = 9`,
//! `MVPcount = MVPbits + 10 = 19` — see [`MV_PROB_COUNT`].)
//!
//! ## What this module does NOT do (deferred to later §16 rounds)
//!
//! This round lands only the §17 component decode and the §17.2
//! probability resolution. The surrounding inter-prediction machinery —
//! the §16.2 `mv_ref` tree, the §16.3 `vp8_find_near_mvs` census that
//! produces the `best`/`nearest`/`near` reference vectors a NEWMV offset
//! is added to, the §16.4 SPLITMV sub-block walk, the §18.1 stored-luma
//! doubling / chroma averaging, and the §18 sub-pixel interpolation — are
//! all out of scope here. [`read_mv`] returns the *raw* decoded
//! (differential) vector; adding it to a reference base and applying the
//! §18.1 range clamp belong to the inter-prediction layer.

use crate::bool_decoder::{BoolDecoder, BoolDecoderError};
use crate::coded_header::{MvProbUpdates, DEFAULT_MV_CONTEXT, MV_PROB_COUNT};
use crate::encoder::BoolEncoder;

/// A decoded motion vector — RFC 6386 §17.2 `MV` (`{ int16 row, col; }`).
///
/// `row` is the vertical component, `col` the horizontal component, each a
/// signed displacement in quarter-pixel (luma) units. As decoded here the
/// values are the raw per-component results of [`read_mv_component`]; the
/// §18.1 stored-luma doubling and reference-vector addition are applied by
/// the inter-prediction layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Mv {
    /// Vertical component (quarter-pixels), `-1023..=1023`.
    pub row: i16,
    /// Horizontal component (quarter-pixels), `-1023..=1023`.
    pub col: i16,
}

/// A single component's 19-position probability table — RFC 6386 §17.1
/// `MV_CONTEXT` (`Prob[MVPcount]`). Index roles are tabulated in the
/// module documentation.
pub type MvContext = [u8; MV_PROB_COUNT];

/// The decoder's pair of [`MvContext`]s — RFC 6386 §17.2 `mvc[2]`,
/// "always row, then column".
pub type MvContexts = [MvContext; 2];

/// `mvpis_short` — index 0: short (`<= 7`) vs long (`>= 8`) selector
/// (§17.1 `MVPindices`).
const MVP_IS_SHORT: usize = 0;
/// `MVPsign` — index 1: sign bit for a non-zero value (§17.1).
const MVP_SIGN: usize = 1;
/// `MVPshort` — index 2: base of the 7-position short-value tree (§17.1).
const MVP_SHORT: usize = 2;
/// `MVPbits` — index 9: base of the 10 long-value bit probabilities
/// (§17.1, `MVPbits = MVPshort + 7`).
const MVP_BITS: usize = MVP_SHORT + 7;

/// `small_mvtree` — RFC 6386 §17.1, the tree used for small absolute
/// values (`0 <= A <= 7`). Leaf values are the absolute value `A`
/// directly. Probabilities are read from the component context at offset
/// `MVPshort + (node_index >> 1)`.
///
/// Transcribed verbatim from the §17.1 listing:
///
/// ```text
/// 2, 8,          "0" subtree, "1" subtree
///  4, 6,         "00" subtree, "01" subtree
///   -0, -1,      0 = "000", 1 = "001"
///   -2, -3,      2 = "010", 3 = "011"
///  10, 12,       "10" subtree, "11" subtree
///   -4, -5,      4 = "100", 5 = "101"
///   -6, -7       6 = "110", 7 = "111"
/// ```
pub const SMALL_MVTREE: [i8; 14] = [
    2, 8, //
    4, 6, //
    -0, -1, //
    -2, -3, //
    10, 12, //
    -4, -5, //
    -6, -7, //
];

/// Resolve the two [`MvContext`]s for the current frame — RFC 6386 §17.2.
///
/// Each [`MvContext`] starts from [`DEFAULT_MV_CONTEXT`] (the in-tree
/// defaults, "set to above defaults every key frame"). For an interframe,
/// the parsed [`MvProbUpdates`] (round-4 `mv_prob_update()`) overlays a
/// replacement probability at each position whose update flag was 1 — per
/// the §17.2 `update_mvcontexts` body, where a transmitted `Some(prob)`
/// replaces the position's probability and a `None` leaves the default
/// (or previously-updated, across interframes) value in force.
///
/// Note the round-4 parser already applied the §17.2 `x ? x << 1 : 1`
/// reconstruction to the 7-bit literal, so the `Some(prob)` values stored
/// in [`MvProbUpdates`] are the final probabilities, copied in directly.
///
/// Persistence across interframes ("updates remain in effect until the
/// next key frame or until replaced") is the caller's responsibility: pass
/// the previously-resolved contexts as `base` when chaining interframes,
/// or [`DEFAULT_MV_CONTEXT`] for a key frame / a fresh start.
pub fn resolve_mv_contexts(base: &MvContexts, updates: &MvProbUpdates) -> MvContexts {
    let mut out = *base;
    for component in 0..2 {
        for position in 0..MV_PROB_COUNT {
            if let Some(prob) = updates[component][position] {
                out[component][position] = prob;
            }
        }
    }
    out
}

/// The key-frame / fresh-start [`MvContexts`] — both components seeded
/// from [`DEFAULT_MV_CONTEXT`]. Per RFC 6386 §17.2 the MV contexts "should
/// be set to their defaults every key frame".
pub fn default_mv_contexts() -> MvContexts {
    DEFAULT_MV_CONTEXT
}

/// Walk [`SMALL_MVTREE`] for a short component value — RFC 6386 §17.1
/// `treed_read(d, small_mvtree, p + MVPshort)`.
///
/// Probabilities are read from `p[MVPshort + (node_index >> 1)]`, i.e. the
/// component table offset by [`MVP_SHORT`]. Returns the absolute value
/// `0..=7`.
fn read_small_mv(dec: &mut BoolDecoder<'_>, p: &MvContext) -> Result<i32, BoolDecoderError> {
    let mut i: usize = 0;
    loop {
        let prob = p[MVP_SHORT + (i >> 1)];
        let bit = dec.read_bool(prob)? as usize;
        let next = SMALL_MVTREE[i + bit];
        if next <= 0 {
            return Ok((-next) as i32);
        }
        i = next as usize;
    }
}

/// Decode one motion-vector component at the current decoder position —
/// RFC 6386 §17.1 `read_mvcomponent`.
///
/// `p` is the component's 19-position [`MvContext`] (row or column). The
/// returned value is the signed component in quarter-pixel units,
/// `-1023..=1023`.
///
/// The §17.1 procedure:
///
/// 1. Read `p[mvpis_short]`. If true, `8 <= A <= 1023` (long form):
///    read bits 0,1,2 then bits 9,8,7,6,5,4 from `p[MVPbits + bit]`, and
///    finally bit 3 — which is *not* explicitly coded when `A <= 15`
///    (because a long-coded value is `>= 8`, so bit 3 is implied 1 there).
/// 2. Otherwise `0 <= A <= 7` (short form): tree-decode against
///    [`SMALL_MVTREE`].
/// 3. If `A` is non-zero, read the sign at `p[MVPsign]`; negate when set.
pub fn read_mv_component(
    dec: &mut BoolDecoder<'_>,
    p: &MvContext,
) -> Result<i32, BoolDecoderError> {
    let mut a: i32 = 0;

    if dec.read_bool(p[MVP_IS_SHORT])? {
        // Long form, 8 <= A <= 1023.
        // Read bits 0, 1, 2 (low order).
        for i in 0..3 {
            a += (dec.read_bool(p[MVP_BITS + i])? as i32) << i;
        }
        // Read bits 9, 8, 7, 6, 5, 4 (high order, descending).
        let mut i = 9;
        loop {
            a += (dec.read_bool(p[MVP_BITS + i])? as i32) << i;
            i -= 1;
            if i <= 3 {
                break;
            }
        }
        // Bit 3: a long-coded value is >= 8, so if A so far is <= 15
        // (no bit above bit 3 set), bit 3 is implicitly 1 and is not
        // explicitly coded. Otherwise it is read.
        if (a & 0xfff0) == 0 || dec.read_bool(p[MVP_BITS + 3])? {
            a += 8;
        }
    } else {
        // Short form, 0 <= A <= 7.
        a = read_small_mv(dec, p)?;
    }

    if a != 0 && dec.read_bool(p[MVP_SIGN])? {
        Ok(-a)
    } else {
        Ok(a)
    }
}

/// Decode a complete motion vector at the current decoder position — RFC
/// 6386 §17.2 `read_mv`.
///
/// Reads the row component against `contexts[0]` then the column component
/// against `contexts[1]` (§17.2: "always row, then column"). The returned
/// [`Mv`] is the raw decoded differential vector; the inter-prediction
/// layer adds it to the §16.3 reference base and applies the §18.1 clamp.
pub fn read_mv(dec: &mut BoolDecoder<'_>, contexts: &MvContexts) -> Result<Mv, BoolDecoderError> {
    let row = read_mv_component(dec, &contexts[0])? as i16;
    let col = read_mv_component(dec, &contexts[1])? as i16;
    Ok(Mv { row, col })
}

/// Walk [`SMALL_MVTREE`] for an encoder-side short component — RFC 6386
/// §17.1, the inverse of [`read_small_mv`]. Writes the path from the root
/// node to leaf `-leaf` against [`BoolEncoder::write_bool`] at the same
/// per-node probability offset the decoder reads at.
fn write_small_mv(enc: &mut BoolEncoder, p: &MvContext, leaf: i32) {
    debug_assert!((0..=7).contains(&leaf));
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
    let found = find_path(&SMALL_MVTREE, 0, leaf as i8, &mut path);
    debug_assert!(found, "no SMALL_MVTREE path for leaf {leaf}");
    let mut i: usize = 0;
    for &bit in &path {
        let prob = p[MVP_SHORT + (i >> 1)];
        enc.write_bool(prob, bit);
        let next = SMALL_MVTREE[i + bit as usize];
        if next <= 0 {
            break;
        }
        i = next as usize;
    }
}

/// Encode one motion-vector component into `enc` — the inverse of
/// [`read_mv_component`] (RFC 6386 §17.1 `read_mvcomponent`).
///
/// `p` is the component's 19-position [`MvContext`] (row or column). The
/// component value `V` is a signed quarter-pixel displacement in
/// `-1023..=1023`; debug builds assert the range. The emitted bits exactly
/// round-trip through `read_mv_component(dec, p)`.
///
/// The §17.1 emit procedure:
///
/// 1. Compute `A = |V|`. If `A >= 8`, write `1` against `p[MVPis_short]`
///    then bits 0, 1, 2, then bits 9..=4 descending. Bit 3 is only
///    explicitly coded when `A > 15` (the long-form decoder assumes bit 3
///    is implicitly 1 in `[8, 15]`).
/// 2. Otherwise (`A <= 7`) write `0` against `p[MVPis_short]` then walk
///    [`SMALL_MVTREE`] to leaf `A`.
/// 3. If `A != 0` write the sign at `p[MVPsign]` (`true` ⇒ negative).
pub fn write_mv_component(enc: &mut BoolEncoder, p: &MvContext, value: i32) {
    debug_assert!(
        (-1023..=1023).contains(&value),
        "§17.1 MV component out of range: {value}"
    );
    let a = value.unsigned_abs() as i32;
    if a > 7 {
        enc.write_bool(p[MVP_IS_SHORT], true);
        for i in 0..3 {
            enc.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
        }
        let mut i = 9;
        loop {
            enc.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
            i -= 1;
            if i <= 3 {
                break;
            }
        }
        // Bit 3 only emitted when A > 15 (long-form decoder treats bit 3
        // as implicit 1 in [8, 15]).
        if (a & 0xfff0) != 0 {
            enc.write_bool(p[MVP_BITS + 3], (a >> 3) & 1 != 0);
        }
    } else {
        enc.write_bool(p[MVP_IS_SHORT], false);
        write_small_mv(enc, p, a);
    }
    if a != 0 {
        enc.write_bool(p[MVP_SIGN], value < 0);
    }
}

/// Encode a complete motion vector into `enc` — RFC 6386 §17.2 `read_mv`,
/// inverse direction. Writes the row component against `contexts[0]` then
/// the column component against `contexts[1]` ("always row, then column").
pub fn write_mv(enc: &mut BoolEncoder, contexts: &MvContexts, mv: Mv) {
    write_mv_component(enc, &contexts[0], mv.row as i32);
    write_mv_component(enc, &contexts[1], mv.col as i32);
}

/// Fractional-bit cost of one §17.1 MV component against context `p`.
///
/// Computes the `-log2(P(bit))` of every §17.1 boolean
/// [`write_mv_component`] would emit, summed. Used by the encoder's
/// non-normative RD trade to compare NEWMV against ZEROMV without
/// actually emitting anything. `prob` is clamped to `1..=255` so the
/// logarithm is finite (the spec's probabilities live in that range).
///
/// Mirrors the [`write_mv_component`] control flow line-for-line, so the
/// cost equals the bits the real pass below will emit (modulo the §7.3
/// renormalisation tail bytes, which add < 1 byte per partition, not per
/// MV — negligible for a per-MB ZEROMV-vs-NEWMV comparison).
pub fn mv_component_bits(p: &MvContext, value: i32) -> f64 {
    debug_assert!((-1023..=1023).contains(&value));
    #[inline]
    fn bool_bits(prob: u8, value: bool) -> f64 {
        let p0 = (prob.max(1) as f64) / 256.0;
        let p = if value { 1.0 - p0 } else { p0 };
        -p.max(1.0 / 256.0).log2()
    }
    fn small_mv_bits(p: &MvContext, leaf: i32) -> f64 {
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
        let found = find_path(&SMALL_MVTREE, 0, leaf as i8, &mut path);
        debug_assert!(found);
        let mut bits = 0.0;
        let mut i: usize = 0;
        for &bit in &path {
            bits += bool_bits(p[MVP_SHORT + (i >> 1)], bit);
            let next = SMALL_MVTREE[i + bit as usize];
            if next <= 0 {
                break;
            }
            i = next as usize;
        }
        bits
    }

    let a = value.unsigned_abs() as i32;
    let mut bits = 0.0;
    if a > 7 {
        bits += bool_bits(p[MVP_IS_SHORT], true);
        for i in 0..3 {
            bits += bool_bits(p[MVP_BITS + i], (a >> i) & 1 != 0);
        }
        let mut i = 9;
        loop {
            bits += bool_bits(p[MVP_BITS + i], (a >> i) & 1 != 0);
            i -= 1;
            if i <= 3 {
                break;
            }
        }
        if (a & 0xfff0) != 0 {
            bits += bool_bits(p[MVP_BITS + 3], (a >> 3) & 1 != 0);
        }
    } else {
        bits += bool_bits(p[MVP_IS_SHORT], false);
        bits += small_mv_bits(p, a);
    }
    if a != 0 {
        bits += bool_bits(p[MVP_SIGN], value < 0);
    }
    bits
}

/// Fractional-bit cost of an entire §17.2 motion vector against `contexts`.
///
/// Equal to `mv_component_bits(&contexts[0], row) +
/// mv_component_bits(&contexts[1], col)`; sums the row-then-column
/// emission order of [`write_mv`].
pub fn mv_bits(contexts: &MvContexts, mv: Mv) -> f64 {
    mv_component_bits(&contexts[0], mv.row as i32) + mv_component_bits(&contexts[1], mv.col as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal test-side VP8 boolean *encoder*, mirroring the proven
    /// encoder in `bool_decoder::tests` / `dct_tokens::tests`. Used to
    /// construct bitstreams the §17 routines decode back, so the round
    /// trips assert against bytes we generated from the spec's coding
    /// rules — not against any external reference.
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
            // Ensure at least two bytes for BoolDecoder::init.
            while self.out.len() < 2 {
                self.out.push(0);
            }
            self.out
        }

        /// Encode one MV component the §17.1 way against context `p`.
        fn write_mv_component(&mut self, p: &MvContext, value: i32) {
            let a = value.unsigned_abs() as i32;
            if a > 7 {
                // Long form.
                self.write_bool(p[MVP_IS_SHORT], true);
                // Bits 0,1,2.
                for i in 0..3 {
                    self.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
                }
                // Bits 9..=4 descending.
                let mut i = 9;
                loop {
                    self.write_bool(p[MVP_BITS + i], (a >> i) & 1 != 0);
                    i -= 1;
                    if i <= 3 {
                        break;
                    }
                }
                // Bit 3 only explicitly coded when A > 15.
                if (a & 0xfff0) != 0 {
                    self.write_bool(p[MVP_BITS + 3], (a >> 3) & 1 != 0);
                }
            } else {
                // Short form: walk SMALL_MVTREE finding the path to leaf `a`.
                self.write_bool(p[MVP_IS_SHORT], false);
                self.write_small_mv(p, a);
            }
            if a != 0 {
                self.write_bool(p[MVP_SIGN], value < 0);
            }
        }

        fn write_small_mv(&mut self, p: &MvContext, leaf: i32) {
            // Find the bit path from root to the -leaf entry.
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
            assert!(
                find_path(&SMALL_MVTREE, 0, leaf as i8, &mut path),
                "no SMALL_MVTREE path for leaf {leaf}"
            );
            let mut i = 0usize;
            for &bit in &path {
                let prob = p[MVP_SHORT + (i >> 1)];
                self.write_bool(prob, bit);
                let next = SMALL_MVTREE[i + bit as usize];
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }
    }

    fn roundtrip_component(ctx: &MvContext, value: i32) -> i32 {
        let mut enc = BoolEncoder::new();
        enc.write_mv_component(ctx, value);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).expect("init");
        read_mv_component(&mut dec, ctx).expect("decode")
    }

    #[test]
    fn small_mvtree_shape_matches_spec() {
        // The §17.1 listing, transcribed verbatim.
        assert_eq!(
            SMALL_MVTREE,
            [2, 8, 4, 6, 0, -1, -2, -3, 10, 12, -4, -5, -6, -7]
        );
    }

    #[test]
    fn mvp_index_offsets_match_spec() {
        // §17.1 MVPindices: MVPshort = 2, MVPbits = MVPshort + 7 = 9,
        // MVPcount = MVPbits + 10 = 19.
        assert_eq!(MVP_IS_SHORT, 0);
        assert_eq!(MVP_SIGN, 1);
        assert_eq!(MVP_SHORT, 2);
        assert_eq!(MVP_BITS, 9);
        assert_eq!(MVP_BITS + 10, MV_PROB_COUNT);
    }

    #[test]
    fn short_values_round_trip() {
        let ctx = DEFAULT_MV_CONTEXT[0];
        for v in -7..=7 {
            assert_eq!(roundtrip_component(&ctx, v), v, "short value {v}");
        }
    }

    #[test]
    fn zero_has_no_sign_bit() {
        // A == 0 must not consume a sign bit (the `A && read_bool(...)`
        // short-circuit). Round trip both contexts.
        for ctx in &DEFAULT_MV_CONTEXT {
            assert_eq!(roundtrip_component(ctx, 0), 0);
        }
    }

    #[test]
    fn long_values_round_trip() {
        let ctx = DEFAULT_MV_CONTEXT[1];
        // Boundary 8 (lowest long), 15 (highest with implicit bit 3),
        // 16 (first explicit bit 3), and a spread up to the §17.1 max.
        for &v in &[8, 9, 14, 15, 16, 17, 31, 100, 511, 512, 1023] {
            assert_eq!(roundtrip_component(&ctx, v), v, "long +{v}");
            assert_eq!(roundtrip_component(&ctx, -v), -v, "long -{v}");
        }
    }

    #[test]
    fn implicit_bit3_boundary() {
        // A = 8..=15 must decode without an explicitly-coded bit 3
        // (the `!(A & 0xfff0)` branch). A = 16..=23 has bit 4 set so the
        // 0xfff0 mask is non-zero and bit 3 IS coded; cross-check that
        // values where bit 3 is 0 (16, 17) and 1 (24, 25) both survive.
        let ctx = DEFAULT_MV_CONTEXT[0];
        for &v in &[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25] {
            assert_eq!(roundtrip_component(&ctx, v), v, "boundary {v}");
        }
    }

    #[test]
    fn full_extremes_round_trip() {
        // §17.1: V may take any value between -1023 and +1023, inclusive.
        let ctx = DEFAULT_MV_CONTEXT[0];
        assert_eq!(roundtrip_component(&ctx, 1023), 1023);
        assert_eq!(roundtrip_component(&ctx, -1023), -1023);
    }

    #[test]
    fn read_mv_uses_row_then_column_contexts() {
        // Encode row=5 against context[0], col=-300 against context[1];
        // a single read_mv must recover both, proving it reads row first
        // (context 0) then column (context 1).
        let contexts = default_mv_contexts();
        let mut enc = BoolEncoder::new();
        enc.write_mv_component(&contexts[0], 5);
        enc.write_mv_component(&contexts[1], -300);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).expect("init");
        let mv = read_mv(&mut dec, &contexts).expect("read_mv");
        assert_eq!(mv, Mv { row: 5, col: -300 });
    }

    #[test]
    fn read_mv_distinct_contexts_are_load_bearing() {
        // Build two contexts that differ at the is_short probability so a
        // swapped-context decode desyncs. Encode against the correct
        // order; a deliberately swapped decode must NOT recover the value.
        let mut contexts = default_mv_contexts();
        contexts[0][MVP_IS_SHORT] = 30;
        contexts[1][MVP_IS_SHORT] = 220;
        let mut enc = BoolEncoder::new();
        enc.write_mv_component(&contexts[0], 200);
        enc.write_mv_component(&contexts[1], 9);
        let bytes = enc.finish();

        let mut dec = BoolDecoder::init(&bytes).expect("init");
        let good = read_mv(&mut dec, &contexts).expect("read_mv");
        assert_eq!(good, Mv { row: 200, col: 9 });

        let swapped = [contexts[1], contexts[0]];
        let mut dec2 = BoolDecoder::init(&bytes).expect("init");
        let bad = read_mv(&mut dec2, &swapped);
        // Either it errors or it produces a different value — the point is
        // it does NOT recover (200, 9).
        if let Ok(v) = bad {
            assert_ne!(v, Mv { row: 200, col: 9 });
        }
    }

    #[test]
    fn default_mv_contexts_equals_spec_defaults() {
        assert_eq!(default_mv_contexts(), DEFAULT_MV_CONTEXT);
        // Spot-check §17.2 default_mv_context values.
        assert_eq!(default_mv_contexts()[0][0], 162); // row is_short
        assert_eq!(default_mv_contexts()[1][0], 164); // col is_short
    }

    #[test]
    fn resolve_no_updates_is_identity() {
        let base = default_mv_contexts();
        let updates: MvProbUpdates = [[None; MV_PROB_COUNT]; 2];
        assert_eq!(resolve_mv_contexts(&base, &updates), base);
    }

    #[test]
    fn resolve_overlays_some_positions() {
        let base = default_mv_contexts();
        let mut updates: MvProbUpdates = [[None; MV_PROB_COUNT]; 2];
        // Replace row is_short and column sign.
        updates[0][MVP_IS_SHORT] = Some(200);
        updates[1][MVP_SIGN] = Some(50);
        let resolved = resolve_mv_contexts(&base, &updates);

        assert_eq!(resolved[0][MVP_IS_SHORT], 200);
        assert_eq!(resolved[1][MVP_SIGN], 50);
        // Untouched positions keep the defaults.
        assert_eq!(resolved[0][MVP_SIGN], base[0][MVP_SIGN]);
        assert_eq!(resolved[1][MVP_IS_SHORT], base[1][MVP_IS_SHORT]);
    }

    #[test]
    fn resolve_persists_across_chained_frames() {
        // §17.2: "updates remain in effect until the next key frame or
        // until replaced". Pass a previously-resolved context as `base`;
        // a None at a position keeps the earlier-updated value, a Some
        // replaces it.
        let mut first_updates: MvProbUpdates = [[None; MV_PROB_COUNT]; 2];
        first_updates[0][MVP_SHORT] = Some(99);
        first_updates[0][MVP_SHORT + 1] = Some(33);
        let frame1 = resolve_mv_contexts(&default_mv_contexts(), &first_updates);

        let mut second_updates: MvProbUpdates = [[None; MV_PROB_COUNT]; 2];
        second_updates[0][MVP_SHORT + 1] = Some(77); // replace
        let frame2 = resolve_mv_contexts(&frame1, &second_updates);

        assert_eq!(frame2[0][MVP_SHORT], 99); // carried forward
        assert_eq!(frame2[0][MVP_SHORT + 1], 77); // replaced this frame
    }

    #[test]
    fn round_trip_under_resolved_context() {
        // Resolve a context with a few overlays, then prove the component
        // codec round-trips against the resolved table (decode reads the
        // updated probabilities, not the defaults).
        let mut updates: MvProbUpdates = [[None; MV_PROB_COUNT]; 2];
        updates[0][MVP_IS_SHORT] = Some(40);
        updates[0][MVP_SHORT + 2] = Some(180);
        updates[0][MVP_BITS] = Some(20);
        let ctx = resolve_mv_contexts(&default_mv_contexts(), &updates)[0];

        for &v in &[0, 3, 7, 8, 15, 16, 200, -200, 1023, -1023] {
            assert_eq!(roundtrip_component(&ctx, v), v, "resolved-ctx {v}");
        }
    }

    #[test]
    fn mv_default_is_zero_vector() {
        assert_eq!(Mv::default(), Mv { row: 0, col: 0 });
    }

    // ─── public write_mv_component / mv_component_bits coverage ──────────

    /// Round-trip a component through the **production** encoder
    /// [`super::write_mv_component`] (using the crate's real [`BoolEncoder`])
    /// and the §17.1 decoder — pins the public API encodes the same bits
    /// the local test encoder produces.
    fn roundtrip_component_public(ctx: &MvContext, value: i32) -> i32 {
        let mut enc = crate::encoder::BoolEncoder::new();
        super::write_mv_component(&mut enc, ctx, value);
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).expect("init");
        read_mv_component(&mut dec, ctx).expect("decode")
    }

    #[test]
    fn public_write_mv_component_round_trips_short_and_long() {
        let ctx = DEFAULT_MV_CONTEXT[0];
        for v in -7..=7 {
            assert_eq!(roundtrip_component_public(&ctx, v), v, "short v={v}");
        }
        for &v in &[8, 9, 15, 16, 17, 24, 100, 511, 512, 1023] {
            assert_eq!(roundtrip_component_public(&ctx, v), v, "long +{v}");
            assert_eq!(roundtrip_component_public(&ctx, -v), -v, "long -{v}");
        }
    }

    #[test]
    fn public_write_mv_round_trips_through_read_mv() {
        // write_mv writes row first (against contexts[0]) then column —
        // a `read_mv` recovers both at the same emission order.
        let contexts = default_mv_contexts();
        let mut enc = crate::encoder::BoolEncoder::new();
        super::write_mv(&mut enc, &contexts, Mv { row: -12, col: 47 });
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).expect("init");
        let mv = read_mv(&mut dec, &contexts).expect("read_mv");
        assert_eq!(mv, Mv { row: -12, col: 47 });
    }

    #[test]
    fn mv_component_bits_zero_is_just_is_short_bit() {
        // Encoding V=0 emits the `is_short=false` bit + the SMALL_MVTREE
        // path to leaf 0 ("000") — three bits at probs[2]/[3]/[4] — and
        // no sign bit (A==0). All bits are emitted at the spec defaults.
        let ctx = DEFAULT_MV_CONTEXT[0];
        let bits = super::mv_component_bits(&ctx, 0);
        // The lower bound is the count of bools written (4): a degenerate
        // p=255 short side would still cost ~0 bits per bool, but with the
        // real defaults each bool costs > 0 bits.
        assert!(
            bits > 0.0,
            "non-zero bool cost expected for V=0, got {bits}"
        );
        assert!(bits < 8.0, "V=0 emits a handful of bools, got {bits}");
    }

    #[test]
    fn mv_component_bits_grow_with_magnitude() {
        // Cheaper to code a zero MV than a far-away MV against the
        // standard MV_CONTEXT defaults — pin this so an accidental
        // probability swap surfaces. (Strict monotonicity holds at every
        // doubled magnitude on the §17.2 default table.)
        let ctx = DEFAULT_MV_CONTEXT[0];
        let b0 = super::mv_component_bits(&ctx, 0);
        let b4 = super::mv_component_bits(&ctx, 4);
        let b32 = super::mv_component_bits(&ctx, 32);
        let b256 = super::mv_component_bits(&ctx, 256);
        assert!(
            b0 < b4 && b4 < b32 && b32 < b256,
            "expected monotone growth, got {b0} < {b4} < {b32} < {b256}"
        );
    }

    #[test]
    fn mv_bits_is_sum_of_component_bits() {
        // mv_bits is documented to equal the row + column component bit
        // cost; pin it against an explicit (row, col).
        let ctx = default_mv_contexts();
        let mv = Mv { row: 7, col: -200 };
        let total = super::mv_bits(&ctx, mv);
        let parts = super::mv_component_bits(&ctx[0], mv.row as i32)
            + super::mv_component_bits(&ctx[1], mv.col as i32);
        assert!((total - parts).abs() < 1e-12);
    }
}
