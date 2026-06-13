#![no_main]

//! Fuzz: panic-freedom of the §14.1 dequantization *factor-derivation*
//! and full-macroblock dequant-apply surface, plus the §13.3→§14.1
//! token→dequant wrapper.
//!
//! The pre-existing transform-primitive target reaches §14.1 only at the
//! single-block leaf (`dequant_block(coeffs, dc_factor, ac_factor)` with
//! the two factors handed in directly), and the token target stops at
//! `decode_mb_coeffs` without ever deriving factors or scaling a block.
//! That leaves the §14.1 / §20.4 `dequant_init` *factor construction*
//! (`MbDequantFactors::from_base_and_deltas` / `from_quant_indices` /
//! `for_segment`), the whole-macroblock apply
//! (`MbDequantFactors::dequantize` over the Y2 + 16 Y + 4 U + 4 V block
//! set), and the bitstream→dequant wrapper (`decode_and_dequantize_mb`)
//! directly under-fuzzed. This target drives all three from raw bytes.
//!
//! Surface coverage:
//!
//! * `MbDequantFactors::from_base_and_deltas(q, ydc, y2dc, y2ac, uvdc,
//!   uvac)` — the §20.4 core: the six per-plane factors built from one
//!   attacker-chosen base index plus five attacker-chosen plane deltas,
//!   each fed through the §9.6 `clamp_qindex` saturation and the §20.4
//!   scaling/clamping rules (`*2` Y2-DC, `*155/100` then floor-8 Y2-AC,
//!   cap-132 chroma-DC). The base index and deltas span the full `i32`
//!   range including the `i32::MIN`/`i32::MAX` cliff endpoints, so the
//!   `q + delta` additions and the `*155` / `*2` products are exercised
//!   at the arithmetic edges where a naïve `i32` add/mul would wrap.
//! * `MbDequantFactors::from_quant_indices(&qi)` — the frame-level
//!   (no-segmentation) derivation from a `QuantIndices` whose base and
//!   five optional `i8` deltas are taken from the byte stream (each
//!   delta independently present/absent).
//! * `MbDequantFactors::for_segment(&qi, segment_quant, absolute)` — the
//!   §10 per-segment override (absolute replaces the base, delta adds to
//!   it) with an attacker-chosen `segment_quant` over the full `i32`
//!   range and both mode bits.
//! * `MbDequantFactors::dequantize(&mut coeffs)` — apply the six derived
//!   factors across a full `MbCoeffs` (Y2, sixteen Y, four U, four V),
//!   with every one of the 25 blocks' sixteen coefficients seeded from
//!   the byte stream including the `i16::MIN`/`i16::MAX` cliffs, so the
//!   `coeff(i32) * factor(i32) as i16` wrapping store is hit at its
//!   widest products.
//! * `decode_and_dequantize_mb(dec, has_y2, mb_skip, probs, factors,
//!   above, left)` — the §13.3→§14.1 wrapper: decode one macroblock's
//!   raw quantized coefficient tokens from a bool partition, then scale
//!   them with the derived factors. Any `Ok`/`Err` is dropped; the only
//!   contract under test is panic-freedom (no index-out-of-bounds, no
//!   arithmetic overflow panic in a debug build, no abort).
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`  | flags: bit0 `has_y2`; bit1 `mb_skip`; bit2 `segment_absolute`; bit3 `above_nz_seed`; bit4 `left_nz_seed`; bit5 `derive_via_segment` (else frame-level); bits 6..7 unused |
//! | `[1..5]`   | base quant index `q` (i32 LE) for the `from_base_and_deltas` leg |
//! | `[5..25]`  | five `i32` LE plane deltas: ydc, y2dc, y2ac, uvdc, uvac |
//! | `[25]`     | `QuantIndices.y_ac_qi` (u8) |
//! | `[26]`     | delta-present bitmap (5 bits used) for the five §9.6 `Option<i8>` deltas |
//! | `[27..32]` | five `i8` delta values (used iff their present-bit is set) |
//! | `[32..36]` | `segment_quant` (i32 LE) for the `for_segment` leg |
//! | `[36..68]` | 32 bytes seeding the 25-block `MbCoeffs` coefficient lattice (folded over every coefficient slot, with the four cliff values force-injected) |
//! | rest       | bool-decoder partition bytes for `decode_and_dequantize_mb` |
//!
//! Caps: input <= 4 KiB (libFuzzer default; re-checked at entry as
//! defence-in-depth). The header is a fixed 68-byte prefix; anything
//! shorter still runs the `from_base_and_deltas` cliff sweep (which is
//! seeded from constants, not the payload) so even a 1-byte input
//! exercises the §20.4 arithmetic edges.
//!
//! All truth: RFC 6386 §14.1 / §20.4 (`dequant_init`) and §10 (the
//! per-segment quantizer override). No reference implementation of any
//! kind was consulted in authoring this harness.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    bool_decoder::BoolDecoder,
    coded_header::QuantIndices,
    dct_tokens::{decode_mb_coeffs, MbEntropyCtx, DEFAULT_COEFF_PROBS, MB_ENTROPY_CTX_LEN},
    dequant::{decode_and_dequantize_mb, MbDequantFactors},
    frame::MbCoeffs,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 68;
const COEFF_SEED_BYTES: usize = 32;

/// The §9.6 / §20.4 arithmetic cliff values for the `i32` base index and
/// plane deltas: the loop-driven payload values are sparse, so these
/// constants guarantee every `from_base_and_deltas` run also probes the
/// edges where `q + delta` and the `*155` / `*2` products are widest.
const I32_CLIFFS: [i32; 7] = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];

/// The §14.1 coefficient cliffs (`coeff(i32) * factor(i32) as i16` store
/// is widest at these), force-injected into the `MbCoeffs` lattice.
const I16_CLIFFS: [i16; 5] = [i16::MIN, i16::MIN + 1, -1, i16::MAX - 1, i16::MAX];

fn read_i32_le(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Seed one `[i16; 16]` block from the 32-byte coefficient window folded
/// at `start`, force-injecting the §14.1 cliffs at the low indices so the
/// widest products are always present in every block.
fn seed_block(window: &[u8], start: usize) -> [i16; 16] {
    let mut blk = [0i16; 16];
    for (i, slot) in blk.iter_mut().enumerate() {
        if i < I16_CLIFFS.len() {
            *slot = I16_CLIFFS[i];
            continue;
        }
        // Fold two bytes of the window (wrapping) into an i16 for the
        // remaining coefficient slots.
        let lo = window[(start + i) % window.len()];
        let hi = window[(start + i + 1) % window.len()];
        *slot = i16::from_le_bytes([lo, hi]);
    }
    blk
}

/// Build a full 25-block `MbCoeffs` from the coefficient seed window so
/// `dequantize` scales non-trivial data in every plane.
fn seed_mb_coeffs(window: &[u8]) -> MbCoeffs {
    let y2 = seed_block(window, 0);
    let mut y = [[0i16; 16]; 16];
    for (j, blk) in y.iter_mut().enumerate() {
        *blk = seed_block(window, j + 1);
    }
    let mut u = [[0i16; 16]; 4];
    for (j, blk) in u.iter_mut().enumerate() {
        *blk = seed_block(window, j + 17);
    }
    let mut v = [[0i16; 16]; 4];
    for (j, blk) in v.iter_mut().enumerate() {
        *blk = seed_block(window, j + 21);
    }
    MbCoeffs { y2, y, u, v }
}

/// Build an `MbEntropyCtx` whose nine non-zero flags are all set to
/// `seed` (the §13.3 predictor lattice's two extreme states).
fn ctx(seed: bool) -> MbEntropyCtx {
    MbEntropyCtx {
        nonzero: [seed; MB_ENTROPY_CTX_LEN],
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // --- §20.4 cliff sweep: always runs, even on a tiny input. ---
    // Every (base, delta) pair from the cliff set, exercising the
    // `q + delta` additions and the `*155` / `*2` products at the
    // arithmetic edges. `from_base_and_deltas` must never panic.
    for &q in &I32_CLIFFS {
        for &d in &I32_CLIFFS {
            let f = MbDequantFactors::from_base_and_deltas(q, d, d, d, d, d);
            // The derived factors must be usable by the apply path
            // without panicking: scale a one-block cliff lattice.
            let mut coeffs = MbCoeffs {
                y2: I16_CLIFFS_BLOCK,
                y: [I16_CLIFFS_BLOCK; 16],
                u: [I16_CLIFFS_BLOCK; 4],
                v: [I16_CLIFFS_BLOCK; 4],
            };
            f.dequantize(&mut coeffs);
        }
    }

    // --- payload-driven legs (only when the fixed header is present). ---
    if data.len() < HEADER_BYTES {
        return;
    }

    let flags = data[0];
    let has_y2 = (flags & 0b0000_0001) != 0;
    let mb_skip = (flags & 0b0000_0010) != 0;
    let seg_absolute = (flags & 0b0000_0100) != 0;
    let above_nz = (flags & 0b0000_1000) != 0;
    let left_nz = (flags & 0b0001_0000) != 0;
    let via_segment = (flags & 0b0010_0000) != 0;

    let base_q = read_i32_le(data, 1);
    let ydc = read_i32_le(data, 5);
    let y2dc = read_i32_le(data, 9);
    let y2ac = read_i32_le(data, 13);
    let uvdc = read_i32_le(data, 17);
    let uvac = read_i32_le(data, 21);

    // Attacker-shaped `from_base_and_deltas` call (full i32 range).
    let _ = MbDequantFactors::from_base_and_deltas(base_q, ydc, y2dc, y2ac, uvdc, uvac);

    // Build a QuantIndices: base + five Option<i8> deltas gated by a
    // present-bitmap so every (present, absent) combination is reachable.
    let present = data[26];
    let mk_delta = |bit: u8, byte: u8| -> Option<i8> {
        if (present >> bit) & 1 != 0 {
            Some(byte as i8)
        } else {
            None
        }
    };
    let qi = QuantIndices {
        y_ac_qi: data[25],
        y_dc_delta: mk_delta(0, data[27]),
        y2_dc_delta: mk_delta(1, data[28]),
        y2_ac_delta: mk_delta(2, data[29]),
        uv_dc_delta: mk_delta(3, data[30]),
        uv_ac_delta: mk_delta(4, data[31]),
    };

    let segment_quant = read_i32_le(data, 32);

    // Derive the macroblock factors via whichever §14.1 path the flags
    // select: frame-level or §10 per-segment override.
    let factors = if via_segment {
        MbDequantFactors::for_segment(&qi, segment_quant, seg_absolute)
    } else {
        MbDequantFactors::from_quant_indices(&qi)
    };

    // Apply the derived factors across a full 25-block coefficient
    // lattice seeded from the payload's coefficient window.
    let coeff_window = &data[36..36 + COEFF_SEED_BYTES];
    let mut coeffs = seed_mb_coeffs(coeff_window);
    factors.dequantize(&mut coeffs);

    // --- §13.3 → §14.1 token→dequant wrapper. ---
    // Drive `decode_and_dequantize_mb` with the trailing partition bytes.
    // `init_partition` tolerates a 0/1-byte partition (§20 reference's
    // short-tolerant init) so this leg always runs.
    let partition = &data[HEADER_BYTES..];
    let mut dec = BoolDecoder::init_partition(partition);
    let mut above = ctx(above_nz);
    let mut left = ctx(left_nz);
    let _ = decode_and_dequantize_mb(
        &mut dec,
        has_y2,
        mb_skip,
        &DEFAULT_COEFF_PROBS,
        &factors,
        &mut above,
        &mut left,
    );

    // Cross-check: the wrapper's token-decode half on its own (no
    // dequant) must also be panic-free on the same partition + context,
    // covering the §13.3 walk independently of the §14.1 scaling.
    let mut dec2 = BoolDecoder::init_partition(partition);
    let mut above2 = ctx(above_nz);
    let mut left2 = ctx(left_nz);
    let _ = decode_mb_coeffs(
        &mut dec2,
        has_y2,
        mb_skip,
        &DEFAULT_COEFF_PROBS,
        &mut above2,
        &mut left2,
    );
});

/// A `[i16; 16]` filled with the §14.1 coefficient cliffs (repeating)
/// so the cliff sweep's `dequantize` scales the widest products in every
/// lane.
const I16_CLIFFS_BLOCK: [i16; 16] = [
    i16::MIN,
    i16::MIN + 1,
    -1,
    i16::MAX - 1,
    i16::MAX,
    i16::MIN,
    i16::MIN + 1,
    -1,
    i16::MAX - 1,
    i16::MAX,
    i16::MIN,
    i16::MIN + 1,
    -1,
    i16::MAX - 1,
    i16::MAX,
    i16::MIN,
];
