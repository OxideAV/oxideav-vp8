#![no_main]

//! Fuzz: panic-freedom of the public §13.2 / §13.3 token-coding
//! primitives.
//!
//! The pre-existing fuzz targets reach §13 only indirectly
//! through `decode_vp8` / `Vp8DecoderState::decode_frame` /
//! `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which gate
//! the per-block token walk behind a fully-formed frame-header +
//! coded-header + dequant state. That leaves the §13 primitive surface
//! itself — `decode_block`, `decode_mb_coeffs`, `merge_default_token_probs`,
//! and the §13.3 above/left predictor lattice — directly under-fuzzed.
//!
//! This target drives that surface directly with an attacker-shaped
//! `(probability override list, predictor lattice, bool partition)`
//! envelope so libFuzzer can hit every node of the §13.2 coefficient
//! tree, every `Cat1..Cat6` extra-bits leg of §13.2, every per-position
//! probability slot of `coeff_probs[4][8][3][11]`, and every transition
//! of the §13.3 `MbEntropyCtx` above/left flag pair across the 25-block
//! `(Y2, 16 Y, 4 U, 4 V)` walk.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`  | flags byte: bit0 path (`decode_block` vs `decode_mb_coeffs`); bit1 `has_y2`; bit2 `mb_skip_coeff`; bit3 `above_has_nonzero`; bit4 `left_has_nonzero`; bits 5..6 `BlockType` selector; bit7 toggles default vs all-128 prob seed |
//! | `[1]`  | override count `n` in `0..=MAX_OVERRIDES` (saturated) |
//! | `[2 .. 2+5n]` | five-byte override tuples — `(plane, band, ctx, pos, prob)` each modulo its respective dimension upper bound (`4, 8, 3, 11, 256`) |
//! | `[2+5n .. 2+5n+2]` | above-context nonzero bitmap (9 bits used, low 7 of byte 0 + low 2 of byte 1) |
//! | `[2+5n+2 .. 2+5n+4]` | left-context nonzero bitmap (same layout) |
//! | rest      | bool-decoder partition bytes |
//!
//! The harness:
//!
//! 1. Reads up to `MAX_OVERRIDES` override tuples and seeds a
//!    `TokenProbUpdates` from them, then calls
//!    [`merge_default_token_probs`] to fold the overrides into a
//!    `CoeffProbs`. When the flags byte's bit7 is set, the seed table
//!    starts from an all-`128` lattice instead of `DEFAULT_COEFF_PROBS`
//!    (constructed via `merge_default_token_probs` on a fully-populated
//!    `TokenProbUpdates`) so the harness covers the rare-but-legal
//!    state where every band/context slot has been replaced by §19.2
//!    `token_prob_update`.
//! 2. Initialises a [`BoolDecoder`] from the partition bytes via
//!    `BoolDecoder::init_partition` (the §20 reference's
//!    short-tolerant init), so the harness never returns early on a
//!    1- or 0-byte partition — those legal inputs still reach the
//!    decoder's renormalisation tail-case.
//! 3. On the `decode_block` leg: zeros a `[i16; 16]` coefficient
//!    buffer, picks `BlockType` from bits 5..6 of the flags byte, and
//!    invokes `decode_block(&mut dec, block_type, &probs,
//!    above_has_nonzero, left_has_nonzero, &mut coeffs)`. Any returned
//!    `Result` (Ok or Err) is dropped; the only assertion is
//!    panic-freedom.
//! 4. On the `decode_mb_coeffs` leg: builds an `MbEntropyCtx` for the
//!    above and left predictor vectors from the two two-byte bitmap
//!    headers (each carries 9 useful bits — the `MB_ENTROPY_CTX_LEN`
//!    width) and invokes `decode_mb_coeffs(&mut dec, has_y2,
//!    mb_skip_coeff, &probs, &mut above, &mut left)`. The §13.3
//!    `mb_skip_coeff` short-circuit also runs the `reset_for_skip` leg
//!    of the predictor lattice.
//!
//! Caps: input <= 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth); `MAX_OVERRIDES = 32` (≥ 16 ensures
//! every position slot of one `(plane, band, ctx)` triple can be
//! replaced).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    bool_decoder::BoolDecoder,
    dct_tokens::{
        decode_block, decode_mb_coeffs, merge_default_token_probs, BlockType, CoeffProbs,
        MbEntropyCtx, MB_ENTROPY_CTX_LEN,
    },
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 2;
const OVERRIDE_BYTES: usize = 5;
const MAX_OVERRIDES: usize = 32;
const CTX_BITMAP_BYTES: usize = 2;
const TOTAL_CTX_BITMAP_BYTES: usize = 2 * CTX_BITMAP_BYTES;

const PROB_PLANES: usize = 4;
const PROB_BANDS: usize = 8;
const PROB_CONTEXTS: usize = 3;
const PROB_POSITIONS: usize = 11;

type TokenProbOverrides =
    [[[[Option<u8>; PROB_POSITIONS]; PROB_CONTEXTS]; PROB_BANDS]; PROB_PLANES];

fn block_type_from_bits(bits: u8) -> BlockType {
    match bits & 0b11 {
        0 => BlockType::YAfterY2,
        1 => BlockType::Y2,
        2 => BlockType::UV,
        _ => BlockType::YNoY2,
    }
}

/// Build a `[bool; MB_ENTROPY_CTX_LEN]` from a two-byte (lo, hi) pair
/// taking 9 bits total. Bits 0..7 come from the low byte; bit 8 comes
/// from bit 0 of the high byte. The high byte's bits 1..7 also seed
/// the spare bit so libFuzzer can flip every slot with a one-byte
/// edit.
fn decode_ctx_bitmap(lo: u8, hi: u8) -> [bool; MB_ENTROPY_CTX_LEN] {
    let mut out = [false; MB_ENTROPY_CTX_LEN];
    for (i, slot) in out.iter_mut().enumerate().take(8) {
        *slot = ((lo >> i) & 1) != 0;
    }
    out[8] = (hi & 1) != 0;
    out
}

/// Seed an all-`Some(128)` `TokenProbUpdates` so
/// `merge_default_token_probs` returns an all-128 `CoeffProbs` (the
/// §19.2 "every slot replaced" envelope).
fn full_128_overrides() -> TokenProbOverrides {
    [[[[Some(128u8); PROB_POSITIONS]; PROB_CONTEXTS]; PROB_BANDS]; PROB_PLANES]
}

/// Decode up to `MAX_OVERRIDES` `(plane, band, ctx, pos, prob)`
/// override tuples from the payload, then fold them via
/// `merge_default_token_probs`.
fn build_probs(seed_full_128: bool, override_payload: &[u8]) -> CoeffProbs {
    let mut overrides: TokenProbOverrides = if seed_full_128 {
        full_128_overrides()
    } else {
        [[[[None; PROB_POSITIONS]; PROB_CONTEXTS]; PROB_BANDS]; PROB_PLANES]
    };

    let mut cursor = 0;
    while cursor + OVERRIDE_BYTES <= override_payload.len() {
        let plane = (override_payload[cursor] as usize) % PROB_PLANES;
        let band = (override_payload[cursor + 1] as usize) % PROB_BANDS;
        let ctx = (override_payload[cursor + 2] as usize) % PROB_CONTEXTS;
        let pos = (override_payload[cursor + 3] as usize) % PROB_POSITIONS;
        let prob = override_payload[cursor + 4];
        overrides[plane][band][ctx][pos] = Some(prob);
        cursor += OVERRIDE_BYTES;
    }

    merge_default_token_probs(&overrides)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let n_raw = data[1] as usize;
    let n = n_raw.min(MAX_OVERRIDES);

    let path_is_mb = (flags & 0b0000_0001) != 0;
    let has_y2 = (flags & 0b0000_0010) != 0;
    let mb_skip = (flags & 0b0000_0100) != 0;
    let above_nz = (flags & 0b0000_1000) != 0;
    let left_nz = (flags & 0b0001_0000) != 0;
    let block_type_bits = (flags >> 5) & 0b11;
    let seed_full_128 = (flags & 0b1000_0000) != 0;

    // Slice the override payload — saturate to whatever the data
    // actually carries. The remainder feeds the predictor bitmap and
    // partition.
    let override_window = n * OVERRIDE_BYTES;
    let after_overrides = HEADER_BYTES.saturating_add(override_window);
    if after_overrides > data.len() {
        return;
    }
    let override_payload = &data[HEADER_BYTES..after_overrides];

    let probs = build_probs(seed_full_128, override_payload);

    // Slice the two predictor bitmaps. When the input is too short
    // for them, fall through with the harness' zero default
    // (off-frame predictors are `false` per §13.3 page 64).
    let mut above_ctx_bits = (0u8, 0u8);
    let mut left_ctx_bits = (0u8, 0u8);
    let mut after_ctx = after_overrides;
    if data.len() >= after_overrides + TOTAL_CTX_BITMAP_BYTES {
        above_ctx_bits = (data[after_overrides], data[after_overrides + 1]);
        left_ctx_bits = (data[after_overrides + 2], data[after_overrides + 3]);
        after_ctx = after_overrides + TOTAL_CTX_BITMAP_BYTES;
    }

    let partition = &data[after_ctx..];

    // §20 reference: `init_partition` tolerates a 0- or 1-byte
    // partition and never returns an error, so the rest of the
    // harness always runs.
    let mut dec = BoolDecoder::init_partition(partition);

    if path_is_mb {
        // §13.3 per-macroblock walk: 25 residual blocks plus the
        // predictor-lattice short-circuit on `mb_skip_coeff`.
        let mut above = MbEntropyCtx {
            nonzero: decode_ctx_bitmap(above_ctx_bits.0, above_ctx_bits.1),
        };
        let mut left = MbEntropyCtx {
            nonzero: decode_ctx_bitmap(left_ctx_bits.0, left_ctx_bits.1),
        };
        let _ = decode_mb_coeffs(&mut dec, has_y2, mb_skip, &probs, &mut above, &mut left);
    } else {
        // §13.3 single-block walk: zero `coeffs` and pick the
        // `BlockType` from bits 5..6 of the flags byte.
        let block_type = block_type_from_bits(block_type_bits);
        let mut coeffs = [0i16; 16];
        let _ = decode_block(&mut dec, block_type, &probs, above_nz, left_nz, &mut coeffs);
    }
});
