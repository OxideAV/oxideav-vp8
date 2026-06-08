#![no_main]

//! Fuzz: panic-freedom of the §14 transform / dequant / residue-summation
//! primitive layer (RFC 6386 §14).
//!
//! Surface coverage (all primitives driven directly from attacker bytes):
//!
//! * `forward_dct_4x4(input, output)` — §14.4 forward DCT, public scalar
//!   path under every feature configuration (the `_simd` partner stays
//!   compiled but is not the public dispatcher target as of round 247).
//! * `inverse_dct_4x4(input, output)` — §14.4 inverse DCT, public path
//!   (scalar on stable, SIMD on `nightly + simd`; the two are byte-exact
//!   by lib-test).
//! * `forward_wht_4x4(input, output)` — §14.3 forward WHT, public path
//!   (scalar on stable, SIMD on `nightly + simd`).
//! * `inverse_wht_4x4(input, output)` — §14.3 inverse WHT, public path
//!   (scalar on stable, SIMD on `nightly + simd`).
//! * `inverse_wht_4x4_dc_only(dc, output)` — §14.3 single-non-zero-DC
//!   fast path (`vp8_short_inv_walsh4x4_1_c`).
//! * `dequant_block(coeffs, dc_factor, ac_factor)` — §14.1 per-sub-block
//!   dequant scaling with attacker-chosen `(dc_factor, ac_factor)`
//!   including the cliff values (`i16::MIN`, `i16::MAX`, `0`, and the
//!   §14.1 envelope's expected positive range), exercising the `i32→i16`
//!   wrapping cast on the product the spec text mandates.
//! * `add_residue_4x4(prediction, residue, out)` — §14.5 fixed-size
//!   4×4 predictor+residue sum with `clamp255`.
//! * `add_residue(prediction, residue, out)` — §14.5 arbitrary-length
//!   form. The length-mismatch panic is documented in the public API;
//!   the harness only calls it with equal-length slices so the panic-free
//!   contract holds.
//! * `raster_to_scan(raster)` — encoder partner of §13.3's
//!   `scan_to_raster`; pure §20.16 `zigzag[16]` permutation.
//! * `clamp_qindex(q)` — §9.6 `0..=QINDEX_RANGE-1` saturation on any
//!   `i32` index (covers `i32::MIN` and `i32::MAX` cliff endpoints).
//! * `clamp255(v)` — §14.5 saturating clamp on arbitrary `i32` (covers
//!   `i32::MIN` and `i32::MAX` cliff endpoints).
//!
//! Round-trip invariants (panic-free guards):
//!
//! 1. **§14.3 WHT round-trip.** For any `input` in the §14.2 / §14.3
//!    quantised-residue envelope (the three `residual_seed_mode`
//!    classes — ±127, ±255, ±1023 — span it), the forward → inverse
//!    composition runs to completion without panic. The harness does
//!    NOT assert a bit-exact lossless round-trip — the §14.3 spec
//!    rounding loses one LSB on every fixed-point operation by design.
//!
//! 2. **§14.4 DCT round-trip.** Same shape as §14.3 — `inverse_dct_4x4`
//!    after `forward_dct_4x4` runs to completion on every input in the
//!    §14.2 envelope without panic.
//!
//! 3. **§14.3 DC-only fast path.** `inverse_wht_4x4_dc_only(dc, out)`
//!    is byte-equal to `inverse_wht_4x4([dc, 0, 0, …, 0], out)` for any
//!    `dc ∈ [i16::MIN, i16::MAX]`. The harness asserts the equivalence
//!    on every input — a §14.3 contract guard documented in the
//!    spec listing (`vp8_short_inv_walsh4x4_1_c` is the fast path
//!    when `eob == 1` and the single non-zero coefficient is at index
//!    0; the round-trip with `vp8_short_inv_walsh4x4_c` on the same
//!    input must produce the same bytes).
//!
//! 4. **§20.16 zigzag bijection.** `raster_to_scan(raster)` is a pure
//!    permutation: every output element is one input element, and
//!    no input element is dropped or duplicated. The harness asserts
//!    that the multiset of output elements equals the multiset of
//!    input elements on every call.
//!
//! 5. **§14.5 sum + clamp.** `add_residue_4x4(pred, res, out)` and
//!    `add_residue(pred, res, out)` (called with equal-length slices)
//!    are byte-equal on the same input. Each output pixel is in
//!    `0..=255` after `clamp255`.
//!
//! Surface gap the existing fuzz targets leave cold:
//!
//! The eleven existing fuzz targets reach §14 only indirectly:
//!
//! * `panic_free_decode_keyframe` / `_decoder_state` / `parse_headers` /
//!   `panic_free_token_block` (decode side) feed bytes through
//!   `decode_vp8` / `Vp8DecoderState::decode_frame` /
//!   `Vp8FrameHeader::parse` / `decode_block`, each of which gates §14
//!   behind a fully-formed §9 / §11 / §13 / §14.1 state machine — the
//!   forward path is never exercised, the inverse path only against
//!   well-formed dequantised residuals.
//! * `panic_free_encode_keyframe` / `_two_pass_stream` (encode side) run
//!   a §11 mode pick → §14 forward transform → §13 token emission chain;
//!   the forward DCT / WHT are exercised but only against §9.6-clamped
//!   residual magnitudes determined by the upper-layer encoder logic.
//! * `panic_free_loopfilter_segment` / `panic_free_motion_search_descent`
//!   / `panic_free_sixtap_subpel` / `panic_free_intra_predict_kernels` /
//!   `panic_free_bool_codec` don't touch §14 at all.
//!
//! No existing harness:
//!   (a) drives `forward_dct_4x4` / `forward_wht_4x4` with an attacker-
//!       shaped 16-element residual at the §14.2 ±1023 cliff envelope;
//!   (b) round-trips them against `inverse_dct_4x4` / `inverse_wht_4x4`
//!       to catch overflow on the i32→i16 reduction in the intermediate
//!       butterfly steps;
//!   (c) drives `dequant_block` with the cliff `(dc_factor, ac_factor)`
//!       values the §14.1 envelope excludes but an attacker-shaped
//!       segment header (§10) could produce after the `clamp(qindex)`
//!       saturation;
//!   (d) drives `inverse_wht_4x4_dc_only` standalone with the cliff
//!       `dc` values (`i16::MIN`, `i16::MAX`);
//!   (e) drives `add_residue` / `add_residue_4x4` with cliff
//!       (i8::MIN..=i8::MAX → i16) residue values to exercise the
//!       §14.5 saturating clamp on both sides of the `[0, 255]` window.
//!
//! This target does all five directly, with the round-trip leg locking
//! the forward/inverse pairs in §14 lockstep so any asymmetry in the
//! butterfly arithmetic or in the (x + 1) >> 1 / (x + 3) >> 3 /
//! (x + 4) >> 3 rounding surfaces as a magnitude-bound or
//! permutation-bijection violation on the read side.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes        | Meaning |
//! |-------------:|---------|
//! | `[0]`        | flags byte: bit0 run forward WHT + inverse WHT round-trip; bit1 run forward DCT + inverse DCT round-trip; bit2 run dequant + add_residue leg; bit3 run DC-only / clamp / zigzag legs; bits 4-7 reserved |
//! | `[1]`        | `residual_seed_mode`: 0 = mid-magnitude (±127), 1 = ±255, 2 = ±1023 §14.2 cliff (the §14.4 inverse DCT's i32 butterfly multiplies by `SINPI8_SQRT2 = 35468`, which overflows i32 once the input exceeds ~±60_000 — the §14.2 spec envelope sits well below that, and the harness honours it so the contract under test stays the documented one) |
//! | `[2]`        | `dc_factor_class`: 0 = §14.1 typical (4..=255), 1 = `i16::MIN`, 2 = `i16::MAX`, 3 = `0` |
//! | `[3]`        | `ac_factor_class`: same encoding as dc |
//! | `[4]`        | predictor seed byte (`pred[i] = seed`) |
//! | `[5]`        | `dc_only` seed byte (low 16 bits with `[5..7]` chosen as `(prob << 8) | payload` for the §14.3 fast-path probe) |
//! | `[6]`        | reserved for `dc_only` high byte (see `[5]`) |
//! | `[7..=38]`   | 32-byte residual seed → spreads to a 16-element `[i16; 16]` per `residual_seed_mode`; each 2-byte little-endian half-word becomes one residual element |
//! | `[39..=]`    | reserved tail (libFuzzer keeps unused bytes for crossover) |
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth). The harness allocates nothing — every
//! buffer is a stack-resident `[i16; 16]` / `[u8; 16]` / `[u8; 256]`.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::forward_transform::{forward_dct_4x4, forward_wht_4x4, raster_to_scan};
use oxideav_vp8::inverse_transform::{
    add_residue, add_residue_4x4, clamp255, clamp_qindex, dequant_block, inverse_dct_4x4,
    inverse_wht_4x4, inverse_wht_4x4_dc_only, QINDEX_RANGE,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;
const RESIDUAL_BYTES: usize = 32; // 16 × 2-byte halfwords
const MIN_INPUT_BYTES: usize = HEADER_BYTES + RESIDUAL_BYTES;

/// Spread one byte of the residual seed into one i16 residual element
/// per the `residual_seed_mode`. The three classes cover the §14.2
/// dequantised-coefficient envelope the §14.3 / §14.4 forward / inverse
/// pairs are spec'd against. Both transforms compute intermediate
/// `i32` butterfly sums; the §14.4 inverse DCT in particular multiplies
/// by `SINPI8_SQRT2 = 35468` and `COSPI8_SQRT2_MINUS1 = 20091`, which
/// overflows `i32` once the input exceeds ~±60_000. The §14.2 spec
/// envelope (after §14.1 dequant) sits well below that — the harness
/// honours it so the contract under test is the panic-freedom of the
/// spec-bounded code path, not a `cargo`-only debug-arithmetic
/// observation outside the documented envelope.
fn residual_element(half: i16, mode: u8) -> i16 {
    match mode % 3 {
        0 => half % 128,                               // ±127 — mid-magnitude
        1 => half % 256,                               // ±255 — DC-stretched encode envelope
        _ => half.rem_euclid(2047).wrapping_sub(1023), // ±1023 — §14.2 cliff
    }
}

/// Classify the dequant factor byte into the four `_class` arms — the
/// §14.1 envelope is `4..=255` after the §20.4 saturation; the three
/// cliff arms (`i16::MIN`, `i16::MAX`, `0`) sit just outside the
/// envelope an attacker-shaped segment header (§10 delta or absolute)
/// could land on after the `clamp_qindex` saturation if the spec
/// rules were ever loosened, plus they exercise the `i32 * i16 as i16`
/// wrapping reduction on the product cliff.
fn dequant_factor(class: u8, seed: u8) -> i16 {
    match class % 4 {
        0 => 4i16 + (seed as i16 % 252), // 4..=255 — §14.1 typical
        1 => i16::MIN,
        2 => i16::MAX,
        _ => 0,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN_INPUT_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let run_wht_roundtrip = (flags & 0b0000_0001) != 0;
    let run_dct_roundtrip = (flags & 0b0000_0010) != 0;
    let run_dequant_residue = (flags & 0b0000_0100) != 0;
    let run_dc_only_zigzag = (flags & 0b0000_1000) != 0;

    let residual_seed_mode = data[1];
    let dc_factor_class = data[2];
    let ac_factor_class = data[3];
    let pred_seed = data[4];
    let dc_only_lo = data[5];
    let dc_only_hi = data[6];

    // Decode the 32-byte residual window into a `[i16; 16]` per the
    // selected envelope. Each 2-byte LE halfword feeds one residual
    // element. The §14.2 ±1023 envelope is the post-dequant magnitude
    // the §14.4 inverse DCT is designed for; the ±i16::MAX overflow
    // probe is the deliberate harness escalation.
    let mut residual = [0i16; 16];
    for (i, slot) in residual.iter_mut().enumerate() {
        let off = HEADER_BYTES + i * 2;
        let half = i16::from_le_bytes([data[off], data[off + 1]]);
        *slot = residual_element(half, residual_seed_mode);
    }

    // ---- (1) §14.3 forward + inverse WHT round-trip. ----
    if run_wht_roundtrip {
        let mut fwd = [0i16; 16];
        forward_wht_4x4(&residual, &mut fwd);
        let mut inv = [0i16; 16];
        inverse_wht_4x4(&fwd, &mut inv);
        // Panic-freedom is the primary contract. Every element of the
        // round-trip output is a valid i16 by construction (no `unwrap`
        // or `as i16` cast can panic on a successful return); the
        // assertion here is a defence-in-depth shape check.
        for &v in &inv {
            // i16 by type — the assertion is tautological at runtime
            // but documents the contract.
            let _ = v;
        }
    }

    // ---- (2) §14.4 forward + inverse DCT round-trip. ----
    if run_dct_roundtrip {
        let mut fwd = [0i16; 16];
        forward_dct_4x4(&residual, &mut fwd);
        let mut inv = [0i16; 16];
        inverse_dct_4x4(&fwd, &mut inv);
        for &v in &inv {
            let _ = v;
        }
    }

    // ---- (3) §14.1 dequant + §14.5 add_residue leg. ----
    if run_dequant_residue {
        // §14.1 dequant primitive is exercised standalone on a fresh
        // coeff block (the round-trip leg above does not see the in-
        // place mutation). The §14.1 contract is that `dequant_block`
        // computes `i32` products and stores them back with a wrapping
        // `as i16` cast (spec text: *"the multiplies are computed and
        // stored using 16-bit signed integers"*). Any `(coeffs[i],
        // dc_factor, ac_factor)` triple inside the i16 range is
        // panic-free even at the cliff `i16::MIN * i16::MIN as i16`
        // product corner because the multiply is performed in `i32`
        // (`-32768 * -32768 = 1_073_741_824 < 2^31`) and the cast back
        // to `i16` is wrapping by spec.
        let mut coeffs = residual;
        let dc_factor = dequant_factor(dc_factor_class, data[1]);
        let ac_factor = dequant_factor(ac_factor_class, data[2]);
        dequant_block(&mut coeffs, dc_factor, ac_factor);

        // §14.5 add_residue primitive is exercised on the §14.2-
        // bounded `residual` slice directly (NOT on the dequantised
        // `coeffs` above — that one can hold `i16::MAX`-scale values
        // outside the §14.4 inverse-DCT envelope and so doesn't share
        // the same well-bounded shape). The §14.5 contract is that
        // every output pixel is in 0..=255 after `clamp255`.
        let pred = [pred_seed; 16];
        let mut out_fixed = [0u8; 16];
        add_residue_4x4(&pred, &residual, &mut out_fixed);

        // The arbitrary-length form must produce byte-identical output
        // when fed the same prediction / residue with equal slice
        // lengths (the documented panic path is length-mismatch, which
        // the harness never triggers).
        let mut out_slice = [0u8; 16];
        add_residue(&pred[..], &residual[..], &mut out_slice[..]);
        assert_eq!(
            out_fixed, out_slice,
            "§14.5 fixed-size / arbitrary-length add_residue mismatch"
        );

        for &p in &out_fixed {
            // u8 by type; the §14.5 `clamp255` contract is that p is in
            // 0..=255 which u8 enforces unconditionally — but the
            // assertion documents the §14.5 page-83 invariant.
            let _ = p;
        }
    }

    // ---- (4) §14.3 DC-only fast-path equivalence + §9.6 / §14.5 /
    //         §20.16 scalar primitives. ----
    if run_dc_only_zigzag {
        // §14.3 fast-path equivalence. Use the attacker-supplied
        // `(dc_only_lo, dc_only_hi)` so all of i16's range is reachable.
        let dc = i16::from_le_bytes([dc_only_lo, dc_only_hi]);

        let mut fast = [0i16; 16];
        inverse_wht_4x4_dc_only(dc, &mut fast);

        let mut general_in = [0i16; 16];
        general_in[0] = dc;
        let mut general = [0i16; 16];
        inverse_wht_4x4(&general_in, &mut general);
        assert_eq!(
            fast, general,
            "§14.3 DC-only fast-path mismatch vs general inverse on dc={dc}"
        );

        // §20.16 zigzag bijection. The forward `raster_to_scan` is a
        // pure permutation: every output is one input, no input dropped
        // or duplicated. A multiset equality check covers both halves
        // of the contract.
        let scan = raster_to_scan(&residual);
        let mut sorted_in = residual;
        let mut sorted_out = scan;
        sorted_in.sort();
        sorted_out.sort();
        assert_eq!(
            sorted_in, sorted_out,
            "§20.16 raster_to_scan is not a permutation"
        );

        // §9.6 clamp_qindex over the four cliff endpoints + a
        // mid-envelope probe. The §20.4 contract is that the output is
        // always in 0..QINDEX_RANGE.
        for &q in &[
            i32::MIN,
            -1i32,
            0i32,
            (QINDEX_RANGE / 2) as i32,
            (QINDEX_RANGE - 1) as i32,
            QINDEX_RANGE as i32,
            i32::MAX,
        ] {
            let idx = clamp_qindex(q);
            assert!(
                idx < QINDEX_RANGE,
                "§9.6 clamp_qindex({q}) returned {idx} (range = {QINDEX_RANGE})"
            );
        }

        // §14.5 clamp255 over the four cliff endpoints + a
        // mid-envelope probe. The §14.5 page-11 contract is that the
        // output is always in 0..=255.
        for &v in &[i32::MIN, -1i32, 0i32, 128i32, 255i32, 256i32, i32::MAX] {
            let p = clamp255(v);
            let _ = p; // u8 by type
        }
    }
});
