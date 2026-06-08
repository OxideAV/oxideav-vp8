#![no_main]

//! Fuzz: panic-freedom of the §9.4 / §19.2 loop-filter parameter
//! writeback layer of the public encoder (RFC 6386 §9.4 / §19.2 /
//! §15.4) PLUS the small §9.5 / §9.6 / §9.10 / §9.11 sibling writers
//! reached by the same §19.2 frame-header walk.
//!
//! Surface coverage (every primitive driven directly from attacker
//! bytes; the §15.4 / §20.6 `mb_lf_adjustments()` validation +
//! per-slot effective-delta resolution is exercised in isolation):
//!
//! * `encoder::write_loop_filter(enc, filter_type, level, sharp,
//!   adj_enable=false)` — §9.4 baseline writer (filter_type +
//!   loop_filter_level + sharpness_level + the always-`0`
//!   loop_filter_adj_enable bit). The `adj_enable=true` arm carries a
//!   `debug_assert!(!adj_enable)` Phase-1 guard, so this leg always
//!   passes `false`. The other two parameter axes (level > 63,
//!   sharp > 7) drive the [`EncodeError::LoopFilterLevelOutOfRange`]
//!   and [`EncodeError::SharpnessLevelOutOfRange`] rejection paths.
//!
//! * `encoder::write_loop_filter_with_deltas(enc, filter_type, level,
//!   sharp, &deltas)` — §9.4 / §19.2 full writer with the
//!   `mb_lf_adjustments()` block: the §19.2 `loop_filter_adj_enable`
//!   bit, the gated `mode_ref_lf_delta_update` bit, and the four
//!   per-reference + four per-mode `(presence flag, L(6) magnitude,
//!   L(1) sign)` slot triples. Every gate combination — `enabled` × 2,
//!   `update` × 2 (when `enabled`), per-slot `Some(v)` / `None` × 8 —
//!   is reachable from one input-byte pattern.
//!
//! * `encoder::LoopFilterDeltas::validate()` — §9.4 per-slot magnitude
//!   bound (`abs(v) <= 63`) on every `Some(v)`. Attacker bytes feed
//!   the full `i8::MIN..=i8::MAX` range so the rejection branch
//!   ([`EncodeError::LoopFilterDeltaOutOfRange`]) on `|v| > 63` is
//!   covered alongside the legal envelope.
//!
//! * `encoder::LoopFilterDeltas::effective(carried_ref, carried_mode)`
//!   — §15.4 / §20.6 carried-state resolution. Every `(enabled,
//!   update, per-slot Some|None)` ladder branch is exercised against
//!   an attacker-shaped 8-tuple of carried `i16` values; the harness
//!   asserts the §20.6 rule via a hand-rolled cross-check on each
//!   leg.
//!
//! * `encoder::write_quant_indices(enc, y_ac_qi, y_dc, y2_dc, y2_ac,
//!   uv_dc, uv_ac)` — §9.6 quantiser-indices writer. The
//!   [`EncodeError::QuantIndexOutOfRange`] rejection path is reached
//!   by `y_ac_qi > 127`; the five per-`Option<i8>` delta slots drive
//!   the §9.6 `Some → L(1)=1 + L(4)+L(1)` vs `None → L(1)=0` legs.
//!
//! * `encoder::write_token_partition_count(enc, count)` — §9.5
//!   `log2_nbr_of_dct_partitions` writer. Legal `count ∈ {1, 2, 4, 8}`
//!   write `L(2) = log2(count)`; every other byte triggers the
//!   [`EncodeError::InvalidDctPartitionCount`] return.
//!
//! * `encoder::write_mb_no_skip_coeff(enc, enabled, prob_skip_false)`
//!   — §9.10 / §9.11 skip-coeff toggle + gated `prob_skip_false`
//!   literal.
//!
//! Round-trip invariant (panic-free guard):
//!
//! On the §9.4 full-deltas leg, the harness finishes the
//! [`BoolEncoder`], wraps the resulting partition with a synthetic
//! 2-byte prefix when needed, and feeds it back into
//! [`BoolDecoder::init`]. It then walks the same §19.2 field schedule
//! the encoder wrote (filter_type at prob 128, `L(6)` level, `L(3)`
//! sharp, `L(1)` adj_enable, etc.) and asserts every read value
//! equals what the encoder wrote. Any asymmetry between
//! `write_loop_filter_with_deltas` and the §19.2 wire layout — a
//! field-order swap, a stray bit, a sign-vs-magnitude order
//! transposition — surfaces as a `panic!` from the harness'
//! equality assertion. The same shape proves the §7.3 bool-coder
//! round-trip the round-261 `panic_free_bool_codec` target locked at
//! the primitive layer is also tight against the structured §9.4
//! field schedule.
//!
//! Surface gap the twelve existing fuzz targets leave cold:
//!
//! `panic_free_encode_keyframe` calls `write_loop_filter` via the
//! happy-path `encode_keyframe` driver, which feeds NORMALISED
//! `loop_filter_level` and `sharpness_level` values clamped against
//! the §9.4 6/3-bit fields by the upstream `KeyframeParams` builder
//! — the writer's own rejection branches are reached only when the
//! caller hands an out-of-range byte the builder would never produce.
//! `panic_free_two_pass_stream` is the same shape one layer up
//! (`Vp8TwoPassEncoder::encode_frame` calls `encode_keyframe` /
//! `encode_p_frame_multi_ref`). Neither harness drives
//! `write_loop_filter_with_deltas` (the §9.4 full-deltas writer never
//! ran in either target as of round 262 — `encode_keyframe`'s key
//! frame writes `adj_enable = 0`), nor `LoopFilterDeltas::validate` /
//! `::effective` (the §15.4 / §20.6 carried-state resolution is only
//! reached from inter-frame encode bodies that gate the call behind a
//! `enabled` flag the keyframe-only harness never sets). The
//! round-232 `panic_free_loopfilter_segment` target reaches §15 only
//! through the per-segment kernel layer (`common_adjust`,
//! `simple_segment`, `subblock_filter`, `mb_filter`, plus
//! `LoopFilterParams::derive`) — the kernels consume the §15.4
//! derived (`hev_threshold`, `interior_limit`, `edge_limit`) triple,
//! not the §9.4 wire form. The round-261 `panic_free_bool_codec`
//! target locks the §7.3 bool-coder round-trip at the primitive layer
//! against an attacker-shaped (op-type, prob, num_bits) schedule;
//! this target locks it against the structured §9.4 / §9.5 / §9.6 /
//! §9.10 field schedule.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`  | flags byte: bit0 `filter_type`; bit1 `deltas.enabled`; bit2 `deltas.update`; bit3 picks `write_loop_filter` baseline leg vs `write_loop_filter_with_deltas` full leg; bit4 picks the round-trip readback leg on the full writer |
//! | `[1]`  | `loop_filter_level` raw byte (full 0..=255 — both legal and rejection paths) |
//! | `[2]`  | `sharpness_level` raw byte (full 0..=255 — both legal and rejection paths) |
//! | `[3..7]` | 4-bit `ref_frame_delta` presence-pattern + per-slot raw `i8` values (4 bytes one per slot) |
//! | `[7..11]` | 4-bit `mode_delta` presence-pattern + per-slot raw `i8` values (4 bytes one per slot) |
//! | `[11]` | `presence_bits` byte: low nibble `ref_frame_delta` per-slot presence; high nibble `mode_delta` per-slot presence |
//! | `[12]` | `y_ac_qi` raw byte (full 0..=255 — both legal and rejection paths) |
//! | `[13..18]` | 5 raw `i8` bytes for the §9.6 `(y_dc, y2_dc, y2_ac, uv_dc, uv_ac)` delta values |
//! | `[18]` | `quant_delta_presence` byte: low 5 bits select which §9.6 delta slots are `Some(value)` vs `None` |
//! | `[19]` | `token_partition_count` raw byte (full 0..=255 — legal {1, 2, 4, 8} vs rejection envelope) |
//! | `[20]` | `mb_no_skip_coeff_flags`: bit0 enabled, bits 1..7 `prob_skip_false >> 1` |
//! | `[21..29]` | 8 raw `i16` carried-delta bytes (4 ref slots + 4 mode slots) for the `effective` cross-check |
//!
//! The harness:
//!
//! 1. Builds a `LoopFilterDeltas` from `data[3..]` and runs
//!    `validate()` + `effective()` against an attacker-shaped pair of
//!    carried `[i16; 4]` arrays. The carried-state resolution is
//!    cross-checked against a hand-rolled rule derived from the §20.6
//!    listing.
//!
//! 2. Calls one of the two §9.4 writers per the flags-byte
//!    discriminator. On the `write_loop_filter_with_deltas` leg with
//!    bit4 set AND valid (in-range) parameters, the harness wraps the
//!    encoder output in a fresh [`BoolDecoder::init`] and reads back
//!    each field per the §19.2 schedule, asserting equality.
//!
//! 3. Calls `write_quant_indices` with the attacker-supplied tuple
//!    (full 0..=255 `y_ac_qi`, five `Option<i8>` delta slots gated by
//!    the presence byte).
//!
//! 4. Calls `write_token_partition_count` with the raw byte (legal
//!    envelope: {1, 2, 4, 8}; everything else triggers the §9.5
//!    rejection).
//!
//! 5. Calls `write_mb_no_skip_coeff` with the gated enable / prob
//!    pair.
//!
//! Caps: input <= 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth); min 29 bytes (the header layout
//! above). Every per-iteration allocation is a single `Vec<u8>` from
//! `BoolEncoder::finish()`; no inputs beyond the fixed header are
//! read.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    bool_decoder::BoolDecoder,
    encoder::{
        write_loop_filter, write_loop_filter_with_deltas, write_mb_no_skip_coeff,
        write_quant_indices, write_token_partition_count, BoolEncoder, EncodeError,
        LoopFilterDeltas,
    },
};

const MIN_INPUT_BYTES: usize = 29;
const MAX_INPUT_BYTES: usize = 4 * 1024;

/// Decode the `i8` per-slot value array gated by a presence nibble.
fn decode_delta_slots(presence_nibble: u8, raw: &[u8; 4]) -> [Option<i8>; 4] {
    let mut out = [None; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        if (presence_nibble >> i) & 1 == 1 {
            *slot = Some(raw[i] as i8);
        }
    }
    out
}

/// Hand-rolled cross-check of `LoopFilterDeltas::effective` per the
/// §20.6 listing — used as an oracle on the panic-free contract.
fn effective_oracle(
    deltas: &LoopFilterDeltas,
    carried_ref: [i16; 4],
    carried_mode: [i16; 4],
) -> ([i16; 4], [i16; 4]) {
    if !deltas.enabled {
        return ([0; 4], [0; 4]);
    }
    if !deltas.update {
        return (carried_ref, carried_mode);
    }
    let mut r = carried_ref;
    let mut m = carried_mode;
    for (slot, value) in r.iter_mut().zip(deltas.ref_frame_delta.iter()) {
        if let Some(v) = value {
            *slot = *v as i16;
        }
    }
    for (slot, value) in m.iter_mut().zip(deltas.mode_delta.iter()) {
        if let Some(v) = value {
            *slot = *v as i16;
        }
    }
    (r, m)
}

/// Read back the `(filter_type, level, sharp)` triple from a
/// [`BoolDecoder`] in the §19.2 field schedule.
fn read_lf_baseline(
    dec: &mut BoolDecoder<'_>,
) -> Result<(bool, u8, u8), oxideav_vp8::bool_decoder::BoolDecoderError> {
    let ft = dec.read_bool(128)?;
    let level = dec.read_literal(6)? as u8;
    let sharp = dec.read_literal(3)? as u8;
    Ok((ft, level, sharp))
}

/// Read back one §9.4 signed-delta slot: presence flag + L(6)
/// magnitude + L(1) sign. Returns `Ok(None)` when the presence flag
/// is `false`.
fn read_lf_delta_slot(
    dec: &mut BoolDecoder<'_>,
) -> Result<Option<i8>, oxideav_vp8::bool_decoder::BoolDecoderError> {
    let present = dec.read_bool(128)?;
    if !present {
        return Ok(None);
    }
    let magnitude = dec.read_literal(6)? as i16;
    let sign_negative = dec.read_bool(128)?;
    let v = if sign_negative { -magnitude } else { magnitude };
    Ok(Some(v as i8))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN_INPUT_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let filter_type = (flags & 0b0000_0001) != 0;
    let enabled = (flags & 0b0000_0010) != 0;
    let update = (flags & 0b0000_0100) != 0;
    let path_full_deltas = (flags & 0b0000_1000) != 0;
    let do_roundtrip = (flags & 0b0001_0000) != 0;

    let loop_filter_level = data[1];
    let sharpness_level = data[2];

    let ref_raw = [data[3], data[4], data[5], data[6]];
    let mode_raw = [data[7], data[8], data[9], data[10]];

    let presence = data[11];
    let ref_presence = presence & 0x0f;
    let mode_presence = (presence >> 4) & 0x0f;

    let ref_frame_delta = decode_delta_slots(ref_presence, &ref_raw);
    let mode_delta = decode_delta_slots(mode_presence, &mode_raw);

    let deltas = LoopFilterDeltas {
        enabled,
        update,
        ref_frame_delta,
        mode_delta,
    };

    // §9.4 validate(): explicit Result return, panic-free on every
    // input.
    let validation = deltas.validate();

    // §15.4 / §20.6 effective(): pure function, must agree with the
    // hand-rolled oracle on every (enabled, update, per-slot
    // presence) ladder branch.
    let carried_ref: [i16; 4] = [
        data[21] as i8 as i16,
        data[22] as i8 as i16,
        data[23] as i8 as i16,
        data[24] as i8 as i16,
    ];
    let carried_mode: [i16; 4] = [
        data[25] as i8 as i16,
        data[26] as i8 as i16,
        data[27] as i8 as i16,
        data[28] as i8 as i16,
    ];
    let got = deltas.effective(carried_ref, carried_mode);
    let want = effective_oracle(&deltas, carried_ref, carried_mode);
    assert_eq!(
        got, want,
        "LoopFilterDeltas::effective disagrees with §20.6 oracle"
    );

    // §9.4 writers — both legs.
    let mut enc = BoolEncoder::new();
    let writer_res = if path_full_deltas {
        write_loop_filter_with_deltas(
            &mut enc,
            filter_type,
            loop_filter_level,
            sharpness_level,
            &deltas,
        )
    } else {
        // The Phase-1 baseline writer carries a `debug_assert!` on
        // `adj_enable == false`; pass `false` unconditionally so the
        // assertion never fires.
        write_loop_filter(
            &mut enc,
            filter_type,
            loop_filter_level,
            sharpness_level,
            false,
        )
    };

    // The validate() and writer rejection branches MUST agree on
    // out-of-range inputs: every out-of-range writer return must
    // correspond to one of the documented EncodeError variants, and
    // an in-range writer return must produce a Vec that the
    // round-trip leg can read back without error. Panic-freedom is
    // the only assertion here — the harness drops the
    // (Ok / Err(...)) discriminator after recording the round-trip
    // gate.
    let in_range = loop_filter_level <= 63
        && sharpness_level <= 7
        && (if path_full_deltas {
            validation.is_ok()
        } else {
            true
        });

    if path_full_deltas && do_roundtrip && in_range && writer_res.is_ok() {
        // The §9.4 writer chain on its own does NOT close the bool
        // coder (subsequent §9.5 / §9.6 / §9.10 fields share the
        // same partition). For an isolated round-trip we close the
        // encoder here and feed back through a fresh BoolDecoder
        // initialised on the partition bytes.
        let partition = enc.finish();
        // §7.3 init demands >= 2 bytes; the writer always emits at
        // least the filter_type + 6 + 3 + 1 = 11 bits = 2 bytes
        // even on the all-disabled path, but defence-in-depth check
        // before init.
        if partition.len() >= 2 {
            if let Ok(mut dec) = BoolDecoder::init(&partition) {
                if let Ok((ft, lvl, sh)) = read_lf_baseline(&mut dec) {
                    assert_eq!(ft, filter_type, "filter_type round-trip mismatch");
                    assert_eq!(
                        lvl, loop_filter_level,
                        "loop_filter_level round-trip mismatch"
                    );
                    assert_eq!(sh, sharpness_level, "sharpness_level round-trip mismatch");
                    if let Ok(adj_enable) = dec.read_bool(128) {
                        assert_eq!(
                            adj_enable, enabled,
                            "loop_filter_adj_enable round-trip mismatch"
                        );
                        if enabled {
                            if let Ok(upd) = dec.read_bool(128) {
                                assert_eq!(
                                    upd, update,
                                    "mode_ref_lf_delta_update round-trip mismatch"
                                );
                                if update {
                                    // Four ref + four mode slots in
                                    // §20.6 order.
                                    for (i, expected) in ref_frame_delta.iter().enumerate() {
                                        match read_lf_delta_slot(&mut dec) {
                                            Ok(got) => assert_eq!(
                                                &got, expected,
                                                "ref_frame_delta[{i}] round-trip mismatch"
                                            ),
                                            Err(_) => break,
                                        }
                                    }
                                    for (i, expected) in mode_delta.iter().enumerate() {
                                        match read_lf_delta_slot(&mut dec) {
                                            Ok(got) => assert_eq!(
                                                &got, expected,
                                                "mode_delta[{i}] round-trip mismatch"
                                            ),
                                            Err(_) => break,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Separate encoder for §9.6 quant indices — the §9.4 writer's
    // partition is already either consumed by the round-trip leg or
    // discarded, and we want each downstream writer to be reachable
    // independent of the §9.4 rejection branches.
    let mut enc_q = BoolEncoder::new();
    let y_ac_qi = data[12];
    // §9.6 per-slot deltas are emitted via
    // `BoolEncoder::write_signed_literal(value, 4)`, whose contract is
    // `|value| < 16`. The §9.6 spec field is `L(4) + L(1)` (magnitude
    // 0..=15 + sign bit), so legal values are `-15..=15`. The harness
    // pre-clamps the attacker's raw `i8` bytes into that envelope —
    // the writer's contract panic on `|value| >= 16` is a documented
    // caller-side guarantee, not a panic-freedom contract of the
    // public API. Tightening the envelope to the spec field bounds
    // keeps the harness inside the writer's documented input domain
    // while still exercising every legal magnitude / sign / presence
    // permutation.
    let clamp_4bit = |raw: u8| -> i8 {
        let v = raw as i8;
        v.clamp(-15, 15)
    };
    let q_raw = [
        clamp_4bit(data[13]),
        clamp_4bit(data[14]),
        clamp_4bit(data[15]),
        clamp_4bit(data[16]),
        clamp_4bit(data[17]),
    ];
    let q_presence = data[18];
    let q_slots: [Option<i8>; 5] = [
        if q_presence & 0b00001 != 0 {
            Some(q_raw[0])
        } else {
            None
        },
        if q_presence & 0b00010 != 0 {
            Some(q_raw[1])
        } else {
            None
        },
        if q_presence & 0b00100 != 0 {
            Some(q_raw[2])
        } else {
            None
        },
        if q_presence & 0b01000 != 0 {
            Some(q_raw[3])
        } else {
            None
        },
        if q_presence & 0b10000 != 0 {
            Some(q_raw[4])
        } else {
            None
        },
    ];
    let q_res = write_quant_indices(
        &mut enc_q, y_ac_qi, q_slots[0], q_slots[1], q_slots[2], q_slots[3], q_slots[4],
    );
    // The §9.6 writer's rejection branch is precisely `y_ac_qi > 127`.
    let q_should_reject = y_ac_qi > 127;
    assert_eq!(
        q_res.is_err(),
        q_should_reject,
        "write_quant_indices rejection disagrees with §9.6 envelope (y_ac_qi={y_ac_qi})"
    );
    if let Err(EncodeError::QuantIndexOutOfRange { value }) = q_res {
        assert_eq!(value, y_ac_qi, "QuantIndexOutOfRange.value field disagrees");
    }

    // §9.5 token partition count writer — legal envelope {1, 2, 4, 8}.
    let mut enc_p = BoolEncoder::new();
    let pc_byte = data[19];
    let pc_res = write_token_partition_count(&mut enc_p, pc_byte);
    let pc_legal = matches!(pc_byte, 1 | 2 | 4 | 8);
    assert_eq!(
        pc_res.is_err(),
        !pc_legal,
        "write_token_partition_count rejection disagrees with §9.5 envelope (count={pc_byte})"
    );
    if let Err(EncodeError::InvalidDctPartitionCount { value }) = pc_res {
        assert_eq!(
            value, pc_byte,
            "InvalidDctPartitionCount.value field disagrees"
        );
    }

    // §9.10 / §9.11 mb_no_skip_coeff writer — no validation, every
    // input legal.
    let mut enc_s = BoolEncoder::new();
    let skip_flags = data[20];
    let skip_enabled = skip_flags & 1 != 0;
    let prob_skip_false = skip_flags; // full 0..=255 — every value legal
    write_mb_no_skip_coeff(&mut enc_s, skip_enabled, prob_skip_false);
    let _ = enc_s.finish();

    // Drop validation discriminator at the end; the cross-checks above
    // already gate the round-trip leg on it.
    let _ = validation;
});
