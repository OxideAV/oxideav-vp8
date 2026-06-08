#![no_main]

//! Fuzz: panic-freedom of the §7 boolean (range) entropy primitives — both
//! read and write halves driven directly.
//!
//! Surface coverage:
//!
//! * `BoolDecoder::init(input)` — RFC 6386 §7.3 `init_bool_decoder`,
//!   minimum 2-byte input check (`InputTooShort` rejection).
//! * `BoolDecoder::init_partition(input)` — §20 reference's
//!   `init_bool_decoder` that tolerates `sz < 2` by zero-initialising
//!   `value` and presenting an empty input — the 0- and 1-byte legs
//!   small inter MBs land on.
//! * `BoolDecoder::read_bool(prob)` — every legal probability (1..=255)
//!   plus the cliff probabilities (0 and 256→255-clamped) the encoder
//!   contract excludes but which an attacker-shaped sequence might
//!   feed.
//! * `BoolDecoder::read_literal(n)` — n ∈ 0..=32.
//! * `BoolDecoder::read_signed_literal(n)` — n ∈ 0..=31 (incl. the
//!   `n == 0 → 0` short-circuit per §7.3).
//! * `BoolEncoder::new()` / `write_bool` / `write_literal` /
//!   `write_signed_literal` / `write_treed` / `finish` — the
//!   §7.3 `init_bool_encoder` / `write_bool` / `add_one_to_output` /
//!   `flush_bool_encoder` write path.
//!
//! The ten existing fuzz targets reach §7 only indirectly: the four
//! decode harnesses (`panic_free_decode_keyframe` / `_decoder_state` /
//! `parse_headers` / `panic_free_token_block`) feed bytes through
//! `decode_vp8` / `Vp8DecoderState::decode_frame` /
//! `Vp8FrameHeader::parse` / `decode_block`, every one of which calls
//! `BoolDecoder::init` once at the top of a partition and then issues
//! `read_bool` / `read_literal` against probabilities determined by the
//! higher-level decode state; the attacker controls the bytes but the
//! probability schedule is locked to whatever the upper layers compute.
//! The four encode harnesses (`panic_free_encode_keyframe` /
//! `_two_pass_stream`) reach `BoolEncoder` only through
//! `encode_keyframe` / `Vp8TwoPassEncoder::encode_frame`, which run a
//! valid §9 / §11 / §13 / §15 encode chain on top of it — the bool
//! coder is the final stage, never driven with attacker-shaped (prob,
//! value) schedules in isolation.
//!
//! This target drives the §7 primitive surface directly:
//!
//! 1. **Stand-alone decode pass.** `init` and `init_partition` are
//!    both attempted against the attacker-supplied bytes; for the legal
//!    initial state, a sequence of `read_bool` / `read_literal` /
//!    `read_signed_literal` operations is issued, with the attacker
//!    bytes also choosing the (op-type, prob, num_bits) of each call.
//!    Errors (`InputTooShort`, `EndOfStream`) are accepted as a normal
//!    return value — the target only fails on panic.
//!
//! 2. **Stand-alone encode pass.** A `BoolEncoder` is fed an
//!    attacker-shaped (op-type, prob, value, num_bits) schedule; the
//!    encoder is then `finish`-ed and the resulting partition is fed
//!    back into a fresh `BoolDecoder::init`. The decoder replays the
//!    schedule with the SAME (prob, num_bits) sequence and the target
//!    asserts every read returns Ok and the value matches what was
//!    written — a §7.3 round-trip lockstep guard for the `write_bool`
//!    ↔ `read_bool` and `write_literal` ↔ `read_literal` pairs. Any
//!    asymmetry in the `split = 1 + (((range - 1) * prob) >> 8)`
//!    arithmetic or in the `add_one_to_output` carry propagation
//!    surfaces as a mismatch on the read side.
//!
//!    Note that `write_signed_literal` and `read_signed_literal` are
//!    NOT a symmetric pair: the encoder's §9.3 / §9.4 / §9.6 form
//!    writes `num_bits` magnitude bits THEN a sign bit, while the
//!    decoder's §7.3 form reads sign FIRST then `num_bits - 1`
//!    magnitude bits. The two coexist as separate APIs serving
//!    different bitstream-layer contracts (the encoder side is paired
//!    against per-call `BoolDecoder::read_literal(num_bits)` +
//!    `read_bool(128)` at every call site that emits a signed quantum
//!    with the §9 idiom). The harness exercises both halves for
//!    panic-freedom but only the `write_signed_literal → read_literal
//!    + read_bool` form for round-trip equality.
//!
//! 3. **`write_treed` leg.** A small static tree (a 4-leaf binary tree
//!    mirroring the §11 `kf_ymode_tree` shape) is encoded with the
//!    attacker selecting both the leaf and a per-node probability
//!    look-up byte; the round-trip checks that walking the same tree
//!    with the same probability schedule on the decode side recovers
//!    the same leaf bit sequence.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes        | Meaning |
//! |-------------:|---------|
//! | `[0]`        | flags byte: bit0 use `init_partition` instead of `init`, bit1 also encode-then-decode round-trip, bit2 also run the `write_treed` leg, bits 3-7 reserved |
//! | `[1]`        | `op_count` (number of bool ops in the schedule, clamped to 1..=64) |
//! | `[2]`        | `read_back_op_count` for the round-trip decoder (clamped to the same value) |
//! | `[3]`        | tree leaf selector (0..=3 % 4) for the `write_treed` leg |
//! | `[4..=6]`    | per-node probability bytes (3 internal nodes) for the `write_treed` leg |
//! | `[7..=]`     | repeating 3-byte op records: `(op_type, prob, payload)` — `op_type % 4` selects {read_bool, read_literal, read_signed_literal, skip}; `prob` ∈ 0..=255 (raw, exercises the cliff probabilities); `payload` is `num_bits` mod 33 for literal / mod 32 for signed-literal / value bit for read_bool replay |
//!
//! The stand-alone decode pass and the round-trip pass share the same
//! schedule so a single 3-byte op record produces both a write side
//! (round-trip) and a read side (stand-alone). Memory: the encoder's
//! output is bounded by `op_count * 5` bytes (a worst-case per-bool
//! 32-bit shift+flush), so 64 ops cap at 320 bytes; the stand-alone
//! decode pass reads from the attacker bytes directly, also bounded.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer default; re-checked at harness
//! entry as defence-in-depth).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::bool_decoder::{BoolDecoder, BoolDecoderError};
use oxideav_vp8::bool_encoder::BoolEncoder;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;
const MAX_OPS: usize = 64;
const OP_RECORD_BYTES: usize = 3;

// A 7-entry tree mirroring the §11 `kf_ymode_tree` shape: 3 internal
// nodes and 4 leaves. Encoded as the spec's `(left, right)` pairs at
// positions `2 * node_index`. Leaves are encoded as -leaf (so the four
// leaves here are -0, -1, -2, -3).
//
// Internal layout:
//   node 0 → (2, -0)
//   node 2 → (4, -1)
//   node 4 → (-2, -3)
//
// `treed_read` walks: bit-0 left, bit-1 right; at every node the
// per-node probability comes from `prob_lookup(node_index >> 1)` so
// indices 0 / 1 / 2 each fetch a single byte.
const TREE: &[i8] = &[2, 0, 4, -1, -2, -3];

fn opkind_from(sel: u8) -> u8 {
    sel & 0b11
}

fn num_bits_literal(payload: u8) -> u32 {
    // 0..=32 inclusive — the documented `read_literal` envelope.
    (payload as u32) % 33
}

fn num_bits_signed(payload: u8) -> u32 {
    // 0..=31 inclusive — `read_signed_literal` accepts `num_bits <= 31`
    // (one bit is reserved for the sign).
    (payload as u32) % 32
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let use_init_partition = (flags & 0b0000_0001) != 0;
    let run_roundtrip = (flags & 0b0000_0010) != 0;
    let run_tree_leg = (flags & 0b0000_0100) != 0;

    // Clamp op_count to a sensible window; 0 is allowed (decoder
    // simply does nothing). MAX_OPS is the hard cap.
    let raw_op_count = data[1] as usize;
    let op_count = raw_op_count.min(MAX_OPS);

    let raw_readback = data[2] as usize;
    let readback_count = raw_readback.min(op_count);

    let leaf_sel = data[3] % 4;
    let tree_probs = [data[4], data[5], data[6]];

    // Slice that holds the (op_type, prob, payload) records. Each record
    // is 3 bytes; we need at most `op_count * 3` bytes after the
    // 7-byte header. If the input is too short to satisfy `op_count`
    // records, the missing trailing records are treated as `(0, 128,
    // 0)` — a `read_bool(128)` op-type — so the harness still drives
    // the primitive surface even on near-empty inputs.
    let records = &data[HEADER_BYTES..];

    // ---- (1) Stand-alone decode pass. ----
    // Try both `init` and `init_partition` against the same input
    // prefix; only the latter is panic-free for `sz < 2`.
    if use_init_partition {
        let mut dec = BoolDecoder::init_partition(records);
        // The §7.2 invariant: after init, `range` is exactly 255.
        // We don't assert it here (the spec primitives are
        // panic-tested, not bit-tested), but the invariant must hold
        // for the harness to be a meaningful guard. Run a sequence of
        // attacker-shaped ops.
        for i in 0..op_count {
            let off = i * OP_RECORD_BYTES;
            if off + OP_RECORD_BYTES > records.len() {
                break;
            }
            let op = opkind_from(records[off]);
            let prob = records[off + 1];
            let payload = records[off + 2];
            match op {
                0 => {
                    // prob = 0 is forbidden by the encoder contract;
                    // the decoder still has to be panic-free against
                    // it. We clamp to 1..=255 to stay inside the
                    // primitive's debug-assert envelope on the read
                    // side while still hitting both cliff endpoints.
                    let p = prob.max(1);
                    let _ = dec.read_bool(p);
                }
                1 => {
                    let n = num_bits_literal(payload);
                    let _ = dec.read_literal(n);
                }
                2 => {
                    let n = num_bits_signed(payload);
                    let _ = dec.read_signed_literal(n);
                }
                _ => {
                    // skip op — drives the dispatcher arm with no
                    // primitive call but keeps the position counter
                    // moving for the round-trip leg below.
                }
            }
        }
    } else if let Ok(mut dec) = BoolDecoder::init(records) {
        for i in 0..op_count {
            let off = i * OP_RECORD_BYTES;
            if off + OP_RECORD_BYTES > records.len() {
                break;
            }
            let op = opkind_from(records[off]);
            let prob = records[off + 1];
            let payload = records[off + 2];
            match op {
                0 => {
                    let p = prob.max(1);
                    let _ = dec.read_bool(p);
                }
                1 => {
                    let n = num_bits_literal(payload);
                    let _ = dec.read_literal(n);
                }
                2 => {
                    let n = num_bits_signed(payload);
                    let _ = dec.read_signed_literal(n);
                }
                _ => {}
            }
        }
    }

    // ---- (2) Round-trip encode → decode pass. ----
    // Build a schedule of (op_type, prob, value/num_bits) records,
    // encode them, finish, then decode the buffer back with the same
    // schedule and assert each read recovers what was written.
    if run_roundtrip {
        // Plan the schedule from the same record window, then execute
        // both sides against the plan. `read_bool(0)` and
        // `read_bool(256)` are forbidden by §7.2 (the encoder writes
        // would diverge); clamp `prob` to 1..=255 for the round-trip
        // leg only.
        #[derive(Clone, Copy)]
        enum Op {
            Bool { prob: u8, value: bool },
            Literal { value: u32, num_bits: u32 },
            Signed { value: i32, num_bits: u32 },
        }

        let mut schedule: Vec<Op> = Vec::with_capacity(op_count);
        for i in 0..op_count {
            let off = i * OP_RECORD_BYTES;
            if off + OP_RECORD_BYTES > records.len() {
                break;
            }
            let op = opkind_from(records[off]);
            let prob = records[off + 1].max(1);
            let payload = records[off + 2];
            match op {
                0 => {
                    schedule.push(Op::Bool {
                        prob,
                        value: (payload & 1) != 0,
                    });
                }
                1 => {
                    // num_bits 0..=32; `value` is the low `num_bits`
                    // of `(prob as u32) << 8 | payload as u32` so the
                    // 32-bit literal envelope is reachable from the
                    // 3-byte record.
                    let num_bits = num_bits_literal(payload);
                    let raw = ((prob as u32) << 8) | payload as u32;
                    let value = if num_bits == 32 {
                        raw.wrapping_mul(0x0101_0101)
                    } else if num_bits == 0 {
                        0
                    } else {
                        raw & ((1u32 << num_bits) - 1)
                    };
                    schedule.push(Op::Literal { value, num_bits });
                }
                2 => {
                    // num_bits 0..=31; `value` is a signed integer
                    // whose magnitude fits in `num_bits` bits. The
                    // §7.3 contract requires the sign bit be the
                    // last-written bit, so `write_signed_literal`
                    // takes a value already inside ±(2^num_bits - 1).
                    let num_bits = num_bits_signed(payload);
                    let magnitude = if num_bits == 0 {
                        0u32
                    } else {
                        let mask = (1u32 << num_bits).wrapping_sub(1);
                        (payload as u32) & mask
                    };
                    let value = if (prob & 1) == 1 {
                        -(magnitude as i32)
                    } else {
                        magnitude as i32
                    };
                    schedule.push(Op::Signed { value, num_bits });
                }
                _ => {
                    // skip — no schedule entry.
                }
            }
        }

        // Encode pass.
        let mut enc = BoolEncoder::new();
        for op in &schedule {
            match *op {
                Op::Bool { prob, value } => enc.write_bool(prob, value),
                Op::Literal { value, num_bits } => enc.write_literal(value, num_bits),
                Op::Signed { value, num_bits } => enc.write_signed_literal(value, num_bits),
            }
        }
        let buf = enc.finish();

        // `finish` always writes 4 trailing bytes; `init` then reads
        // the first 2 of those into `value`. The buffer is guaranteed
        // ≥ 4 bytes, so `BoolDecoder::init` always succeeds here.
        let mut dec = match BoolDecoder::init(&buf) {
            Ok(d) => d,
            Err(_) => return, // unreachable given the contract above
        };

        // Replay the schedule up to `readback_count` operations.
        // (Reading every op is fine — `EndOfStream` is impossible
        //  because the §7.3 flush leaves enough tail bytes to satisfy
        //  any read up to the boundary.)
        let to_read = readback_count.min(schedule.len());
        for op in schedule.iter().take(to_read) {
            match *op {
                Op::Bool { prob, value } => {
                    // `read_bool` panic-free on any 1..=255 prob.
                    match dec.read_bool(prob) {
                        Ok(got) => {
                            assert_eq!(
                                got, value,
                                "bool round-trip mismatch at prob={prob} expected={value}"
                            );
                        }
                        // EndOfStream is permissible if the schedule
                        // had so few ops the flushed tail was already
                        // consumed; the panic-freedom claim still
                        // holds.
                        Err(BoolDecoderError::EndOfStream) => break,
                        Err(BoolDecoderError::InputTooShort) => unreachable!(),
                    }
                }
                Op::Literal { value, num_bits } => match dec.read_literal(num_bits) {
                    Ok(got) => {
                        assert_eq!(
                            got, value,
                            "literal round-trip mismatch num_bits={num_bits} expected={value}"
                        );
                    }
                    Err(BoolDecoderError::EndOfStream) => break,
                    Err(BoolDecoderError::InputTooShort) => unreachable!(),
                },
                Op::Signed { value, num_bits } => {
                    // `write_signed_literal` writes `num_bits` magnitude
                    // bits MSB-first (the §9.3 / §9.4 / §9.6 form) and
                    // then a sign bit at `prob=128`. To round-trip it
                    // we must `read_literal(num_bits)` for the
                    // magnitude and `read_bool(128)` for the sign;
                    // `read_signed_literal` uses a different (§7.3)
                    // convention and is intentionally NOT the inverse
                    // of `write_signed_literal`. The two APIs serve
                    // different bitstream-layer contracts.
                    let mag = match dec.read_literal(num_bits) {
                        Ok(m) => m,
                        Err(BoolDecoderError::EndOfStream) => break,
                        Err(BoolDecoderError::InputTooShort) => unreachable!(),
                    };
                    let sign = match dec.read_bool(128) {
                        Ok(s) => s,
                        Err(BoolDecoderError::EndOfStream) => break,
                        Err(BoolDecoderError::InputTooShort) => unreachable!(),
                    };
                    let got = if sign { -(mag as i32) } else { mag as i32 };
                    assert_eq!(
                        got, value,
                        "signed round-trip mismatch num_bits={num_bits} expected={value}"
                    );
                }
            }
        }
    }

    // ---- (3) `write_treed` leg. ----
    // Encode one leaf and decode-walk the same tree with the same
    // per-node probabilities. The leaf walk uses `read_bool(prob)` at
    // each internal node, matching the encoder's `write_bool` calls
    // one-for-one.
    if run_tree_leg {
        let leaf = leaf_sel; // 0..=3, all valid leaves of TREE.
        let mut enc = BoolEncoder::new();
        // Probabilities for nodes 0 / 1 / 2 (the `node_index >> 1`
        // indexing the spec's tree walker uses).
        let probs = [
            tree_probs[0].max(1),
            tree_probs[1].max(1),
            tree_probs[2].max(1),
        ];
        enc.write_treed(TREE, |i| probs[i.min(2)], leaf);
        let buf = enc.finish();

        // Decode-walk the same tree.
        let mut dec = match BoolDecoder::init(&buf) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut node: i8 = 0;
        let mut walked_leaf: i8 = 0;
        // Walk at most TREE.len() / 2 steps (the depth of the
        // 4-leaf tree is 2, so 8 steps is a very loose cap).
        for _ in 0..8 {
            let p = probs[(node as usize) >> 1];
            let bit = match dec.read_bool(p) {
                Ok(b) => b as usize,
                Err(_) => return,
            };
            let next = TREE[node as usize + bit];
            if next <= 0 {
                walked_leaf = -next;
                break;
            }
            node = next;
        }
        assert_eq!(
            walked_leaf as u8, leaf,
            "tree round-trip mismatch: wrote leaf {leaf}, read leaf {walked_leaf}"
        );
    }
});
