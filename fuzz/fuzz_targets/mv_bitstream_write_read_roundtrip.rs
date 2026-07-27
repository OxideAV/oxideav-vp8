#![no_main]

//! Fuzz: the §17 motion-vector bitstream primitives under
//! attacker-controlled probability tables — [`write_mv`] /
//! [`read_mv`] exact roundtrip plus the [`mv_bits`] costing mirror.
//!
//! The decode side of §17 is reached indirectly today (the stateful
//! decoder target feeds real interframes through `read_mv`, and
//! `panic_free_near_mv_mode_decode` walks the §16 layer above it),
//! but always against *resolved* probability tables that start from
//! [`DEFAULT_MV_CONTEXT`] and mutate only through the §17.2 F-gated
//! update path — so degenerate per-position probabilities (0, 255)
//! never reach the component codec from those harnesses. The encode
//! side ([`write_mv`] / [`write_mv_component`]) and the costing
//! mirror ([`mv_bits`]) had no fuzz coverage at all.
//!
//! Three oracles per iteration, all against a fully fuzz-chosen
//! `MvContexts` (38 raw probability bytes, degenerate values
//! included):
//!
//! 1. **Exact roundtrip.** Up to 32 fuzz-chosen vectors (components
//!    mapped into the §17 legal `-1023..=1023`) are written with
//!    [`write_mv`] into one §7.3 boolean stream; decoding the stream
//!    with [`read_mv`] against the same contexts must reproduce every
//!    vector exactly, in order. Any drift is a codec asymmetry
//!    finding.
//! 2. **Costing mirror.** For every vector, [`mv_bits`] must be
//!    finite and non-negative (it mirrors the writer's §17.1 control
//!    flow line-for-line, so a NaN / infinity means the mirror
//!    diverged into an impossible branch).
//! 3. **Range envelope on arbitrary bytes.** The input tail is
//!    treated as a raw boolean stream and pumped through [`read_mv`]
//!    against the same contexts until exhaustion (≤ 64 vectors);
//!    every decoded component must stay inside `-1023..=1023` — §17.1
//!    guarantees the long form cannot exceed 10 bits plus the
//!    implicit bit-3 rule, so an escape is a decoder bug.
//!
//! Input layout (consumed front-to-back):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0..19]`  | row [`MvContext`] — 19 raw probability bytes |
//! | `[19..38]` | column [`MvContext`] — 19 raw probability bytes |
//! | `[38]`     | vector count `1 + (b % 32)` |
//! | `[39..39+4n]` | `n` vectors: row/col LE u16 each → `(v % 2047) - 1023` |
//! | rest       | raw boolean stream for the decode-only leg |
//!
//! Hard input cap 4 KiB; every leg is allocation-light (one encoded
//! `Vec` per iteration), so wall time stays negligible.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::bool_encoder::BoolEncoder;
use oxideav_vp8::{mv_bits, read_mv, write_mv, BoolDecoder, Mv, MvContexts};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const CTX_BYTES: usize = 38;
const MAX_VECTORS: usize = 32;
const MAX_DECODE_ONLY_VECTORS: usize = 64;

fn component(lo: u8, hi: u8) -> i16 {
    let v = u16::from_le_bytes([lo, hi]);
    (i32::from(v % 2047) - 1023) as i16
}

fuzz_target!(|data: &[u8]| {
    if data.len() < CTX_BYTES + 1 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mut contexts: MvContexts = [[0u8; 19]; 2];
    contexts[0].copy_from_slice(&data[0..19]);
    contexts[1].copy_from_slice(&data[19..38]);

    let n = 1 + usize::from(data[CTX_BYTES]) % MAX_VECTORS;
    let vec_bytes = &data[CTX_BYTES + 1..];
    let n = n.min(vec_bytes.len() / 4);

    // Leg 1 + 2 — write every vector, price every vector.
    let mut mvs = Vec::with_capacity(n);
    let mut enc = BoolEncoder::new();
    for i in 0..n {
        let mv = Mv {
            row: component(vec_bytes[4 * i], vec_bytes[4 * i + 1]),
            col: component(vec_bytes[4 * i + 2], vec_bytes[4 * i + 3]),
        };
        write_mv(&mut enc, &contexts, mv);
        let bits = mv_bits(&contexts, mv);
        assert!(
            bits.is_finite() && bits >= 0.0,
            "mv_bits({mv:?}) produced a non-finite / negative cost: {bits}"
        );
        mvs.push(mv);
    }
    let stream = enc.finish();

    if !mvs.is_empty() {
        let mut dec = match BoolDecoder::init(&stream) {
            Ok(dec) => dec,
            Err(e) => panic!(
                "BoolDecoder rejected a stream the crate's own encoder emitted: {e:?} \
                 ({} bytes, {} vectors)",
                stream.len(),
                mvs.len()
            ),
        };
        for (i, expected) in mvs.iter().enumerate() {
            match read_mv(&mut dec, &contexts) {
                Ok(got) => assert_eq!(
                    got, *expected,
                    "MV roundtrip drift at vector {i} (contexts {contexts:?})"
                ),
                Err(e) => panic!(
                    "read_mv ran dry at vector {i} of {} on a self-emitted stream: {e:?}",
                    mvs.len()
                ),
            }
        }
    }

    // Leg 3 — arbitrary bytes through the decoder; components must
    // stay inside the §17 legal range.
    let tail = &vec_bytes[4 * n..];
    if let Ok(mut dec) = BoolDecoder::init(tail) {
        for _ in 0..MAX_DECODE_ONLY_VECTORS {
            match read_mv(&mut dec, &contexts) {
                Ok(mv) => {
                    assert!(
                        (-1023..=1023).contains(&mv.row) && (-1023..=1023).contains(&mv.col),
                        "read_mv escaped the §17 component range: {mv:?}"
                    );
                }
                Err(_) => break,
            }
        }
    }
});
