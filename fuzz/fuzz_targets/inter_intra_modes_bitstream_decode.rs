#![no_main]

//! Fuzz: the §16.1 interframe intra-mode parser —
//! [`parse_inter_frame_intra_macroblock_modes`] — under
//! attacker-controlled per-frame probability tables.
//!
//! The stateful decoder target reaches this parser only through real
//! interframe headers, whose resolved [`InterFrameIntraProbs`] start
//! from the §16.1 defaults and change only via the F-gated §9.10
//! overlay — so degenerate per-node probabilities (0, 255) and the
//! deep `B_PRED` sixteen-sub-block walk under them never get direct
//! attacker pressure. The key-frame twin has its own harness
//! (`panic_free_kf_mb_mode_decode`); the interframe variant — a
//! different Y-mode tree layout (root + two depth-1 subtrees against
//! four probabilities), the context-free [`IF_BMODE_PROB`] sub-block
//! walk, and the caller-forwarded `segment_id` / `mb_skip_coeff`
//! plumbing — was never fuzzed directly.
//!
//! Oracles, per parsed macroblock record:
//!
//! 1. **No panic / no UB** across the full probability × bitstream
//!    product space (the §7.3 boolean reads and the three tree walks
//!    must be total).
//! 2. **Structural invariant** — `subblock_modes` is `Some` **iff**
//!    `y_mode == IntraYMode::B` (§16.1: sub-block modes are read
//!    exactly when the 16×16 mode is `B_PRED`).
//! 3. **Forwarding contract** — the caller-supplied `segment_id` /
//!    `mb_skip_coeff` come back verbatim in the record (the parser
//!    must not invent or drop the §10 / §11.1 sideband).
//!
//! A parse error ends the iteration (the boolean decoder ran dry —
//! expected on arbitrary bytes).
//!
//! Input layout (consumed front-to-back):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0..4]` | `y_mode_prob` — four raw probability bytes |
//! | `[4..7]` | `uv_mode_prob` — three raw probability bytes |
//! | `[7]`    | bit 0: pass a `segment_id` (`Some(b >> 1 & 3)`); bit 3: `mb_skip_coeff` |
//! | `[8..]`  | raw §7.3 boolean stream (≤ 256 MB records per iteration) |
//!
//! Hard input cap 4 KiB; the parser allocates nothing, so wall time
//! is dominated by the boolean reads.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    parse_inter_frame_intra_macroblock_modes, BoolDecoder, InterFrameIntraProbs, IntraYMode,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 8;
const MAX_RECORDS: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let probs = InterFrameIntraProbs {
        y_mode_prob: [data[0], data[1], data[2], data[3]],
        uv_mode_prob: [data[4], data[5], data[6]],
    };
    let flags = data[7];
    let segment_id = if flags & 0x01 != 0 {
        Some((flags >> 1) & 3)
    } else {
        None
    };
    let mb_skip_coeff = flags & 0x08 != 0;

    let Ok(mut dec) = BoolDecoder::init(&data[HEADER_BYTES..]) else {
        return;
    };

    for _ in 0..MAX_RECORDS {
        match parse_inter_frame_intra_macroblock_modes(&mut dec, &probs, segment_id, mb_skip_coeff)
        {
            Ok(modes) => {
                // §16.1 — sub-block modes travel iff the Y mode is B_PRED.
                assert_eq!(
                    modes.subblock_modes.is_some(),
                    modes.y_mode == IntraYMode::B,
                    "subblock_modes presence contradicts y_mode ({:?})",
                    modes.y_mode
                );
                // §10 / §11.1 sideband must be forwarded verbatim.
                assert_eq!(modes.segment_id, segment_id, "segment_id not forwarded");
                assert_eq!(
                    modes.mb_skip_coeff, mb_skip_coeff,
                    "mb_skip_coeff not forwarded"
                );
            }
            Err(_) => break,
        }
    }
});
