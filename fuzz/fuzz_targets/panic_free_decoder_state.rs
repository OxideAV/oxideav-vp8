#![no_main]

//! Fuzz: multi-packet `Vp8DecoderState::decode_frame` driver.
//!
//! Drives the stateful decoder with a sequence of length-prefixed
//! packets (see `oxideav_vp8_fuzz::split_packets`). This exercises
//! the §9.7 reference-frame refresh ladder — the LAST / GOLDEN /
//! ALTREF carry-over a one-shot `decode_vp8` call by definition
//! can't reach. Random bytes routinely yield an inter-frame that
//! references a "previous" frame that itself only ever existed as
//! garbage; the contract is that we surface an `Err` and keep going,
//! never panic.
//!
//! Per-iteration packet cap (32) bounds wall time even when libFuzzer
//! feeds a long input. Per-packet dimensions are NOT pre-screened
//! here: the first valid keyframe's width/height is what locks the
//! state machine's geometry, and the state code is itself responsible
//! for rejecting subsequent packets that don't match — so refusing a
//! crazy width up front would hide the very edge cases this target
//! is meant to find.
//!
//! Hard caps: input ≤ 4 KiB (libFuzzer's default; defence-in-depth
//! at harness entry); at most 32 packets per iteration.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::Vp8DecoderState;
use oxideav_vp8_fuzz::split_packets;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const MAX_PACKETS_PER_ITER: usize = 32;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let packets = split_packets(data, MAX_PACKETS_PER_ITER);
    if packets.is_empty() {
        return;
    }
    let mut state = Vp8DecoderState::new();
    for pkt in packets {
        let _ = state.decode_frame(pkt);
    }
});
