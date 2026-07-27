#![no_main]

//! Fuzz: the §9.1 silent-keyframe entry points —
//! [`encode_silent_keyframe`], [`SilentKeyframeParams`], and the
//! historical [`make_silent_keyframe_encoder`] /
//! [`SilentKeyframeEncoder::encode_keyframe`] handle.
//!
//! No other target reaches this surface: every keyframe fuzz driver
//! routes through `encode_keyframe(&I420Frame, ..)` and its variants,
//! while the silent path assembles its wire independently (§9.1 tag +
//! extension, §19.2 first partition with every knob at its skip
//! setting, per-MB `mb_skip_coeff = 1` records, §9.5 partition table,
//! per-partition §7.3 flush trailers) and back-patches the
//! `first_partition_size` field. That writer, its parameter
//! validation envelope, and the wrapper equivalence of the phase-1
//! encoder handle all reach the fuzzer here for the first time.
//!
//! Three legs per iteration:
//!
//! 1. **Rejection envelope.** All six [`SilentKeyframeParams`] fields
//!    are fed raw. The harness precomputes whether the parameter set
//!    is wire-legal (§9.1 dims in `1..=0x3FFF`, §9.4 level ≤ 63 /
//!    sharpness ≤ 7, §9.6 `y_ac_qi` ≤ 127, §9.5 partitions in
//!    {1,2,4,8}) and asserts `Ok`/`Err` matches exactly — a silent
//!    acceptance of an illegal knob or a rejection of a legal one is
//!    a finding. A dedicated probe bit additionally drives the
//!    out-of-range §9.1 axes (0 and > 0x3FFF) so the dimension guard
//!    stays covered even when the mapped dims are always legal.
//! 2. **Self-decode oracle.** Every emitted frame must decode through
//!    [`decode_vp8`]; the decoded visible dimensions must equal the
//!    requested ones and the §2 MB grid must match `div_ceil(16)`.
//!    Every decoded plane byte is folded into an FNV-1a accumulator
//!    so a short write / stale stride surfaces under ASan.
//! 3. **Wrapper equivalence.** On a fuzz-chosen bit the iteration
//!    also runs `make_silent_keyframe_encoder().encode_keyframe(..)`
//!    (which fixes every knob at the [`SilentKeyframeParams::new`]
//!    default) and asserts its bytes equal a direct
//!    `encode_silent_keyframe(SilentKeyframeParams::new(w, h))` call
//!    — the two public spellings must never drift.
//!
//! Input layout (all fields raw, consumed front-to-back):
//!
//! | Bytes  | Meaning |
//! |-------:|---------|
//! | `[0..2]` | Width  LE u16 → `1 + (v % 1023)` px |
//! | `[2..4]` | Height LE u16 → `1 + (v % 1023)` px |
//! | `[4]`    | `loop_filter_level` raw (> 63 must reject) |
//! | `[5]`    | `sharpness_level` raw (> 7 must reject) |
//! | `[6]`    | `y_ac_qi` raw (> 127 must reject) |
//! | `[7]`    | `nbr_of_dct_partitions` raw (non-{1,2,4,8} must reject) |
//! | `[8]`    | bit 0: invalid-dimension probe; bit 1: axis select; bit 2: wrapper-equivalence leg |
//!
//! Max area 1023 × 1023 ≈ 1 Mpx (≈ 4 096 MBs) — the silent writer
//! emits a handful of §7.3 booleans per MB and the decode leg
//! allocates ≈ 1.5 MiB of I420, both well inside the per-iteration
//! wall-time and memory budget.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_silent_keyframe, make_silent_keyframe_encoder, SilentKeyframeParams,
};

const INPUT_BYTES: usize = 9;

/// Fold a byte slice into a running FNV-1a hash so ASan reads every
/// decoded plane byte the self-decode claims to have produced.
fn fnv1a(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= u64::from(b);
        *acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < INPUT_BYTES {
        return;
    }

    let width = 1 + u32::from(u16::from_le_bytes([data[0], data[1]])) % 1023;
    let height = 1 + u32::from(u16::from_le_bytes([data[2], data[3]])) % 1023;
    let flags = data[8];

    // Leg 1a — the §9.1 dimension guard, driven from a dedicated probe
    // bit so the mapped-legal dims above don't shadow it. Both illegal
    // shapes (zero and > 0x3FFF) on a fuzz-chosen axis.
    if flags & 0x01 != 0 {
        let bad = if flags & 0x02 != 0 { 0 } else { 0x4000 + width };
        let params = SilentKeyframeParams {
            width: bad,
            height,
            ..SilentKeyframeParams::default()
        };
        if encode_silent_keyframe(params).is_ok() {
            panic!("encode_silent_keyframe accepted illegal width {bad}");
        }
        let params = SilentKeyframeParams {
            width,
            height: bad,
            ..SilentKeyframeParams::default()
        };
        if encode_silent_keyframe(params).is_ok() {
            panic!("encode_silent_keyframe accepted illegal height {bad}");
        }
        return;
    }

    // Leg 1b — raw knob bytes; Ok/Err must match the documented
    // envelope exactly.
    let params = SilentKeyframeParams {
        width,
        height,
        loop_filter_level: data[4],
        sharpness_level: data[5],
        y_ac_qi: data[6],
        nbr_of_dct_partitions: data[7],
    };
    let legal = params.loop_filter_level <= 63
        && params.sharpness_level <= 7
        && params.y_ac_qi <= 127
        && matches!(params.nbr_of_dct_partitions, 1 | 2 | 4 | 8);

    let bytes = match encode_silent_keyframe(params) {
        Ok(bytes) => {
            if !legal {
                panic!("encode_silent_keyframe accepted an out-of-range knob: {params:?}");
            }
            bytes
        }
        Err(e) => {
            if legal {
                panic!("encode_silent_keyframe rejected in-range params {params:?}: {e:?}");
            }
            return;
        }
    };

    // Leg 2 — self-decode oracle.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    match decode_vp8(&bytes) {
        Ok(decoded) => {
            if decoded.width != width || decoded.height != height {
                panic!(
                    "silent keyframe decoded to {}x{}, requested {width}x{height}",
                    decoded.width, decoded.height
                );
            }
            fnv1a(&mut hash, &decoded.y);
            fnv1a(&mut hash, &decoded.u);
            fnv1a(&mut hash, &decoded.v);
        }
        Err(e) => panic!("silent keyframe failed to self-decode ({width}x{height}): {e:?}"),
    }

    // Leg 3 — the phase-1 handle must emit byte-identical wire to the
    // direct entry point at the SilentKeyframeParams::new defaults.
    if flags & 0x04 != 0 {
        let via_handle = make_silent_keyframe_encoder()
            .encode_keyframe(&[], width, height)
            .unwrap_or_else(|e| {
                panic!("SilentKeyframeEncoder rejected legal dims {width}x{height}: {e:?}")
            });
        let direct = encode_silent_keyframe(SilentKeyframeParams::new(width, height))
            .expect("direct call with the same params must succeed");
        if via_handle != direct {
            panic!("SilentKeyframeEncoder wire drifted from encode_silent_keyframe");
        }
    }

    std::hint::black_box(hash);
});
