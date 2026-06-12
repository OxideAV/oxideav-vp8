//! Regression pin for the round-284 `decode_stream_token_descent` fuzz
//! finding: the stateless [`oxideav_vp8::decode_vp8`] entry point
//! initialised consumed DCT partitions with the strict control-partition
//! `BoolDecoder::init` (rejecting `len < 2` with `InputTooShort`), while
//! the stateful keyframe path behind
//! [`oxideav_vp8::Vp8DecoderState::decode_frame`] used the tolerant
//! §20.2 `init_bool_decoder` form (`sz < 2` → zero-initialised value
//! register, empty input — RFC 6386 §20.2), which is what the spec
//! reference does. A spec-legal key frame whose consumed DCT partition
//! rounds down to a 0- or 1-byte slice therefore decoded through the
//! stateful path but was rejected by the one-shot path.
//!
//! The packet below is the fuzz-found witness: a 16×16 key frame whose
//! DCT-partition carve leaves a consumed partition shorter than 2 bytes.
//! Both entry points must accept it and produce byte-identical planes.

use oxideav_vp8::{decode_vp8, Vp8DecoderState};

/// Fuzz-found 16×16 key frame (57 bytes) whose consumed DCT partition is
/// shorter than the 2-byte bool-decoder priming read.
const SHORT_DCT_PARTITION_KEYFRAME: [u8; 57] = [
    0xf0, 0x02, 0x00, 0x9d, 0x01, 0x2a, 0x10, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0x85, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[test]
fn decode_vp8_accepts_short_consumed_dct_partition() {
    let frame = decode_vp8(&SHORT_DCT_PARTITION_KEYFRAME)
        .expect("§20.2: a sub-2-byte consumed DCT partition is spec-legal and must decode");
    assert_eq!((frame.width, frame.height), (16, 16));
    assert_eq!(frame.y.len(), 16 * 16);
}

#[test]
fn one_shot_and_stateful_keyframe_paths_agree_on_short_dct_partition() {
    let one_shot = decode_vp8(&SHORT_DCT_PARTITION_KEYFRAME).expect("one-shot decode");
    let stateful = Vp8DecoderState::new()
        .decode_frame(&SHORT_DCT_PARTITION_KEYFRAME)
        .expect("stateful decode");
    assert_eq!(
        (one_shot.width, one_shot.height),
        (stateful.width, stateful.height)
    );
    assert_eq!(one_shot.y, stateful.y, "Y-plane drift between entry points");
    assert_eq!(one_shot.u, stateful.u, "U-plane drift between entry points");
    assert_eq!(one_shot.v, stateful.v, "V-plane drift between entry points");
}
