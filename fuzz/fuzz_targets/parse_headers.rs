#![no_main]

//! Fuzz: panic-freedom of the public parse APIs.
//!
//! Five public parse entry points, all `fn(&[u8]) -> Result<_, _>`,
//! get the same arbitrary input:
//!
//! | Surface                                       | RFC 6386 § |
//! |-----------------------------------------------|------------|
//! | `frame_tag::parse_header`                     | §9.1 tag   |
//! | `frame_tag::parse_keyframe_header`            | §9.1 KF    |
//! | `frame_header::Vp8FrameHeader::parse`         | §9.1 wrap  |
//! | `coded_header::Vp8CodedHeader::parse(.., true)`  | §19.2 KF |
//! | `coded_header::Vp8CodedHeader::parse(.., false)` | §19.2 IF |
//!
//! Plus the two IVF framing entry points (`ivf::parse_header`,
//! `ivf::parse_frame_header`).
//!
//! The §19.2 coded-header walk in particular routes through
//! `update_segmentation`, `mb_lf_adjustments`, `quant_indices`,
//! `token_prob_update`, and `mv_prob_update` — the malformed
//! segmentation-map / LF-delta / token-prob-update paths the depth-mode
//! brief calls out. Driving it on both `key_frame=true` and `false`
//! exercises the gated inter-only blocks (ref-frame refresh / copy,
//! sign-bias, intra-Y/UV-mode probability replacements).
//!
//! No decode, no allocation beyond the parsers' own internal state, so
//! this target runs at the fastest exec/s of the three.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    coded_header::Vp8CodedHeader,
    frame_header::Vp8FrameHeader,
    frame_tag::{parse_header as parse_frame_tag, parse_keyframe_header},
    ivf::{parse_frame_header as parse_ivf_frame_header, parse_header as parse_ivf_header},
};

fuzz_target!(|data: &[u8]| {
    // §9.1 frame-tag pair.
    let _ = parse_frame_tag(data);
    let _ = parse_keyframe_header(data);

    // §9.1 framed header.
    let _ = Vp8FrameHeader::parse(data);

    // §19.2 coded-header partition — both key-frame and inter-frame
    // variants. The first-partition byte range is unknown to us
    // ahead of decode; libFuzzer's bytes get to play the role of an
    // arbitrary partition payload directly.
    let _ = Vp8CodedHeader::parse(data, true);
    let _ = Vp8CodedHeader::parse(data, false);

    // IVF container framing.
    let _ = parse_ivf_header(data);
    let _ = parse_ivf_frame_header(data);
});
