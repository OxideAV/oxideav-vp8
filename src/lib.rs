//! Pure-Rust VP8 video codec + IVF container — encoder + decoder.
//!
//! Status (RFC 6386 surface coverage):
//! * Boolean (arithmetic) decoder + encoder — §7. Round-trip tested.
//! * Frame tag + uncompressed key-frame chunk — §9.1. Both directions.
//! * Frame header (segmentation / loop filter / quant / probs) — §9.2-§9.10.
//!   Both directions, including ref-frame sign bias, refresh + copy-buffer
//!   flags, mb_skip_enabled + per-frame entropy-prob updates.
//! * Macroblock prediction modes — intra-16×16 (DC / V / H / TM / B_PRED),
//!   intra-4×4 (all 10 sub-modes: B_DC, B_TM, B_VE, B_HE, B_LD, B_RD, B_VR,
//!   B_VL, B_HD, B_HU), intra-8×8 chroma (same 4 modes as 16×16). Encoder
//!   picks via SSE-on-source for every candidate; decoder reconstructs.
//! * Inter modes — NEAREST_MV / NEAR_MV / ZERO_MV / NEW_MV plus all four
//!   SPLIT_MV partitionings (16×8 / 8×16 / 8×8 / 4×4). Per-partition
//!   integer-pel + quarter-pel motion search on the encoder side.
//! * Token / coefficient decoding — §13. Flat unrolled `GetCoeffs`.
//!   Encoder mirrors with its emit-side companion.
//! * Inverse 4×4 DCT + 4×4 WHT (§14) + forward counterparts. Bit-exact
//!   round-trip; column-then-row IDCT pass order matches the RFC C ref.
//! * Loop filter — §15. Decoder handles simple + normal modes, both edge
//!   directions, MB + sub-block. Encoder emits normal mode only (with
//!   level derived from qindex via libvpx's `clamp(15 + qindex/8, 1, 63)`).
//! * Reference management — LAST / GOLDEN / ALTREF, with refresh +
//!   copy-buffer per RFC 6386 §9.7. Sign-bias handled in find_near_mvs.
//! * Segmentation map (§10) — per-MB segment id + per-segment quantiser
//!   delta, both directions; encoder classifies by source-luma variance
//!   and emits the entropy-matched tree_probs triple.
//! * IVF container — demuxer + muxer (FourCC `VP80` probe).
//!
//! Encoder-only smarts (not in spec, layered on top):
//! * Lagrangian rate-distortion mode decision (`D + λ·R`), per-frame
//!   scene-cut detection, look-ahead alt-ref synthesis, multi-reference
//!   per-MB picker, context-adaptive `prob_intra/last/gf` triple.

#![cfg_attr(feature = "simd", feature(portable_simd))]
// When built without the `registry` feature, large swathes of the
// encoder (P-frame helpers, alt-ref synthesis, scene-cut state, etc.)
// have no callers — they're only reachable via the `Encoder` trait
// implementation that's gated behind `registry`. Suppress the
// resulting dead-code warnings rather than gating every helper.
#![cfg_attr(not(feature = "registry"), allow(dead_code))]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::too_many_arguments)]
// VP8's bitstream/transform code reads more naturally with explicit shifts
// and bit ops; clippy's identity_op / manual_div_ceil / etc. lints flag a
// number of these as "could be simplified" but the explicit form is what
// the spec mirrors.
#![allow(clippy::identity_op)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_slice_fill)]
#![allow(clippy::let_and_return)]
#![allow(clippy::useless_asref)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::ptr_arg)]

pub mod bool_decoder;
pub mod bool_encoder;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod fdct;
pub mod frame;
pub mod frame_header;
pub mod frame_tag;
pub mod inter;
pub mod intra;
#[cfg(feature = "registry")]
pub mod ivf;
pub mod loopfilter;
pub mod mv;
#[cfg(feature = "registry")]
pub mod registry;
pub mod tables;
pub mod tokens;
pub mod transform;

pub const CODEC_ID_STR: &str = "vp8";

// Public unconditional API — works whether or not `registry` is enabled.
pub use decoder::decode_vp8;
pub use error::{Result, Vp8Error};
pub use frame::Vp8Frame;
pub use frame_header::{parse_keyframe_header, FrameHeader};
pub use frame_tag::{parse_header, FrameTag, FrameType, KeyframeHeader, ParsedHeader};

// Public registry-gated API — keeps the framework integration surface
// (Decoder/Encoder traits, IVF Demuxer/Muxer, register helpers) behind
// the default-on `registry` feature so image-library callers can build
// the crate without dragging in `oxideav-core`.
#[cfg(feature = "registry")]
pub use decoder::{decode_frame, Vp8Decoder};
#[cfg(feature = "registry")]
pub use registry::{__oxideav_entry, register, register_codecs, register_containers};
