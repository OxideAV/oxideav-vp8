//! # oxideav-vp8
//!
//! **Status:** clean-room rebuild in progress (post 2026-05-20 audit).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy and the crate is being re-implemented from scratch against
//! RFC 6386, using only material under `docs/` and black-box
//! validator binaries.
//!
//! Currently landed:
//!
//! * VP8 boolean (range) entropy decoder — the foundational primitive
//!   every higher-level decode step is built on (RFC 6386 §7); see
//!   [`bool_decoder`].
//! * VP8 uncompressed frame header (frame tag + key-frame start code +
//!   width / height / scale codes) per RFC 6386 §9.1; see
//!   [`frame_header`].
//! * VP8 boolean-coded frame header — the complete §19.2 table per
//!   RFC 6386; see [`coded_header`]. Covers segmentation, loop-filter
//!   knobs, MB loop-filter adjustments, DCT partition count, quantiser
//!   indices, the `token_prob_update()` sweep, entropy-probability
//!   refresh, the inter-frame reference refresh / copy / sign-bias
//!   ladder, the per-MB skip flag, the §9.10 tail of `prob_intra` /
//!   `prob_last` / `prob_gf`, the gated Y and UV intra-mode
//!   probability replacements, and the `mv_prob_update()` sub-block
//!   of §17.2 (two 19-position MV_CONTEXTs, each `F? P(7)`).
//! * VP8 key-frame macroblock mode decoding (RFC 6386 §11) — the
//!   per-MB segment id, mb_skip_coeff, Y mode (`kf_ymode_tree`),
//!   sixteen sub-block modes when `B_PRED` is selected
//!   (`bmode_tree` driven by the §11.3 context predictors and the
//!   §11.5 `kf_bmode_prob[10][10][9]` table), and the UV mode
//!   (`uv_mode_tree`). See [`macroblock`].
//! * VP8 intra-prediction pixel kernels (RFC 6386 §12) — the four
//!   16×16 luma modes, the four 8×8 chroma modes, and the ten 4×4
//!   sub-block modes. Pure pixel-shape kernels operating on small
//!   neighbour arrays; no entropy decode, no IDCT, no loop filter.
//!   See [`intra_predict`].
//! * VP8 DCT-coefficient token decoding (RFC 6386 §13) — the
//!   `coeff_tree` walker, the `DCTextra` extra-bits decode, the §13.5
//!   default token-probability table, and a per-sub-block
//!   `decode_block` primitive that recovers a `[i16; 16]` of
//!   quantised coefficients. The §13.3 per-macroblock token walk
//!   ([`decode_mb_coeffs`]) sits on top of `decode_block`: it walks the
//!   24/25 residual blocks of one macroblock (Y2 first when present, then
//!   16 Y, 4 U, 4 V), threads the above / left non-zero predictor context
//!   through the §20.16 `left_context_index` / `above_context_index` slot
//!   tables (a nine-entry [`MbEntropyCtx`]: 4 Y, 2 U, 2 V, 1 Y2), honours
//!   the §13.1 `mb_skip_coeff` short-circuit with the §20.16
//!   `reset_mb_context` Y2-preserving rule, selects the `YAfterY2` /
//!   `YNoY2` plane per the macroblock's `has_y2`, and reorders each block
//!   into raster order via the §20.16 `zigzag[16]` table — producing an
//!   [`frame::MbCoeffs`] directly from the bitstream. The coefficients are
//!   the raw quantized token values; the §14.1 Y2 / chroma dequant scaling
//!   is applied by the [`dequant`] layer. See [`dct_tokens`].
//! * VP8 §14.1 dequantization scaling and the bitstream→dequant wrapper —
//!   [`dequant::MbDequantFactors`] computes the six §14.1 dequant factors
//!   (Y1 DC/AC, Y2 DC/AC, chroma DC/AC) from the §9.6 quantiser indices,
//!   applying the §20.4 `dixie.c` `dequant_init` scaling / clamping rules
//!   (Y2 DC × 2, Y2 AC × 155/100 floored at 8, chroma DC capped at 132)
//!   over the §14.1 `dc_qlookup` / `ac_qlookup` tables with the §20.4
//!   `clamp_q` index saturation. [`dequant::MbDequantFactors::for_segment`]
//!   layers the §10 per-segment quantizer override (delta or absolute) on
//!   the frame baseline. [`dequant::MbDequantFactors::dequantize`] scales a
//!   raw [`frame::MbCoeffs`] in place, and
//!   [`dequant::decode_and_dequantize_mb`] is the wrapper that runs
//!   [`dct_tokens::decode_mb_coeffs`] then dequantizes — turning the
//!   bitstream straight into the pre-dequantized [`frame::MbCoeffs`] that
//!   [`frame::decode_keyframe`] consumes, completing the keyframe decode
//!   chain bitstream → dequant → reconstruct → pixels. See [`dequant`].
//! * VP8 dequantization and inverse transforms (RFC 6386 §14) — the
//!   §14.1 `dc_qlookup` / `ac_qlookup` tables with Y-plane factor
//!   computation, the §14.3 inverse WHT (general + single-DC fast
//!   path), the §14.4 inverse DCT, and the §14.5 predictor+residue
//!   summation with `clamp255`. Per-block raster-order primitives;
//!   the Y2/chroma dequant scaling and the zig-zag→raster reordering
//!   are documented spec gaps left to the integration round. See
//!   [`inverse_transform`].
//! * VP8 loop filter per-segment primitives (RFC 6386 §15) — the §15.2
//!   simple filter, the §15.3 normal subblock / macroblock filters, and
//!   the §15.4 control-parameter derivation
//!   ([`LoopFilterParams::derive`]). Pure per-edge-segment kernels
//!   operating on a caller-supplied pixel window. See [`loop_filter`].
//! * VP8 §15.1 loop-filter frame geometry — [`filter_frame`], the
//!   per-frame post-pass that walks the reconstructed [`KeyframePlanes`]
//!   in raster order and applies the §15.2 / §15.3 kernels across each
//!   macroblock's left inter-MB edge, three internal vertical subblock
//!   edges, top inter-MB edge, and three internal horizontal subblock
//!   edges (chroma analogues for the normal filter; the simple filter is
//!   luma-only), honouring the §15.1 page-86 steps-2/4 skip rule (mode
//!   neither `B_PRED` nor `SPLITMV` *and* no coded coefficient) and the
//!   §15 page-84 level-0 whole-MB skip. The per-MB level derivation
//!   ([`calculate_mb_filter_level`]) implements the §20.6 `dixie.c`
//!   `calculate_filter_parameters` body: base `loop_filter_level` + §10
//!   per-segment override (delta or absolute, clamped `0..=63`) + §9.4
//!   reference / mode deltas (clamped again). [`FrameFilterConfig`]
//!   carries the resolved frame state, with
//!   [`FrameFilterConfig::keyframe`] building it from a
//!   [`Vp8CodedHeader`]. See [`loop_filter`].
//! * VP8 interframe intra-predicted macroblock mode decoding (RFC 6386
//!   §16.1) — the dynamic `ymode_prob` / `uv_mode_prob` resolution
//!   ([`InterFrameIntraProbs::for_frame_header`]) including the §16.1
//!   last-paragraph reset on key frames and the §9.10 per-frame
//!   overlay, plus the `ymode_tree` / context-free `bmode_prob` /
//!   `uv_mode_tree` walks
//!   ([`parse_inter_frame_intra_macroblock_modes`]). Returns a
//!   [`MacroblockModes`] for one intra-predicted MB on an interframe;
//!   the caller forwards the `segment_id` and `mb_skip_coeff` already
//!   read before the intra-vs-inter discriminator. See
//!   [`macroblock`].
//! * VP8 §14.2 per-macroblock reconstruction orchestration —
//!   [`decode_keyframe_mb_non_bpred`] ties together token-decoded
//!   coefficients, the §14.3 inverse WHT, the §14.4 inverse DCT, the
//!   §12 intra-prediction kernels and the §14.5 predictor+residue
//!   summation to produce reconstructed Y / U / V pixels for one
//!   macroblock whose 16×16 luma mode is one of the four non-`B_PRED`
//!   modes. Honours the §11.1 `mb_skip_coeff` short-circuit.
//!   [`decode_keyframe_mb_bpred`] is the companion §11.3 / §12.3
//!   `B_PRED` orchestrator: it drives the sixteen 4×4 luma sub-blocks
//!   with per-sub-block neighbour evolution (each sub-block's
//!   reconstructed pixels become the next sub-block's `above` / `left`),
//!   applies the §12.3 right-edge above-right fixup for sub-blocks
//!   7 / 11 / 15, omits the Y2 WHT seeding (a `B_PRED` MB has no Y2
//!   block), and reconstructs the chroma planes with the same 8×8
//!   §12.2 path. See [`reconstruct`].
//! * VP8 per-frame keyframe raster walker (RFC 6386 §12 / §14.2) —
//!   [`decode_keyframe`] iterates a key frame's macroblocks in
//!   raster-scan order, assembling each MB's [`reconstruct::MbNeighbors`]
//!   strips from the already-reconstructed full-frame plane buffers (the
//!   bottom row of the MB above, the rightmost column of the MB to the
//!   left, the corner pixel, and — for `B_PRED` — the four §12.3
//!   above-and-to-the-right pixels with the rightmost-MB `(-1,15)`
//!   clamp), selecting the non-`B_PRED` vs `B_PRED` orchestrator per the
//!   MB's decoded luma mode, and writing reconstructed pixels into the
//!   [`frame::KeyframePlanes`] I420 buffers. Off-frame edges are reported
//!   as `None` so the §12.2 `DC_PRED` averaging distinguishes genuine
//!   pixels from 127 / 129 fills. Consumes [`frame::MbCoeffs`]
//!   (pre-dequantized) produced by the [`dequant`] layer. See [`frame`].
//!
//! * VP8 motion-vector component decoding (RFC 6386 §17) —
//!   [`read_mv_component`] (§17.1 `read_mvcomponent`: the short-vs-long
//!   range selector, the [`SMALL_MVTREE`] tree-coded `0..=7` short form,
//!   the independent-bit long form with the implicit bit-3 rule, and the
//!   conditional sign) and [`read_mv`] (§17.2 `read_mv`: row then column).
//!   [`resolve_mv_contexts`] applies the round-4 §17.2 `mv_prob_update()`
//!   overlays onto a base [`MvContexts`] (defaulting via
//!   [`default_mv_contexts`] on key frames, carried-forward across
//!   interframes), turning the parsed updates into the live decoding
//!   tables. Returns the raw differential [`Mv`]; the §16.3 reference-base
//!   addition, the §18.1 range clamp / stored-luma doubling, and the §16
//!   `mv_ref` / SPLITMV machinery are deferred to later inter-prediction
//!   rounds. See [`motion_vector`].
//!
//! * VP8 interframe motion compensation (RFC 6386 §16.2 / §18) — the
//!   inter-prediction slice that *consumes* a decoded motion vector.
//!   [`select_ref_frame`] reads the §16.2 `prob_last` / `prob_gf`
//!   reference selector into a [`RefFrame`]; [`stored_luma_mv`] applies
//!   the §18.1 stored-luma doubling, [`chroma_mv`] the §18.1 `avg()`
//!   chroma averaging (cross-checked against the §20.14 closed form), and
//!   [`apply_full_pixel`] the version-3 full-pel-chroma truncation.
//!   [`fetch_block_whole_pixel`] is the §20.14 `build_mc_border`
//!   edge-replicated 4×4 reference fetch specialised to whole-pixel
//!   offsets; [`fetch_block_halo`] is the 9×9 support-halo fetch the
//!   six-tap convolution needs. The §18.3 sub-pixel interpolation lands
//!   here: [`SIXTAP_FILTERS`] / [`BILINEAR_FILTERS`] are the §18.3 tap
//!   tables ([`filter_set_for_version`] / [`FilterSet`] pick the set by
//!   frame-tag version), [`interp`] / [`sixtap_2d`] are the §20.14
//!   one-dimensional convolutions, and [`filter_block_4x4`] is the
//!   §20.14 `filter_block` per-sub-block dispatcher (whole-pixel copy or
//!   horizontal-then-vertical six-tap synthesis). [`predict_inter_mb`]
//!   assembles the §18.2 whole-MB prediction buffer for a non-SPLITMV
//!   macroblock ([`ReferencePlanes`] borrows the reference frame's I420
//!   planes), and [`reconstruct_inter_mb`] folds in the §14 dequantized
//!   residual (Y2 WHT seeding + per-sub-block inverse DCT + §14.5
//!   summation) to complete inter-MB reconstruction. The whole-pixel-only
//!   [`predict_inter_mb_whole_pixel`] /
//!   [`reconstruct_inter_mb_whole_pixel`] entry points are retained and
//!   refuse a sub-pixel vector with
//!   [`MotionCompError::SubPixelNotSupported`]. See [`motion_comp`].
//!
//! * VP8 near/nearest motion-vector census and inter-mode tree (RFC 6386
//!   §16.2 / §16.3 / §18.1) — [`near_mv`], the slice that decides *which*
//!   vector a whole-MB inter macroblock uses. [`find_near_mvs`] is the
//!   §16.3 `vp8_find_near_mvs` spatial census over the above / left /
//!   above-left [`MbInfo`] neighbours ([`MbInfo::border`] for the §16.3
//!   1-MB off-frame border), producing the `best` / `nearest` / `near`
//!   candidates and the weighted `cnt` census (with the §16.3 dedupe,
//!   SPLITMV-merge, near↔nearest swap, and best:=nearest store).
//!   [`SignBias`] carries the §9.7 sign-bias bits and drives the §16.3
//!   `mv_bias` negation; [`mv_ref_probs`] derives the §16.2 tree
//!   probabilities from the census via [`MV_COUNTS_TO_PROBS`]; and
//!   [`read_inter_mode`] walks [`MV_REF_TREE`] into an [`InterMode`].
//!   [`clamp_mv`] / [`MvClampRect`] are the §16.3 / §18.1 one-MB-border
//!   clamp. [`resolve_inter_mb_mv`] ties census → probabilities → mode →
//!   the single per-MB vector (ZEROMV / NEARESTMV / NEARMV / NEWMV, the
//!   last adding the §17 differential to the clamped best per §18.1), and
//!   [`decode_inter_mb`] drives [`reconstruct_inter_mb`] with that vector
//!   so a whole-MB inter macroblock decodes from bitstream to
//!   reconstructed pixels. §16.4 SPLITMV's per-sub-block walk is a
//!   follow-up ([`ResolvedInterMode::Split`] /
//!   [`InterMbError::SplitNotSupported`] carry the clamped best base).
//!   See [`near_mv`].
//!
//! * VP8 top-level per-frame decode driver ([`decode_vp8`]) and the
//!   [`oxideav_core::Decoder`] integration ([`decoder::Vp8Decoder`],
//!   gated on the default-on `registry` feature). [`decode_vp8`] takes
//!   the raw bytes of one VP8 frame, parses the §9.1 uncompressed header
//!   and the §19.2 boolean-coded frame header, runs the §11 macroblock
//!   prediction layer, carves the §9.5 DCT partitions (round-robin row
//!   striping per §20.4), decodes + dequantizes each macroblock's §13
//!   residuals, reconstructs the frame via [`decode_keyframe`], applies
//!   the §15.1 loop-filter post-pass, and emits a visible-cropped I420
//!   [`decoder::Vp8DecodedFrame`]. Interframes (§16) are surfaced as a
//!   clean [`decoder::DecodeError::Unsupported`]. See [`decoder`].
//!
//! * VP8 multi-frame stateful decoder driver
//!   ([`state::Vp8DecoderState`], [`state::Vp8DecoderState::decode_frame`])
//!   owning the RFC 6386 §9 three-slot reference-frame buffer
//!   ([`state::RefFrameSlot`] for `LAST` / `GOLDEN` / `ALTREF`), the §9.10
//!   entropy / intra-mode / MV carry-state (with the §20 `saved_entropy`
//!   rollback when `refresh_entropy_probs` is false), and the §9
//!   `copy_buffer_to_*` / `refresh_*` slot rotation. The per-frame walker
//!   dispatches each macroblock to either the §11 keyframe intra path or
//!   the §16 inter path (intra-on-interframe via
//!   [`parse_inter_frame_intra_macroblock_modes`], whole-MB inter via
//!   `select_ref_frame` → §16.3 census → §16.2 inter-mode tree →
//!   `reconstruct_inter_mb`, SPLITMV via `decode_split_mv` →
//!   `reconstruct_split_mv_mb`), runs the §15.1 loop-filter post-pass
//!   with the full §9.4 ref + mode delta ladder
//!   ([`loop_filter::FrameFilterConfig::interframe`] +
//!   [`loop_filter::filter_inter_frame`] +
//!   [`loop_filter::calculate_mb_filter_level_inter`]), and emits a
//!   visible-cropped I420 picture per frame.
//!
//! With the §13.3 per-MB token walk ([`dct_tokens::decode_mb_coeffs`]),
//! the §14.1 dequant scaling ([`dequant::decode_and_dequantize_mb`]), the
//! per-key-frame [`decode_vp8`] driver, the §16.4 SPLITMV walk, and now
//! the multi-frame [`state::Vp8DecoderState`] all landed, the full
//! key+inter decode chain is bit-exact against the libvpx/ffmpeg reference
//! on multi-frame fixtures (single I + P; 5-frame mid-GOP golden refresh;
//! 10-frame `auto-alt-ref` + ARNR).
//!
//! * **VP8 encoder Phase 1** — the bitstream-formatting half of the
//!   encoder ([`encoder`]). Ships the §7.3 [`encoder::BoolEncoder`] (a
//!   Rust port of the `bool_encoder` C listing embedded in RFC 6386
//!   §7.3), the §9.1 / §9.3 / §9.4 / §9.5 / §9.6 / §9.9 / §9.10 / §9.11
//!   frame-header writer subroutines, and the top-level
//!   [`encoder::encode_silent_keyframe`] driver. Every emitted MB is
//!   coded as `mb_skip_coeff = 1` with `DC_PRED` luma + chroma, so the
//!   §13 token / §14 transform / mode-selection logic is not exercised
//!   yet — but the resulting bytes are a structurally-valid VP8 key
//!   frame the crate's own [`decode_vp8`] consumes and that
//!   `ffmpeg -c:v vp8` accepts when wrapped in an IVF container. The
//!   §13 / §14 rate-distortion encode side is future work.
//!
//! * **VP8 encoder Phase 2 begin: §14 forward 4×4 DCT + WHT
//!   primitives** ([`forward_transform`]) — [`forward_dct_4x4`],
//!   [`forward_wht_4x4`], and [`raster_to_scan`], the encoder partners
//!   of the §14.3 / §14.4 inverse transforms and the §20.16 zig-zag
//!   reorder. Mechanically derived as the transpose of the §14.3 /
//!   §14.4 inverse listings (the §14.4 preamble itself notes the
//!   transform is *"a classical 2-D inverse discrete cosine
//!   transform"*); both reuse the same `COSPI8_SQRT2_MINUS1 = 20091`
//!   / `SINPI8_SQRT2 = 35468` fixed-point constants so the
//!   forward / inverse rounding shapes match. Wired through the
//!   existing [`encoder::TokenEncoder`] for a per-MB roundtrip proof
//!   (FDCT → quantize → raster-to-scan → encode → decode → scan-to-
//!   raster → dequant → IDCT) at 48.13 dB PSNR on a synthetic flat
//!   block at `yac_qi = 32` (lossless at `yac_qi = 0`); see
//!   `tests/encoder_transform_roundtrip.rs`. The per-MB block-set
//!   wiring (Y2 DC seeding, 24/25-block walk, RD-driven mode +
//!   quant-step selection) is the next encoder round.

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(missing_debug_implementations)]

pub mod bool_decoder;
pub mod coded_header;
pub mod dct_tokens;
pub mod decoder;
pub mod dequant;
pub mod encoder;
pub mod error;
pub mod forward_transform;
pub mod frame;
pub mod frame_header;
pub mod frame_tag;
pub mod intra_predict;
pub mod inverse_transform;
pub mod ivf;
pub mod loop_filter;
pub mod macroblock;
pub mod motion_comp;
pub mod motion_search;
pub mod motion_vector;
pub mod near_mv;
pub mod reconstruct;
pub mod state;
pub mod stream;

#[cfg(feature = "registry")]
pub mod registry;

// ───────────────────── 0.1.13 module-path aliases ─────────────────────
//
// The crates.io `oxideav-vp8 0.1.13` release organised its public
// modules under a slightly different naming scheme. The aliases below
// restore the historical module paths without renaming the underlying
// files, so downstream consumers that wrote
// `oxideav_vp8::loopfilter::LoopFilterParams` keep building.

/// 0.1.13 alias of [`crate::forward_transform`] — the §14 forward 4×4
/// DCT / WHT primitives.
pub mod fdct {
    pub use crate::forward_transform::*;
}

/// 0.1.13 alias of [`crate::inverse_transform`] — the §14 inverse 4×4
/// DCT / WHT primitives and §14.5 predictor+residue summation.
pub mod transform {
    pub use crate::inverse_transform::*;
}

/// 0.1.13 alias of [`crate::intra_predict`] — the §12 intra-prediction
/// pixel kernels.
pub mod intra {
    pub use crate::intra_predict::*;
}

/// 0.1.13 alias of [`crate::loop_filter`] — the §15 loop-filter
/// primitives.
pub mod loopfilter {
    pub use crate::loop_filter::*;
}

/// 0.1.13 alias of [`crate::motion_vector`] — the §17 motion-vector
/// primitives.
pub mod mv {
    pub use crate::motion_vector::*;
}

/// 0.1.13 alias of [`crate::dct_tokens`] — the §13 DCT-token decoding /
/// classification.
pub mod tokens {
    pub use crate::dct_tokens::*;
}

/// 0.1.13 grouped alias of the inter-prediction modules
/// ([`crate::motion_comp`] + [`crate::motion_search`] +
/// [`crate::motion_vector`] + [`crate::near_mv`]).
pub mod inter {
    pub use crate::motion_comp::*;
    pub use crate::motion_search::*;
    pub use crate::motion_vector::*;
    pub use crate::near_mv::*;
}

/// 0.1.13 alias surface for the §7.3 bool encoder. The current
/// [`BoolEncoder`](crate::encoder::BoolEncoder) lives in
/// [`crate::encoder`]; this module re-exposes it (and the
/// classification helper) under the original 0.1.13 path.
pub mod bool_encoder {
    pub use crate::encoder::BoolEncoder;
    pub use crate::encoder::{
        classify_coeff_token, encode_coeff_block, TokenEncodeError, TokenEncoder,
    };
}

/// 0.1.13 grouping module for the in-tree default-probability / scan
/// / dequant tables. Today these tables live next to the layer that
/// consumes them; this module re-exports them under the original
/// `tables` path.
pub mod tables {
    pub use crate::coded_header::{DEFAULT_MV_CONTEXT, MV_PROB_COUNT};
    pub use crate::dct_tokens::{COEFF_BANDS, DEFAULT_COEFF_PROBS, MB_ENTROPY_CTX_LEN, ZIGZAG};
    pub use crate::inverse_transform::{AC_QLOOKUP, DC_QLOOKUP, QINDEX_RANGE};
    pub use crate::loop_filter::{MAX_MB_SEGMENTS, MAX_MODE_LF_DELTAS, MAX_REF_LF_DELTAS};
    pub use crate::macroblock::{IF_BMODE_PROB, IF_UV_MODE_PROB_DEFAULTS, IF_YMODE_PROB_DEFAULTS};
    pub use crate::motion_comp::{BILINEAR_FILTERS, SIXTAP_FILTERS};
    pub use crate::near_mv::{
        MV_COUNTS_TO_PROBS, MV_PARTITIONS, MV_PARTITION_PROBS, MV_PARTITION_TREE, MV_REF_TREE,
        SUBMV_REF_PROBS, SUBMV_REF_TREE,
    };
}

pub use bool_decoder::{BoolDecoder, BoolDecoderError};
pub use coded_header::{
    CodedHeaderError, MbLfAdjustments, MvProbUpdates, QuantIndices, TokenProbUpdates,
    UpdateSegmentation, Vp8CodedHeader, DEFAULT_MV_CONTEXT, MV_PROB_COUNT,
};
pub use dct_tokens::{
    decode_block, decode_mb_coeffs, merge_default_token_probs, BlockType, CoeffProbs, DctToken,
    DctTokenError, MbCoeffError, MbEntropyCtx, COEFF_BANDS, DEFAULT_COEFF_PROBS,
    MB_ENTROPY_CTX_LEN, ZIGZAG,
};
pub use decoder::{decode_vp8, DecodeError, Vp8DecodedFrame};

/// 0.1.13 alias for [`decoder::Vp8DecodedFrame`] — the per-frame
/// reconstructed I420 picture is now named `Vp8Frame` at the crate root
/// to match the historical `oxideav_vp8::Vp8Frame` path that downstream
/// consumers wrote against. The underlying struct is unchanged.
pub type Vp8Frame = decoder::Vp8DecodedFrame;

/// 0.1.13 alias for the registry-gated [`decoder::Vp8Decoder`] handle.
/// Re-exported at the crate root so `use oxideav_vp8::Vp8Decoder;`
/// works. Standalone callers without `oxideav-core` should use
/// [`state::Vp8DecoderState`] instead.
#[cfg(feature = "registry")]
pub use decoder::Vp8Decoder;

/// Legacy alias of [`decode_vp8`] yielding the framework's
/// [`oxideav_core::VideoFrame`] instead of [`Vp8Frame`]. Gated on the
/// `registry` feature; standalone callers should use [`decode_vp8`].
#[cfg(feature = "registry")]
pub fn decode_frame(
    bytes: &[u8],
) -> core::result::Result<oxideav_core::VideoFrame, decoder::DecodeError> {
    decode_vp8(bytes).map(Into::into)
}
pub use dequant::{decode_and_dequantize_mb, MbDequantFactors, UV_DC_MAX, Y2_AC_MIN};
pub use encoder::{
    classify_coeff_token, count_block_branches, count_inter_frame_branches,
    count_keyframe_branches, count_mb_branches, empty_branch_counts, encode_coeff_block,
    encode_keyframe, encode_keyframe_with_fitted_token_prob_updates,
    encode_keyframe_with_reconstruction,
    encode_keyframe_with_reconstruction_and_fitted_token_prob_updates,
    encode_keyframe_with_reconstruction_and_token_updates, encode_keyframe_with_token_prob_updates,
    encode_mb_block_set, encode_mb_block_set_with_neighbors, encode_p_frame_multi_ref,
    encode_p_frame_multi_ref_with_fitted_token_prob_updates,
    encode_p_frame_multi_ref_with_intra_pick, encode_p_frame_multi_ref_with_refresh,
    encode_p_frame_multi_ref_with_refresh_and_intra_pick,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_fitted_token_prob_updates,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_intra_pick_and_fitted_token_prob_updates,
    encode_p_frame_multi_ref_with_refresh_and_lf_deltas_and_token_updates,
    encode_p_frame_multi_ref_with_token_updates, encode_p_frame_zero_mv, encode_silent_keyframe,
    fit_token_prob_updates, make_silent_keyframe_encoder, patch_first_partition_size,
    quality_to_qindex, write_frame_tag, write_loop_filter, write_loop_filter_with_deltas,
    write_mb_no_skip_coeff, write_no_token_prob_updates, write_quant_indices,
    write_segment_update_flags, write_token_partition_count, write_token_prob_updates, BoolEncoder,
    BranchCounts, CopyBufferSelector, EncodeError, EncodedMb, I420Frame, KeyframeParams,
    LoopFilterDeltaSlot, LoopFilterDeltas, MbPixels, RefreshControls,
    ScaleCode as EncoderScaleCode, SilentKeyframeEncoder, SilentKeyframeParams, TokenEncodeError,
    TokenEncoder,
};

/// Framework `make_encoder` / `make_encoder_with_qindex` /
/// `make_encoder_with_quality` re-exports (registry-gated; require
/// `oxideav-core`). These mirror the `encoder::` module path so
/// `use oxideav_vp8::make_encoder;` works for binding-compatible
/// downstream code (notably `oxideav-webp`'s lossy VP8 path).
#[cfg(feature = "registry")]
pub use encoder::{make_encoder, make_encoder_with_qindex, make_encoder_with_quality};
pub use forward_transform::{forward_dct_4x4, forward_wht_4x4, raster_to_scan};
pub use frame::{decode_keyframe, FrameError, KeyframePlanes, MbCoeffs};
pub use frame_header::{
    FrameHeaderError, LoopFilterPolicy, ReconstructionFilter, ScaleCode, Vp8FrameHeader,
    KEY_FRAME_START_CODE,
};
pub use intra_predict::{
    predict_b4x4, predict_uv8x8, predict_uv8x8_dc, predict_uv8x8_h, predict_uv8x8_tm,
    predict_uv8x8_v, predict_y16x16, predict_y16x16_dc, predict_y16x16_h, predict_y16x16_tm,
    predict_y16x16_v, DEFAULT_ABOVE_PIXEL, DEFAULT_LEFT_PIXEL, DEFAULT_TOPLEFT_DC,
};
pub use inverse_transform::{
    add_residue, add_residue_4x4, clamp255, clamp_qindex, dequant_block, inverse_dct_4x4,
    inverse_wht_4x4, inverse_wht_4x4_dc_only, Y1DequantFactors, AC_QLOOKUP, DC_QLOOKUP,
    QINDEX_RANGE,
};
pub use loop_filter::{
    calculate_mb_filter_level, clamp_s8, common_adjust, filter_frame, mb_filter, s2u,
    simple_segment, subblock_filter, u2s, FrameFilterConfig, LoopFilterParams, MAX_MB_SEGMENTS,
    MAX_MODE_LF_DELTAS, MAX_REF_LF_DELTAS,
};
pub use macroblock::{
    parse_inter_frame_intra_macroblock_modes, parse_key_frame_macroblock_modes,
    InterFrameIntraProbs, IntraBmode, IntraUvMode, IntraYMode, MacroblockError, MacroblockModes,
    IF_BMODE_PROB, IF_UV_MODE_PROB_DEFAULTS, IF_YMODE_PROB_DEFAULTS,
};
pub use motion_comp::{
    apply_full_pixel, chroma_idx_for_luma_subblock, chroma_mv, fetch_block_halo,
    fetch_block_whole_pixel, filter_block_4x4, filter_set_for_version, interp, predict_inter_mb,
    predict_inter_mb_whole_pixel, predict_split_mv, reconstruct_inter_mb,
    reconstruct_inter_mb_whole_pixel, reconstruct_split_mv_mb, select_ref_frame, sixtap_2d,
    split_chroma_mvs, stored_luma_mv, whole_pixel_fraction_is_zero, FilterSet, MotionCompError,
    RefFrame, ReferencePlanes, BILINEAR_FILTERS, SIXTAP_FILTERS,
};
pub use motion_search::{
    block_sad_16x16, half_pixel_refine_luma, mb_luma_sad_at_mv, mb_luma_sad_at_whole_mv,
    quarter_pixel_refine_luma, small_diamond_search_luma, LumaRef, SearchResult, HALF_PIXEL_STEP,
    MV_MAX, MV_MIN, QUARTER_PIXEL_STEP, WHOLE_PIXEL_STEP,
};
pub use motion_vector::{
    default_mv_contexts, mv_bits, mv_component_bits, read_mv, read_mv_component,
    resolve_mv_contexts, write_mv, write_mv_component, Mv, MvContext, MvContexts, SMALL_MVTREE,
};
pub use near_mv::{
    above_block_mv, clamp_mv, decode_inter_mb, decode_split_mv, decode_split_mv_mb, find_near_mvs,
    left_block_mv, mv_ref_probs, partition_id, read_inter_mode, read_mv_partition,
    resolve_inter_mb_mv, submv_ref, submv_ref_context, InterMbError, InterMode, MbInfo,
    MvClampRect, MvPartition, NearMvs, ResolvedInterMode, SignBias, SplitMvResult, SubMvRefMode,
    MV_COUNTS_TO_PROBS, MV_PARTITIONS, MV_PARTITION_PROBS, MV_PARTITION_TREE, MV_REF_TREE,
    SUBMV_REF_PROBS, SUBMV_REF_TREE,
};
pub use reconstruct::{
    decode_keyframe_mb_bpred, decode_keyframe_mb_non_bpred, MbNeighbors, ReconstructError,
    ReconstructedMb,
};
pub use state::{RefFrameSlot, Vp8DecoderState};
pub use stream::{
    EncodedStreamFrame, FrameKind, StreamEncodeError, Vp8InterStreamEncoder,
    Vp8KeyframeStreamEncoder,
};

/// Crate-local error type for the not-yet-implemented surfaces.
///
/// The decoder path (`decode_vp8`) has its own richer
/// [`decoder::DecodeError`]; this type now only backs the encoder stub,
/// which is still scaffolded pending the §13 / §14 encode round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The requested operation is not wired up yet (currently: the
    /// encoder). The decode path is implemented for key frames — see
    /// [`decode_vp8`] / [`decoder::DecodeError`].
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "oxideav-vp8: requested operation not implemented")
    }
}

impl std::error::Error for Error {}

/// Unified top-level error surface for the crate — re-exported here
/// from [`error::Vp8Error`] so downstream consumers can write
/// `use oxideav_vp8::Vp8Error;` without naming the submodule.
///
/// See the [`error`] module for the variant shape, conversion adapters,
/// and the rationale behind the flat-four design.
pub use error::Vp8Error;

/// Crate-wide [`Result`](core::result::Result) alias — every fallible
/// entry point that surfaces a [`Vp8Error`] uses this.
pub use error::Result;

// ── 0.1.13 frame_header / frame_tag re-exports ──
pub use frame_tag::{
    parse_header, parse_keyframe_header, FrameTag, FrameType, KeyframeHeader, ParsedHeader,
};

/// 0.1.13 alias of [`frame_header::Vp8FrameHeader`] under the historical
/// name `FrameHeader`. The struct is unchanged.
pub type FrameHeader = frame_header::Vp8FrameHeader;

// ── 0.1.13 encoder surface re-exports ──
pub use encoder::{
    first_pass_analyze, make_encoder_typed_with_config, make_two_pass_encoder,
    two_pass_qindex_for_frame, two_pass_qindices, FrameComplexity, LoopFilterMode, Vp8Encoder,
    Vp8EncoderConfig, Vp8EncoderStats, Vp8TwoPassConfig, Vp8TwoPassEncoder, AQ_QINDEX_RANGE_MAX,
    DEFAULT_ADAPTIVE_SEGMENT_THRESHOLDS, DEFAULT_ALT_REF_INTERVAL, DEFAULT_AQ_QINDEX_RANGE,
    DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256, DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
    DEFAULT_GOLDEN_INTERVAL, DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
    DEFAULT_KMEANS_CONVERGENCE_THRESHOLD, DEFAULT_KMEANS_SPATIAL_ALPHA_X256,
    DEFAULT_LAMBDA_LONG_REF_SCALE_X256, DEFAULT_LOOKAHEAD_WINDOW, DEFAULT_NLM_H2,
    DEFAULT_PSY_RD_STRENGTH, DEFAULT_QINDEX, DEFAULT_SCENE_CUT_BOOST_FRAMES,
    DEFAULT_SCENE_CUT_QUANT_BOOST, DEFAULT_SCENE_CUT_THRESHOLD, DEFAULT_SEGMENT_LF_DELTAS,
    DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL, DEFAULT_SPATIAL_LF_N_COL_BANDS,
    DEFAULT_SPATIAL_LF_N_ROW_BANDS, DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES,
    INTRA_IN_P_BPRED_VARIANCE_THRESHOLD, JOINT_R44R49_PICKER_MAX_ITERS_MAX,
    KMEANS_SPATIAL_MAX_ITERS, LAMBDA_SCALE_DEFAULT, QP_SENSITIVITY_X8, SCENE_CUT_ABS_FLOOR,
    SEGMENT_VARIANCE_THRESHOLDS, SPLIT_MV_JOINT_REFINE_PASSES_MAX,
};

#[cfg(feature = "registry")]
pub use encoder::make_encoder_with_config;

/// Crate-root codec id string. Matches the registry id installed by
/// the `oxideav_core::register!` hook below and the
/// `oxideav-vp8 0.1.13` constant of the same name.
pub const CODEC_ID_STR: &str = "vp8";

// ── 0.1.13 registry surface re-exports ──
#[cfg(feature = "registry")]
pub use registry::{register_codecs, register_containers};

/// Encode a VP8 keyframe.
///
/// Phase 1 of the encoder is implemented: this routes to
/// [`encode_silent_keyframe`], which emits a structurally-valid
/// all-zero-quantization key frame for the supplied dimensions
/// (`_pixels` is currently ignored — the §13 / §14 encode round will
/// wire in actual pixel-driven residual coding).
///
/// Errors are surfaced as the legacy crate-level [`Error`] (currently
/// only [`Error::NotImplemented`]) for backwards compatibility with
/// callers that already pattern-match against it. To consume the
/// richer [`EncodeError`] surface directly, call
/// [`encode_silent_keyframe`] instead.
pub fn encode_vp8_keyframe(
    _pixels: &[u8],
    width: u32,
    height: u32,
) -> core::result::Result<Vec<u8>, Error> {
    encode_silent_keyframe(SilentKeyframeParams::new(width, height))
        .map_err(|_| Error::NotImplemented)
}

/// Install the VP8 codec (decoder) into a runtime context. Delegates to
/// [`decoder::register`].
#[cfg(feature = "registry")]
pub use decoder::register;

#[cfg(feature = "registry")]
oxideav_core::register!("vp8", register);
