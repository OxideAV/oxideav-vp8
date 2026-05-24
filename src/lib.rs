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
//! With the §13.3 per-MB token walk ([`dct_tokens::decode_mb_coeffs`]),
//! the §14.1 dequant scaling ([`dequant::decode_and_dequantize_mb`]), and
//! the top-level [`decode_vp8`] driver now landed, the keyframe decode
//! chain bitstream → dequant → reconstruct → loop-filter → pixels is
//! complete end-to-end. Motion-vector decoding (§17), the inter-predicted
//! §16.2 / §16.3 / §16.4 branch (`mv_ref` tree, near/nearest/best census,
//! split prediction), and the encoder are all still scaffolded — the
//! `encode_vp8_*` entry point returns [`Error::NotImplemented`] and
//! interframes return [`decoder::DecodeError::Unsupported`].

#![warn(missing_debug_implementations)]

pub mod bool_decoder;
pub mod coded_header;
pub mod dct_tokens;
pub mod decoder;
pub mod dequant;
pub mod frame;
pub mod frame_header;
pub mod intra_predict;
pub mod inverse_transform;
pub mod loop_filter;
pub mod macroblock;
pub mod reconstruct;

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
pub use dequant::{decode_and_dequantize_mb, MbDequantFactors, UV_DC_MAX, Y2_AC_MIN};
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
pub use reconstruct::{
    decode_keyframe_mb_bpred, decode_keyframe_mb_non_bpred, MbNeighbors, ReconstructError,
    ReconstructedMb,
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

/// Encode a VP8 keyframe.
///
/// Still scaffolded — the encoder has not been written yet (the §13 /
/// §14 encode side is a future round). Returns [`Error::NotImplemented`].
pub fn encode_vp8_keyframe(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Install the VP8 codec (decoder) into a runtime context. Delegates to
/// [`decoder::register`].
#[cfg(feature = "registry")]
pub use decoder::register;

#[cfg(feature = "registry")]
oxideav_core::register!("vp8", register);
