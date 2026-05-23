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
//!   quantised coefficients. No per-macroblock walker yet. See
//!   [`dct_tokens`].
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
//!   operating on a caller-supplied pixel window; the §15.1
//!   raster-order edge geometry is the integration round's job. See
//!   [`loop_filter`].
//!
//! Motion-vector decoding (§17), the per-macroblock reconstruction walk
//! (including the §15.1 loop-filter geometry that drives the
//! [`loop_filter`] segment kernels), and the encoder are all still
//! scaffolded — the top-level `decode_vp8` / `encode_vp8_*` entry
//! points return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

pub mod bool_decoder;
pub mod coded_header;
pub mod dct_tokens;
pub mod frame_header;
pub mod intra_predict;
pub mod inverse_transform;
pub mod loop_filter;
pub mod macroblock;

pub use bool_decoder::{BoolDecoder, BoolDecoderError};
pub use coded_header::{
    CodedHeaderError, MbLfAdjustments, MvProbUpdates, QuantIndices, TokenProbUpdates,
    UpdateSegmentation, Vp8CodedHeader, DEFAULT_MV_CONTEXT, MV_PROB_COUNT,
};
pub use dct_tokens::{
    decode_block, merge_default_token_probs, BlockType, CoeffProbs, DctToken, DctTokenError,
    COEFF_BANDS, DEFAULT_COEFF_PROBS,
};
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
    clamp_s8, common_adjust, mb_filter, s2u, simple_segment, subblock_filter, u2s, LoopFilterParams,
};
pub use macroblock::{
    parse_key_frame_macroblock_modes, IntraBmode, IntraUvMode, IntraYMode, MacroblockError,
    MacroblockModes,
};

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

/// Crate-local error type. Until the clean-room rebuild lands every
/// public API path returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "oxideav-vp8: orphan-rebuild scaffold — no decoder/encoder wired up"
        )
    }
}

impl std::error::Error for Error {}

/// Decode a VP8 elementary stream.
pub fn decode_vp8(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode a VP8 keyframe.
pub fn encode_vp8_keyframe(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the orphan-rebuild scaffold registers
/// nothing into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("vp8", register);
