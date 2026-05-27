//! Compile-only assertion that the crates.io `oxideav-vp8 0.1.13`
//! public surface stays reachable on the post-0.2 master.
//!
//! Every public symbol enumerated by the published 0.1.13 surface is bound
//! here by fully-qualified name. If a future change removes or
//! re-shapes any of them, this file stops compiling — that is the
//! signal to bump the major version and revisit the upgrade-transparent
//! migration contract.
//!
//! The test contains zero runtime assertions; the compile is the
//! assertion.

#![allow(unused_imports, dead_code)]

// ── Crate-root constant ──
use oxideav_vp8::CODEC_ID_STR;

// ── Crate-root error / result ──
use oxideav_vp8::{Result as Vp8Result, Vp8Error};

// ── Crate-root frame container ──
use oxideav_vp8::{Vp8DecodedFrame, Vp8Frame};

// ── Crate-root frame_header / FrameHeader alias ──
use oxideav_vp8::{FrameHeader, Vp8FrameHeader};

// ── Crate-root frame_tag re-exports ──
use oxideav_vp8::{
    parse_header, parse_keyframe_header, FrameTag, FrameType, KeyframeHeader, ParsedHeader,
};

// ── Crate-root decoder re-exports ──
use oxideav_vp8::{decode_vp8, DecodeError};

// ── Crate-root encoder surface ──
use oxideav_vp8::{
    encode_vp8_keyframe, first_pass_analyze, make_encoder_typed_with_config, make_two_pass_encoder,
    two_pass_qindex_for_frame, two_pass_qindices, FrameComplexity, LoopFilterMode, Vp8Encoder,
    Vp8EncoderConfig, Vp8EncoderStats, Vp8TwoPassConfig, Vp8TwoPassEncoder,
};

// ── Crate-root encoder constants ──
use oxideav_vp8::{
    AQ_QINDEX_RANGE_MAX, DEFAULT_ADAPTIVE_SEGMENT_THRESHOLDS, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_AQ_QINDEX_RANGE, DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
    DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256, DEFAULT_GOLDEN_INTERVAL,
    DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS, DEFAULT_KMEANS_CONVERGENCE_THRESHOLD,
    DEFAULT_KMEANS_SPATIAL_ALPHA_X256, DEFAULT_LAMBDA_LONG_REF_SCALE_X256,
    DEFAULT_LOOKAHEAD_WINDOW, DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH, DEFAULT_QINDEX,
    DEFAULT_SCENE_CUT_BOOST_FRAMES, DEFAULT_SCENE_CUT_QUANT_BOOST, DEFAULT_SCENE_CUT_THRESHOLD,
    DEFAULT_SEGMENT_LF_DELTAS, DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL,
    DEFAULT_SPATIAL_LF_N_COL_BANDS, DEFAULT_SPATIAL_LF_N_ROW_BANDS,
    DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES, INTRA_IN_P_BPRED_VARIANCE_THRESHOLD,
    JOINT_R44R49_PICKER_MAX_ITERS_MAX, KMEANS_SPATIAL_MAX_ITERS, LAMBDA_SCALE_DEFAULT,
    QP_SENSITIVITY_X8, SCENE_CUT_ABS_FLOOR, SEGMENT_VARIANCE_THRESHOLDS,
    SPLIT_MV_JOINT_REFINE_PASSES_MAX,
};

// ── 0.1.13 module-path aliases ──
use oxideav_vp8::{
    bool_encoder, fdct, frame_tag, inter, intra, ivf, loopfilter, mv, tables, tokens, transform,
};

// ── registry-gated entries ──
#[cfg(feature = "registry")]
use oxideav_vp8::{decode_frame, make_encoder_with_config, register, registry, Vp8Decoder};

#[test]
fn api_compat_0_1_13_crate_root_constant_value() {
    // Bind every imported item; this also forces the test crate to
    // actually link the symbols rather than just import them in the
    // dead-code-pruned section above.

    // Constants
    let _: &str = CODEC_ID_STR;
    assert_eq!(CODEC_ID_STR, "vp8");

    let _: i32 = AQ_QINDEX_RANGE_MAX;
    let _: [u32; 4] = DEFAULT_ADAPTIVE_SEGMENT_THRESHOLDS;
    let _: u32 = DEFAULT_ALT_REF_INTERVAL;
    let _: i32 = DEFAULT_AQ_QINDEX_RANGE;
    let _: i32 = DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256;
    let _: i32 = DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256;
    let _: u32 = DEFAULT_GOLDEN_INTERVAL;
    let _: u32 = DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS;
    let _: i32 = DEFAULT_KMEANS_CONVERGENCE_THRESHOLD;
    let _: i32 = DEFAULT_KMEANS_SPATIAL_ALPHA_X256;
    let _: i32 = DEFAULT_LAMBDA_LONG_REF_SCALE_X256;
    let _: u32 = DEFAULT_LOOKAHEAD_WINDOW;
    let _: i32 = DEFAULT_NLM_H2;
    let _: i32 = DEFAULT_PSY_RD_STRENGTH;
    let _: u8 = DEFAULT_QINDEX;
    let _: u32 = DEFAULT_SCENE_CUT_BOOST_FRAMES;
    let _: i32 = DEFAULT_SCENE_CUT_QUANT_BOOST;
    let _: i32 = DEFAULT_SCENE_CUT_THRESHOLD;
    let _: [i8; 4] = DEFAULT_SEGMENT_LF_DELTAS;
    let _: [i8; 4] = DEFAULT_SEGMENT_QUANT_DELTAS;
    let _: u8 = DEFAULT_SIMPLE_LF_MAX_LEVEL;
    let _: u32 = DEFAULT_SPATIAL_LF_N_COL_BANDS;
    let _: u32 = DEFAULT_SPATIAL_LF_N_ROW_BANDS;
    let _: u32 = DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES;
    let _: i32 = INTRA_IN_P_BPRED_VARIANCE_THRESHOLD;
    let _: u32 = JOINT_R44R49_PICKER_MAX_ITERS_MAX;
    let _: u32 = KMEANS_SPATIAL_MAX_ITERS;
    let _: i32 = LAMBDA_SCALE_DEFAULT;
    let _: i32 = QP_SENSITIVITY_X8;
    let _: i32 = SCENE_CUT_ABS_FLOOR;
    let _: [u32; 4] = SEGMENT_VARIANCE_THRESHOLDS;
    let _: u32 = SPLIT_MV_JOINT_REFINE_PASSES_MAX;
}

#[test]
fn api_compat_0_1_13_encoder_constructors() {
    let cfg = Vp8EncoderConfig::default();
    let _: Vp8Encoder = Vp8Encoder::new(cfg);
    let _: Vp8Encoder = make_encoder_typed_with_config(cfg);

    // Two-pass family — the bodies are real (round 168 drive-to-100%),
    // so the surface-lock test now checks the live signatures + the
    // documented behaviour: empty input produces an empty schedule and
    // does not error; a single-frame complexity record produces a
    // valid `0..=127` qindex; the encoder factory hands back a working
    // instance.
    let two = Vp8TwoPassConfig::default();
    let mut tp: Vp8TwoPassEncoder = make_two_pass_encoder(two);
    let empty: Vec<FrameComplexity> = tp
        .first_pass_analyze(&[])
        .expect("empty input must succeed");
    assert!(empty.is_empty(), "empty input must produce empty stats");

    let empty_free: Vec<FrameComplexity> =
        first_pass_analyze(&[], &two).expect("free-fn empty input must succeed");
    assert!(empty_free.is_empty());

    let fake = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 0.0,
        scene_cut: false,
    };
    let qi = two_pass_qindex_for_frame(&two, fake).expect("stateless picker must succeed");
    assert!(qi <= 127, "qindex must respect RFC 6386 §9.6 0..=127");
    let sched = two_pass_qindices(&two, &[fake]).expect("schedule must succeed");
    assert_eq!(sched.len(), 1);
    assert!(sched[0] <= 127);

    // LoopFilterMode default is `Auto`.
    let lfm: LoopFilterMode = LoopFilterMode::default();
    assert!(matches!(lfm, LoopFilterMode::Auto));

    let _stats: Vp8EncoderStats = Vp8EncoderStats::default();
}

#[test]
fn api_compat_0_1_13_frame_tag_parse_routes_to_module() {
    // 3-byte interframe tag.
    let parsed: ParsedHeader = parse_header(&[0x11, 0x00, 0x00]).expect("interframe header parses");
    let _: FrameTag = parsed.tag;
    assert!(matches!(parsed.tag.frame_type, FrameType::Inter));
    assert!(parsed.keyframe.is_none());

    // Module-aliased path identical to crate root.
    let parsed2 = frame_tag::parse_header(&[0x11, 0x00, 0x00]).unwrap();
    assert_eq!(parsed, parsed2);

    // Reject interframe in keyframe-only parser.
    assert!(parse_keyframe_header(&[0x11, 0x00, 0x00]).is_err());
}

#[test]
fn api_compat_0_1_13_module_path_aliases_resolve() {
    // Each module alias must resolve a known symbol via its 0.1.13 path.
    let _ = fdct::forward_dct_4x4;
    let _ = transform::inverse_dct_4x4;
    let _ = intra::predict_y16x16_dc;
    let _ = loopfilter::LoopFilterParams::derive;
    let _ = mv::default_mv_contexts;
    let _ = tokens::decode_block;
    let _: [usize; 16] = tables::ZIGZAG;
    let _ = inter::select_ref_frame;
    let _ = bool_encoder::classify_coeff_token;
    let _ = ivf::write_header;
}

#[test]
fn api_compat_0_1_13_ivf_roundtrip_smoke() {
    let header = ivf::IvfHeader::vp8(16, 16, 30, 1);
    let bytes = ivf::write_header(&header);
    let parsed = ivf::parse_header(&bytes).unwrap();
    assert_eq!(parsed.width, 16);
    assert_eq!(parsed.height, 16);
}

#[cfg(feature = "registry")]
#[test]
fn api_compat_0_1_13_registry_entries_link() {
    // Linkage check — every registry-side fn pointer resolves.
    let _: fn(&mut oxideav_core::RuntimeContext) = register;
    let _: fn(&mut oxideav_core::CodecRegistry) = registry::register_codecs;
    let _: fn(&mut oxideav_core::RuntimeContext) = registry::register_containers;
    let _: fn(&[u8]) -> core::result::Result<oxideav_core::VideoFrame, DecodeError> = decode_frame;
}

#[test]
fn api_compat_0_1_13_frame_header_alias_compiles() {
    let _: Option<FrameHeader> = None::<Vp8FrameHeader>;
}

#[test]
fn api_compat_0_1_13_decoded_frame_alias_is_identity() {
    // `Vp8Frame` and `Vp8DecodedFrame` must be the same type.
    fn is_same<T: 'static>(_: &T) -> bool {
        true
    }
    // The compile check is "if Vp8Frame == Vp8DecodedFrame then the
    // following is_same call resolves uniquely." Construction is not
    // needed to assert type equality at compile time, but we provide
    // it for the test runner to see.
    let _: Option<Vp8Frame> = None::<Vp8DecodedFrame>;
    let _: Option<Vp8DecodedFrame> = None::<Vp8Frame>;
}
