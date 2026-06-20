//! End-to-end exercise of the [`Vp8TwoPassEncoder`] family.
//!
//! Builds a small synthetic four-frame Yuv420P clip with deliberately
//! varying complexity (solid → gradient → checkerboard → pseudo-random
//! noise), runs the first-pass analyser, asserts the qindex schedule
//! responds to the complexity stats (lower qindex for heavier frames),
//! then drives the second-pass encoder + the in-tree
//! [`oxideav_vp8::decode_vp8`] decoder to confirm every emitted bitstream
//! is decodable.
//!
//! The algorithm is documented in the rustdoc on
//! [`oxideav_vp8::Vp8TwoPassEncoder`]; the tests below assert the
//! observable shape of that algorithm:
//!
//! * `first_pass_analyze` returns one record per input frame,
//!   `frame_index` matching the input position.
//! * The qindex schedule emitted by `two_pass_qindices` is not
//!   monotonically constant — heavier frames receive lower qindex than
//!   lighter frames (this is the whole point of the complexity-aware
//!   scheduler; without it the test would fail).
//! * Every emitted frame decodes through `decode_vp8` (first frame is a
//!   key frame; the rest decode as P frames against the running
//!   reference inside the driver).
//! * The free-function `two_pass_qindex_for_frame` returns a value
//!   inside the RFC 6386 §9.6 `0..=127` range and applies the
//!   scene-cut boost.

use oxideav_vp8::{
    decode_vp8, first_pass_analyze, two_pass_qindex_for_frame, two_pass_qindices, FrameComplexity,
    I420Frame, Vp8DecoderState, Vp8TwoPassConfig, Vp8TwoPassEncoder,
};

/// Build a 32×32 solid-grey Yuv420P frame.  Lowest complexity test
/// frame; first_pass should land near zero cost.
fn solid_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    (
        vec![128u8; w * h],
        vec![128u8; cw * ch],
        vec![128u8; cw * ch],
    )
}

/// Build a 32×32 horizontal-gradient Yuv420P frame.  Modest spatial
/// variance, low motion vs the previous solid frame; complexity will
/// land above solid but well below noise.
fn gradient_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = (c * 8).min(255) as u8;
        }
    }
    (y, vec![120u8; cw * ch], vec![136u8; cw * ch])
}

/// Build a 32×32 checkerboard Yuv420P frame.  High spatial variance
/// and large per-pixel deviation from the previous gradient frame.
fn checker_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // 4×4 tile checker so the variance is large but the
            // intra-tile content is uniform.
            let block = ((r / 4) + (c / 4)) & 1;
            y[r * w + c] = if block == 0 { 32 } else { 224 };
        }
    }
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

/// Build a 32×32 pseudo-random noise Yuv420P frame.  Highest cost in
/// the test set — every pixel diverges from the previous frame's
/// checkerboard, and the per-frame variance saturates.
fn noise_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (32usize, 32usize);
    let (cw, ch) = (16usize, 16usize);
    let mut y = vec![0u8; w * h];
    let mut state: u32 = 0xc0ffee01;
    for px in y.iter_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *px = (state >> 16) as u8;
    }
    (y, vec![128u8; cw * ch], vec![128u8; cw * ch])
}

/// Run the four-frame synthetic clip end to end and return the
/// per-frame stats + qindex schedule + emitted bitstream lengths.
fn run_four_frame_clip(config: Vp8TwoPassConfig) -> (Vec<FrameComplexity>, Vec<u8>, Vec<Vec<u8>>) {
    let frames = [
        solid_frame(),
        gradient_frame(),
        checker_frame(),
        noise_frame(),
    ];
    let i420: Vec<I420Frame<'_>> = frames
        .iter()
        .map(|(y, u, v)| I420Frame::packed(32, 32, y, u, v))
        .collect();

    // First pass.
    let stats = first_pass_analyze(&i420, &config).expect("first_pass_analyze must succeed");
    assert_eq!(
        stats.len(),
        i420.len(),
        "first_pass_analyze must produce one record per input frame"
    );
    for (i, s) in stats.iter().enumerate() {
        assert_eq!(
            s.frame_index as usize, i,
            "frame_index must equal slice index"
        );
    }

    // Qindex schedule.
    let schedule = two_pass_qindices(&config, &stats).expect("two_pass_qindices must succeed");
    assert_eq!(
        schedule.len(),
        stats.len(),
        "schedule length must match stats"
    );
    for &qi in &schedule {
        assert!(qi <= 127, "schedule must respect RFC 6386 §9.6 0..=127");
    }

    // Second pass: drive the encoder + decode every frame.
    let mut encoder = Vp8TwoPassEncoder::new(config);
    let stats_via_encoder = encoder
        .first_pass_analyze(&i420)
        .expect("encoder first_pass_analyze must succeed");
    assert_eq!(
        stats_via_encoder, stats,
        "encoder.first_pass_analyze must agree with the free function"
    );
    // Drive the stateful decoder so P-frames decode against the
    // previous key/P reconstruction; `decode_vp8` is keyframe-only
    // and would reject the inter frames in this stream.
    let mut decoder_state = Vp8DecoderState::new();
    let mut bytes_out: Vec<Vec<u8>> = Vec::new();
    for (i, (frame, stat)) in i420.iter().zip(stats.iter()).enumerate() {
        let bytes = encoder
            .encode_frame(frame, *stat)
            .expect("encode_frame must succeed");
        assert!(!bytes.is_empty(), "encoded frame must carry bytes");
        let decoded = decoder_state
            .decode_frame(&bytes)
            .unwrap_or_else(|e| panic!("frame {i} must decode through Vp8DecoderState: {e:?}"));
        assert_eq!(decoded.width, 32);
        assert_eq!(decoded.height, 32);
        // First frame must be a keyframe — confirm via a standalone
        // decode_vp8 (which only accepts keyframes).  Subsequent frames
        // are P-frames that the standalone path will reject.
        if i == 0 {
            let kf_decoded = decode_vp8(&bytes).expect("first frame must be a keyframe");
            assert_eq!(kf_decoded.width, 32);
        }
        bytes_out.push(bytes);
    }
    (stats, schedule, bytes_out)
}

#[test]
fn first_pass_analyze_returns_one_record_per_frame() {
    let (stats, _schedule, _bytes) = run_four_frame_clip(Vp8TwoPassConfig::default());
    assert_eq!(stats.len(), 4);
    // Solid frame must be the cheapest; noise frame must be the most
    // expensive.  (Checker is in between — high variance but a
    // structured per-pixel difference.)
    let solid_cost = stats[0].bits_per_mb;
    let noise_cost = stats[3].bits_per_mb;
    assert!(
        noise_cost > solid_cost,
        "noise frame ({noise_cost}) must cost more than solid frame ({solid_cost})"
    );
}

#[test]
fn qindex_schedule_responds_to_complexity() {
    let (stats, schedule, _bytes) = run_four_frame_clip(Vp8TwoPassConfig::default());
    // Find the highest-cost and lowest-cost frames.  Their qindices
    // must obey: heavier ⇒ smaller qindex.  (Equal cost ⇒ equal
    // qindex; we require at least one strict pair.)
    let (mut hi_idx, mut lo_idx) = (0usize, 0usize);
    for (i, s) in stats.iter().enumerate() {
        if s.bits_per_mb > stats[hi_idx].bits_per_mb {
            hi_idx = i;
        }
        if s.bits_per_mb < stats[lo_idx].bits_per_mb {
            lo_idx = i;
        }
    }
    assert!(
        schedule[hi_idx] < schedule[lo_idx],
        "heavier frame {} ({}) should get smaller qindex ({}) than lighter frame {} ({}) qindex ({})",
        hi_idx,
        stats[hi_idx].bits_per_mb,
        schedule[hi_idx],
        lo_idx,
        stats[lo_idx].bits_per_mb,
        schedule[lo_idx]
    );
}

#[test]
fn schedule_is_not_trivially_constant_across_the_clip() {
    // The whole point of the two-pass scheduler is that it varies the
    // qindex across the GOP.  A schedule that returns a constant for
    // every frame would mean the algorithm is broken (or the test
    // input is degenerate; the four-frame clip above is deliberately
    // chosen to span solid → noise so the scheduler must respond).
    let (_stats, schedule, _bytes) = run_four_frame_clip(Vp8TwoPassConfig::default());
    let min = *schedule.iter().min().unwrap();
    let max = *schedule.iter().max().unwrap();
    assert!(
        max > min,
        "schedule must not be constant across a complexity-varied clip: {schedule:?}"
    );
}

#[test]
fn every_emitted_frame_decodes_via_stateful_decoder() {
    // The run_four_frame_clip helper already asserts the stateful
    // Vp8DecoderState succeeds for every frame; this test exists to
    // surface the invariant as a discrete CI signal so a regression
    // in either the encoder OR the decoder fails this test by name
    // rather than hiding inside the helper.
    let (_stats, _schedule, bytes_out) = run_four_frame_clip(Vp8TwoPassConfig::default());
    assert_eq!(bytes_out.len(), 4);
    let mut decoder = Vp8DecoderState::new();
    for (i, bytes) in bytes_out.iter().enumerate() {
        let decoded = decoder.decode_frame(bytes).expect("re-decode must succeed");
        assert_eq!(decoded.width, 32, "frame {i} width mismatch");
        assert_eq!(decoded.height, 32, "frame {i} height mismatch");
        assert_eq!(decoded.y.len(), 32 * 32, "frame {i} luma plane size");
    }
}

#[test]
fn two_pass_qindex_for_frame_handles_baseline_complexity() {
    let config = Vp8TwoPassConfig::default();
    let complexity = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 100.0,
        scene_cut: false,
    };
    let qi = two_pass_qindex_for_frame(&config, complexity).expect("stateless picker must succeed");
    // With no GOP-mean reference the per-frame delta collapses to zero,
    // so the picker must return the base qindex unchanged.
    assert_eq!(qi, config.base.qindex);
}

#[test]
fn two_pass_qindex_for_frame_applies_scene_cut_boost() {
    let mut config = Vp8TwoPassConfig::default();
    config.base.qindex = 60;
    let baseline = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 100.0,
        scene_cut: false,
    };
    let cut = FrameComplexity {
        scene_cut: true,
        ..baseline
    };
    let qi_baseline = two_pass_qindex_for_frame(&config, baseline).unwrap();
    let qi_cut = two_pass_qindex_for_frame(&config, cut).unwrap();
    assert!(
        qi_cut < qi_baseline,
        "scene-cut frame must receive a lower qindex (better quality): \
         baseline={qi_baseline} cut={qi_cut}"
    );
}

#[test]
fn two_pass_qindex_for_frame_rejects_out_of_range_base() {
    let mut config = Vp8TwoPassConfig::default();
    config.base.qindex = 200; // Beyond RFC 6386 §9.6 0..=127.
    let complexity = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 100.0,
        scene_cut: false,
    };
    let r = two_pass_qindex_for_frame(&config, complexity);
    assert!(
        r.is_err(),
        "out-of-range base qindex must surface InvalidData"
    );
}

#[test]
fn two_pass_qindices_empty_input_yields_empty_schedule() {
    let config = Vp8TwoPassConfig::default();
    let schedule = two_pass_qindices(&config, &[]).expect("empty input must succeed");
    assert!(schedule.is_empty());
}

/// Encode the four-frame clip under the closed-loop bitrate controller
/// at a tight vs a generous target, and confirm: (1) the tight target
/// never emits more bytes than the generous one (the solver picks a
/// coarser quantiser under pressure), and (2) every frame still decodes
/// through the stateful decoder under both targets.
#[test]
fn closed_loop_bitrate_target_is_monotone_and_decodable() {
    let frames = [
        solid_frame(),
        gradient_frame(),
        checker_frame(),
        noise_frame(),
    ];
    let i420: Vec<I420Frame<'_>> = frames
        .iter()
        .map(|(y, u, v)| I420Frame::packed(32, 32, y, u, v))
        .collect();

    // Encode the clip at a given bitrate; return the emitted byte total.
    // Every frame is decoded so the test fails loudly on a bad stream.
    let encode_at = |bps: u32| -> usize {
        let config = Vp8TwoPassConfig {
            target_bitrate_bps: bps,
            fps_num: 30,
            fps_den: 1,
            ..Vp8TwoPassConfig::default()
        };
        assert!(config.bitrate_targeted(), "config must be bitrate-targeted");
        let mut enc = Vp8TwoPassEncoder::new(config);
        let stats = enc.first_pass_analyze(&i420).unwrap();
        let mut decoder = Vp8DecoderState::new();
        let mut total = 0usize;
        for (i, (f, s)) in i420.iter().zip(stats.iter()).enumerate() {
            let bytes = enc.encode_frame(f, *s).unwrap();
            total += bytes.len();
            let decoded = decoder
                .decode_frame(&bytes)
                .unwrap_or_else(|e| panic!("frame {i} @ {bps}bps must decode: {e:?}"));
            assert_eq!(decoded.width, 32);
            assert_eq!(decoded.height, 32);
        }
        total
    };

    // A starved target must not emit more than a generous one. (For a
    // tiny 32×32 clip the per-frame VP8 header dominates, so equality is
    // permitted; the load-bearing invariant is monotonicity.)
    let tight = encode_at(2_000); // 250 B/s — very tight
    let generous = encode_at(2_000_000); // 250 kB/s — effectively unconstrained
    assert!(
        tight <= generous,
        "tight bitrate target ({tight} B) must not emit more than the generous \
         target ({generous} B)"
    );
}

/// A bitrate target absent a frame rate must NOT engage the solver —
/// `bitrate_targeted()` is false, so the schedule equals the plain
/// constant-quality encode (the rate is undefined without an fps).
#[test]
fn bitrate_without_frame_rate_stays_constant_quality() {
    let frames = [solid_frame(), gradient_frame()];
    let i420: Vec<I420Frame<'_>> = frames
        .iter()
        .map(|(y, u, v)| I420Frame::packed(32, 32, y, u, v))
        .collect();

    let config = Vp8TwoPassConfig {
        target_bitrate_bps: 50_000,
        fps_num: 0, // no rate → solver disabled
        ..Vp8TwoPassConfig::default()
    };
    assert!(!config.bitrate_targeted());

    let mut enc = Vp8TwoPassEncoder::new(config);
    let stats = enc.first_pass_analyze(&i420).unwrap();
    // No bitrate solving ⇒ zero accumulated debt throughout.
    let mut decoder = Vp8DecoderState::new();
    for (f, s) in i420.iter().zip(stats.iter()) {
        let bytes = enc.encode_frame(f, *s).unwrap();
        decoder.decode_frame(&bytes).expect("frame must decode");
    }
    assert_eq!(
        enc.rate_debt_bytes(),
        0,
        "untargeted encode must accumulate no rate debt"
    );
}

#[test]
fn encoder_fallback_without_first_pass_uses_base_qindex() {
    // If a caller skips first_pass_analyze, encode_frame must still
    // run — it falls back to no-delta picking (base qindex).  The
    // bytestream must still be a decodable VP8 keyframe.
    let mut encoder = Vp8TwoPassEncoder::new(Vp8TwoPassConfig::default());
    let (y, u, v) = solid_frame();
    let frame = I420Frame::packed(32, 32, &y, &u, &v);
    let complexity = FrameComplexity {
        frame_index: 0,
        bits_per_mb: 100.0,
        scene_cut: false,
    };
    let bytes = encoder
        .encode_frame(&frame, complexity)
        .expect("encode without first_pass must still succeed");
    let decoded = decode_vp8(&bytes).expect("fallback path must emit decodable bytes");
    assert_eq!(decoded.width, 32);
    assert_eq!(decoded.height, 32);
}
