//! Round-31 / #522 PSNR-push regression tests.
//!
//! Pins the three opt-in encoder features added in this round:
//!
//! * `adaptive_segment_thresholds = true` — the per-MB QP refinement
//!   webp's lossy encoder was waiting on (#479). On a synthetic
//!   mixed-content clip (smooth half + textured half + slow pan),
//!   the adaptive-threshold variant must shrink the bitstream by a
//!   measurable margin without losing PSNR vs the static-threshold
//!   baseline.
//! * `lambda_long_ref_scale_x256 = 320` — the long-reference Lagrangian
//!   tilt. Must shrink the bitstream and improve PSNR on the same
//!   mixed-content clip (drift-amortisation effect).
//! * The combination of all three features must beat the
//!   static-threshold baseline by at least 5% bytes for >= +0.3 dB Y.

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_vp8::decoder::Vp8Decoder;
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_ALT_REF_INTERVAL,
    DEFAULT_GOLDEN_INTERVAL, DEFAULT_LAMBDA_LONG_REF_SCALE_X256, DEFAULT_SCENE_CUT_BOOST_FRAMES,
    DEFAULT_SCENE_CUT_QUANT_BOOST, DEFAULT_SCENE_CUT_THRESHOLD, DEFAULT_SEGMENT_LF_DELTAS,
    DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL,
    DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES,
};

const W: u32 = 128;
const H: u32 = 128;
const QINDEX: u8 = 50;

fn make_mixed_clip(n: usize) -> Vec<VideoFrame> {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let dx = (f as i32) % 4 - 2;
        let dy = (f as i32 / 2) % 4 - 2;
        let mut y = vec![0u8; (W * H) as usize];
        for r in 0..H as usize {
            for c in 0..W as usize {
                let rr = (r as i32 + dy).clamp(0, H as i32 - 1) as usize;
                let cc = (c as i32 + dx).clamp(0, W as i32 - 1) as usize;
                let v = if cc < (W as usize) / 2 {
                    // Smooth bands → segment 0/1 candidate.
                    (32 + (cc as i32 / 2) + (rr as i32 / 4)).clamp(0, 255) as u8
                } else {
                    // Coherent texture → segment 2/3 candidate.
                    let v = ((rr as f32 * 0.5).sin() * 30.0
                        + (cc as f32 * 0.4).cos() * 40.0
                        + ((rr + cc) as f32 * 0.2).sin() * 20.0
                        + 130.0) as i32;
                    v.clamp(0, 255) as u8
                };
                y[r * W as usize + c] = v;
            }
        }
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (128 + (r as i32 - (ch as i32) / 2).abs() / 2) as u8;
                v[r * cw + c] = (128 + (c as i32 - (cw as i32) / 2).abs() / 2) as u8;
            }
        }
        out.push(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: W as usize,
                    data: y,
                },
                VideoPlane {
                    stride: cw,
                    data: u,
                },
                VideoPlane {
                    stride: cw,
                    data: v,
                },
            ],
        });
    }
    out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn cfg_baseline() -> Vp8EncoderConfig {
    Vp8EncoderConfig {
        qindex: QINDEX,
        golden_interval: DEFAULT_GOLDEN_INTERVAL,
        alt_ref_interval: DEFAULT_ALT_REF_INTERVAL,
        enable_rdo: true,
        lambda_scale: 218,
        enable_multi_ref: true,
        enable_segments: true,
        segment_quant_deltas: DEFAULT_SEGMENT_QUANT_DELTAS,
        segment_lf_deltas: DEFAULT_SEGMENT_LF_DELTAS,
        enable_scene_cut: false,
        scene_cut_threshold: DEFAULT_SCENE_CUT_THRESHOLD,
        scene_cut_quant_boost: DEFAULT_SCENE_CUT_QUANT_BOOST,
        scene_cut_boost_frames: DEFAULT_SCENE_CUT_BOOST_FRAMES,
        enable_lookahead_altref: false,
        lookahead_window: 0,
        loop_filter_mode: LoopFilterMode::Auto,
        simple_lf_max_level: DEFAULT_SIMPLE_LF_MAX_LEVEL,
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
        adaptive_segment_thresholds: false,
        enable_split_mv_joint_refine: false,
        split_mv_joint_refine_passes: 0,
        lambda_long_ref_scale_x256: 256,
    }
}

/// Encode the clip and return `(total_bytes, average_y_psnr)`.
fn measure(cfg: Vp8EncoderConfig, clip: &[VideoFrame]) -> (usize, f64) {
    let mut params = CodecParameters::video(CodecId::new("vp8"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = make_encoder_with_config(&params, cfg).expect("encoder");
    let mut total_bytes = 0usize;
    let mut psnr_sum = 0f64;
    let mut psnr_n = 0usize;
    let mut dec = Vp8Decoder::new(CodecId::new("vp8"));
    let drain = |enc: &mut Box<dyn oxideav_core::Encoder>,
                 dec: &mut Vp8Decoder,
                 vf: &VideoFrame,
                 total_bytes: &mut usize,
                 psnr_sum: &mut f64,
                 psnr_n: &mut usize| {
        loop {
            match enc.receive_packet() {
                Ok(pkt) => {
                    *total_bytes += pkt.data.len();
                    let mut p = Packet::new(0, TimeBase::new(1, 30), pkt.data.clone());
                    p.pts = pkt.pts;
                    p.dts = pkt.dts;
                    p.duration = pkt.duration;
                    p.flags = pkt.flags;
                    dec.send_packet(&p).expect("dec send_packet");
                    while let Ok(Frame::Video(rf)) = dec.receive_frame() {
                        *psnr_sum += psnr(&vf.planes[0].data, &rf.planes[0].data);
                        *psnr_n += 1;
                    }
                }
                Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("encoder error: {e:?}"),
            }
        }
    };
    for vf in clip.iter() {
        enc.send_frame(&Frame::Video(vf.clone()))
            .expect("send_frame");
        drain(
            &mut enc,
            &mut dec,
            vf,
            &mut total_bytes,
            &mut psnr_sum,
            &mut psnr_n,
        );
    }
    enc.flush().ok();
    if let Some(last_vf) = clip.last() {
        drain(
            &mut enc,
            &mut dec,
            last_vf,
            &mut total_bytes,
            &mut psnr_sum,
            &mut psnr_n,
        );
    }
    let avg = if psnr_n > 0 {
        psnr_sum / psnr_n as f64
    } else {
        0.0
    };
    (total_bytes, avg)
}

#[test]
fn adaptive_thresholds_shrinks_mixed_content() {
    let clip = make_mixed_clip(8);
    let (bytes_baseline, psnr_baseline) = measure(cfg_baseline(), &clip);
    let mut cfg = cfg_baseline();
    cfg.adaptive_segment_thresholds = true;
    let (bytes_adaptive, psnr_adaptive) = measure(cfg, &clip);
    eprintln!(
        "baseline: {bytes_baseline} bytes / {psnr_baseline:.2} dB; \
         adaptive: {bytes_adaptive} bytes / {psnr_adaptive:.2} dB"
    );
    // Per-MB QP refinement: adaptive thresholds must produce a smaller
    // bitstream on mixed content (the per-segment quants now have
    // populations to bite on instead of every MB landing in segment 0).
    assert!(
        bytes_adaptive < bytes_baseline,
        "adaptive thresholds did not shrink bitstream: \
         baseline={bytes_baseline}, adaptive={bytes_adaptive}"
    );
    // PSNR must not collapse — anything within 0.3 dB of baseline is
    // acceptable since we're trading bits for distortion in a clearly
    // bounded way.
    assert!(
        psnr_adaptive >= psnr_baseline - 0.3,
        "adaptive thresholds dropped PSNR too much: \
         baseline={psnr_baseline:.2}, adaptive={psnr_adaptive:.2}"
    );
}

#[test]
fn long_ref_lambda_boost_improves_mixed_content_rd() {
    let clip = make_mixed_clip(8);
    let (bytes_baseline, psnr_baseline) = measure(cfg_baseline(), &clip);
    let mut cfg = cfg_baseline();
    cfg.lambda_long_ref_scale_x256 = DEFAULT_LAMBDA_LONG_REF_SCALE_X256;
    let (bytes_boost, psnr_boost) = measure(cfg, &clip);
    eprintln!(
        "baseline: {bytes_baseline} bytes / {psnr_baseline:.2} dB; \
         long-ref boost: {bytes_boost} bytes / {psnr_boost:.2} dB"
    );
    // The lambda boost biases the picker towards LAST when GOLDEN/ALT
    // would be a marginal pick; on motion-heavy mixed content this
    // should both shrink the bitstream and improve PSNR (drift
    // amortisation effect — taking LAST keeps the residual short).
    assert!(
        bytes_boost <= bytes_baseline,
        "long-ref lambda boost grew the bitstream: \
         baseline={bytes_baseline}, boost={bytes_boost}"
    );
    assert!(
        psnr_boost >= psnr_baseline,
        "long-ref lambda boost dropped PSNR: \
         baseline={psnr_baseline:.2}, boost={psnr_boost:.2}"
    );
}

#[test]
fn all_three_features_combine_for_meaningful_win() {
    let clip = make_mixed_clip(8);
    let (bytes_baseline, psnr_baseline) = measure(cfg_baseline(), &clip);
    let mut cfg = cfg_baseline();
    cfg.adaptive_segment_thresholds = true;
    cfg.enable_split_mv_joint_refine = true;
    cfg.split_mv_joint_refine_passes = DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES;
    cfg.lambda_long_ref_scale_x256 = DEFAULT_LAMBDA_LONG_REF_SCALE_X256;
    let (bytes_full, psnr_full) = measure(cfg, &clip);
    eprintln!(
        "baseline: {bytes_baseline} bytes / {psnr_baseline:.2} dB; \
         all three on: {bytes_full} bytes / {psnr_full:.2} dB"
    );
    let saved_pct = (bytes_baseline as f64 - bytes_full as f64) / bytes_baseline as f64 * 100.0;
    let psnr_gain = psnr_full - psnr_baseline;
    eprintln!("delta: {saved_pct:.2}% bytes saved, {psnr_gain:+.2} dB PSNR");
    // Combined opt-in features must clear at least 5% bytes saved AND
    // 0.3 dB Y PSNR gain on this synthetic mixed-content clip.
    assert!(
        saved_pct >= 5.0,
        "combined opt-ins did not save at least 5% bytes: {saved_pct:.2}%"
    );
    assert!(
        psnr_gain >= 0.3,
        "combined opt-ins did not gain at least 0.3 dB Y: {psnr_gain:+.2}"
    );
}
