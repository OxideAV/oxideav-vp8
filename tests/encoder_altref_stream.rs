//! End-to-end coverage of [`oxideav_vp8::Vp8AltrefStreamEncoder`] — the
//! lagged auto-altref stream driver (round 384): lookahead groups, an
//! invisible ARNR anchor per group, visible multi-reference P-frames,
//! full stateful self-decode.
//!
//! Source: 12 frames of a slowly-translating textured scene with
//! independent per-frame noise — the content class temporal filtering
//! exists for (noise is uncorrelated across frames; the scene motion is
//! trackable by the whole-pel aligner).
//!
//! What this pins:
//!
//!   * packet structure: with `altref_window = 4` and keyframes only at
//!     frame 0, 12 source frames emit 3 groups → 15 packets
//!     (1 K + 3 invisible anchors + 11 P), visible packets mapping 1:1
//!     onto source frames in order;
//!   * every packet decodes through [`Vp8DecoderState`] in emission
//!     order; `last_frame_shown()` mirrors each packet's visibility;
//!   * visible decoded pictures hit a PSNR floor against their clean
//!     (pre-noise) source — the stream is a faithful encode, and
//!     altref prediction from the denoised anchor keeps quality up on
//!     noisy content;
//!   * the anchored stream's visible P-frame bytes undercut a
//!     no-anchor baseline (same params, same §16.2 multi-ref P-frame
//!     path, no invisible updates): predicting noisy frames from a
//!     noise-reduced anchor beats predicting them from the previous
//!     noisy reconstruction;
//!   * `finish()` drains the tail group and is idempotent.
//!
//! Black-box: encoder output feeds the crate's own decoder only.

use oxideav_vp8::{
    encode_p_frame_multi_ref, AltrefPacketKind, AltrefStreamConfig, ArnrConfig, I420Frame,
    KeyframeParams, Vp8AltrefStreamEncoder, Vp8DecoderState,
};

const W: usize = 64;
const H: usize = 64;
const FRAMES: usize = 12;
const NOISE_AMP: i32 = 5;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 33
    }
    fn noise(&mut self, amp: i32) -> i32 {
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

/// Clean scene at time `t`: a textured field translating 1 px right per
/// frame (edge-clamped), chroma slowly ramping.
fn clean_frame(t: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let sc = (c as i32 - t as i32).clamp(0, W as i32 - 1) as usize;
            y[r * W + c] = (48 + ((r * 5 + sc * 3) & 0x9f)) as u8;
        }
    }
    let (cw, ch) = (W / 2, H / 2);
    let u = vec![(118 + t) as u8; cw * ch];
    let v = vec![(134 - t) as u8; cw * ch];
    (y, u, v)
}

/// The noisy source actually fed to the encoder.
fn noisy_frame(t: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (y, u, v) = clean_frame(t);
    let mut rng = Lcg(0x5eed_0000 + t as u64);
    let ny = y
        .iter()
        .map(|&p| (p as i32 + rng.noise(NOISE_AMP)).clamp(0, 255) as u8)
        .collect();
    (ny, u, v)
}

fn plane_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sse: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i64 - y as i64;
            (d * d) as u64
        })
        .sum();
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn lagged_altref_stream_structure_decode_and_payoff() {
    let sources: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..FRAMES).map(noisy_frame).collect();
    let params = KeyframeParams {
        y_ac_qi: 44,
        ..KeyframeParams::default()
    };
    let config = AltrefStreamConfig {
        params,
        keyframe_interval: 0, // key only frame 0
        altref_window: 4,
        arnr: ArnrConfig::default(),
    };

    // ---- Drive the lagged encoder -----------------------------------
    let mut enc = Vp8AltrefStreamEncoder::new(config).expect("window > 0");
    let mut packets = Vec::new();
    for (t, (y, u, v)) in sources.iter().enumerate() {
        let frame = I420Frame::packed(W as u32, H as u32, y, u, v);
        let emitted = enc.push_frame(&frame).expect("push_frame");
        // Lag property: nothing comes out until a group completes.
        if (t + 1) % 4 != 0 {
            assert!(emitted.is_empty(), "mid-group push must emit nothing");
        } else {
            assert!(!emitted.is_empty(), "group boundary must emit");
        }
        packets.extend(emitted);
    }
    packets.extend(enc.finish().expect("finish"));
    assert!(enc.finish().expect("finish idempotent").is_empty());
    assert_eq!(enc.input_count(), FRAMES as u64);
    assert_eq!(enc.buffered(), 0);

    // ---- Packet structure: 1 K + 3 anchors + 11 P -------------------
    assert_eq!(packets.len(), FRAMES + 3, "one invisible anchor per group");
    let kinds: Vec<AltrefPacketKind> = packets.iter().map(|p| p.kind).collect();
    assert_eq!(kinds[0], AltrefPacketKind::Key);
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == AltrefPacketKind::AltrefUpdate)
            .count(),
        3
    );
    // Visible packets cover source indices 0..12 in order.
    let visible_indices: Vec<u64> = packets.iter().filter_map(|p| p.source_index).collect();
    assert_eq!(visible_indices, (0..FRAMES as u64).collect::<Vec<_>>());
    for p in &packets {
        assert_eq!(
            p.is_visible(),
            p.kind != AltrefPacketKind::AltrefUpdate,
            "only anchors are invisible"
        );
    }

    // ---- Stateful decode in emission order ---------------------------
    let mut dec = Vp8DecoderState::new();
    let mut mean_psnr = 0.0f64;
    let mut visible_seen = 0usize;
    for p in &packets {
        let picture = dec.decode_frame(&p.bytes).expect("packet decodes");
        assert_eq!(
            dec.last_frame_shown(),
            Some(p.is_visible()),
            "wire visibility must match the packet classification"
        );
        if let Some(src_idx) = p.source_index {
            let (cy, _, _) = clean_frame(src_idx as usize);
            let psnr = plane_psnr(&picture.y, &cy);
            assert!(
                psnr >= 26.0,
                "visible frame {src_idx} PSNR-Y vs clean scene too low: {psnr:.2} dB"
            );
            mean_psnr += psnr;
            visible_seen += 1;
        }
    }
    assert_eq!(visible_seen, FRAMES);
    mean_psnr /= FRAMES as f64;

    // ---- Payoff: anchored P-frames beat a no-anchor baseline --------
    // Baseline: same params, same multi-ref P-frame encoder, but the
    // reference ladder never receives an altref anchor (ALTREF stays at
    // the keyframe, as a plain streaming encoder would leave it).
    let anchored_p_bytes: usize = packets
        .iter()
        .filter(|p| p.kind == AltrefPacketKind::Inter)
        .map(|p| p.bytes.len())
        .sum();

    let mut baseline_p_bytes = 0usize;
    {
        let (y0, u0, v0) = &sources[0];
        let kf = I420Frame::packed(W as u32, H as u32, y0, u0, v0);
        let (_, kf_recon) =
            oxideav_vp8::encode_keyframe_with_reconstruction(&kf, &params).expect("baseline K");
        let mut last = kf_recon.clone();
        for (y, u, v) in sources.iter().skip(1) {
            let frame = I420Frame::packed(W as u32, H as u32, y, u, v);
            let (bytes, recon) =
                encode_p_frame_multi_ref(&frame, &last, Some(&kf_recon), Some(&kf_recon), &params)
                    .expect("baseline P");
            baseline_p_bytes += bytes.len();
            last = recon;
        }
    }
    assert!(
        anchored_p_bytes < baseline_p_bytes,
        "anchored P-frames must undercut the no-anchor baseline: {anchored_p_bytes} vs {baseline_p_bytes} bytes \
         (mean visible PSNR-Y {mean_psnr:.2} dB)"
    );
}

#[test]
fn keyframe_cadence_closes_groups_early() {
    // keyframe_interval = 5 with window 4: frame 5 must open with a key
    // frame even though it lands mid-group-rhythm, and frame 10 again.
    let params = KeyframeParams::default();
    let config = AltrefStreamConfig {
        params,
        keyframe_interval: 5,
        altref_window: 4,
        arnr: ArnrConfig::default(),
    };
    let mut enc = Vp8AltrefStreamEncoder::new(config).expect("window > 0");
    let mut packets = Vec::new();
    for t in 0..11usize {
        let (y, u, v) = noisy_frame(t);
        let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
        packets.extend(enc.push_frame(&frame).expect("push"));
    }
    packets.extend(enc.finish().expect("finish"));

    let key_indices: Vec<u64> = packets
        .iter()
        .filter(|p| p.kind == AltrefPacketKind::Key)
        .map(|p| p.source_index.expect("keys are visible"))
        .collect();
    assert_eq!(
        key_indices,
        vec![0, 5, 10],
        "keys at the configured cadence"
    );

    // The whole stream still decodes in emission order.
    let mut dec = Vp8DecoderState::new();
    let visible: usize = packets
        .iter()
        .map(|p| {
            dec.decode_frame(&p.bytes).expect("decodes");
            usize::from(p.is_visible())
        })
        .sum();
    assert_eq!(visible, 11);
}

#[test]
fn zero_window_is_rejected_and_dimension_lock_holds() {
    let config = AltrefStreamConfig {
        altref_window: 0,
        ..AltrefStreamConfig::default()
    };
    assert!(Vp8AltrefStreamEncoder::new(config).is_none());

    let mut enc = Vp8AltrefStreamEncoder::new(AltrefStreamConfig::default()).expect("default ok");
    let (y, u, v) = noisy_frame(0);
    let frame = I420Frame::packed(W as u32, H as u32, &y, &u, &v);
    enc.push_frame(&frame).expect("first frame");
    let small_y = vec![0u8; 32 * 32];
    let small_c = vec![128u8; 16 * 16];
    let small = I420Frame::packed(32, 32, &small_y, &small_c, &small_c);
    let err = enc.push_frame(&small).expect_err("resize rejected");
    assert!(matches!(
        err,
        oxideav_vp8::StreamEncodeError::DimensionsChanged { .. }
    ));
}
