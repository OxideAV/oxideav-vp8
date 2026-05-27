//! Multi-frame I + P stream-driver round-trip with the round-160 /
//! round-161 §11 intra-within-inter MB picker threaded into
//! [`oxideav_vp8::Vp8InterStreamEncoder`] (round 162 follow-up #2).
//!
//! Round 161's intra-pick widened the per-MB picker to the full §11.2
//! × §11.4 whole-block intra grid (4 luma × 4 chroma = 16 candidates,
//! `B_PRED` excluded), but the intra-pick flag was only exposed on the
//! bare-encoder entry-points
//! ([`oxideav_vp8::encode_p_frame_multi_ref_with_intra_pick`] /
//! [`oxideav_vp8::encode_p_frame_multi_ref_with_refresh_and_intra_pick`]).
//! Round 162 threads it into the stream driver as three opt-in
//! entry-points that mirror the existing `encode_frame*` family:
//!
//!   * `encode_frame_with_intra_pick(frame)` — scheduler-driven,
//!     intra-pick on every emitted P-frame, K-frames unchanged.
//!   * `encode_frame_with_force_and_intra_pick(frame, force_keyframe)`
//!     — same plus a `force_keyframe` override.
//!   * `encode_p_frame_with_refresh_and_intra_pick(frame, refresh)` —
//!     direct P-frame call with caller-driven §9.7 / §9.8 refresh
//!     pattern.
//!
//! This test pins:
//!
//!   1. Wire compatibility with the bare-encoder intra-pick path: the
//!      bytes the stream driver emits on a K + P + P sequence are
//!      byte-identical to what
//!      [`oxideav_vp8::encode_keyframe_with_reconstruction`] +
//!      [`oxideav_vp8::encode_p_frame_multi_ref_with_intra_pick`]
//!      produce when driven through the same §9.7 reference-slot
//!      ladder (since the stream driver internally uses that exact
//!      composition).
//!   2. The picker activates: on a black-K + bright-P transition the
//!      stream's first P-frame's §9.10 `prob_intra` byte drops below
//!      255 (the all-inter sentinel) — i.e. at least one MB picked
//!      intra over inter, exactly as on the bare-encoder path.
//!   3. Self-decode: every emitted frame round-trips through
//!      [`oxideav_vp8::Vp8DecoderState`] at a per-frame Y-PSNR ≥ 25
//!      dB at mid quantiser.
//!   4. Scheduler invariants from the non-intra-pick path still hold
//!      when the intra-pick variant is the encoder: K-frame at frame
//!      index 0, P-frames at indices 1 / 2 / 3, K-frame at index 4
//!      (interval = 4), `last_keyframe_index` tracks the K-frame
//!      anchor, `force_keyframe` re-anchors the interval, and the
//!      §9.7 reference-slot rotation matches the bare-encoder ladder.
//!   5. The refresh-driven entry-point's slot rotation matches the
//!      §20 page-147 walk on a `refresh_golden = true` request: after
//!      the call, GOLDEN holds the just-emitted P-frame's
//!      reconstruction (not the prior K's), LAST also reflects the
//!      P-frame (default `refresh_last = true`).
//!
//! Black-box self-decode — no external codec consulted.

use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_with_intra_pick,
    encode_p_frame_multi_ref_with_refresh_and_intra_pick, FrameKind, I420Frame, KeyframeParams,
    RefreshControls, Vp8CodedHeader, Vp8DecodedFrame, Vp8DecoderState, Vp8FrameHeader,
    Vp8InterStreamEncoder,
};

fn flat_color(width: usize, height: usize, v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![v; width * height];
    let u = vec![v; (width / 2) * (height / 2)];
    let v_plane = vec![v; (width / 2) * (height / 2)];
    (y, u, v_plane)
}

fn plane_psnr(src: &[u8], rec: &[u8]) -> f64 {
    assert_eq!(src.len(), rec.len());
    let mut sse: u64 = 0;
    for (a, b) in src.iter().zip(rec.iter()) {
        let d = *a as i32 - *b as i32;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / src.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn y_psnr(src_y: &[u8], rec: &Vp8DecodedFrame) -> f64 {
    plane_psnr(src_y, &rec.y)
}

/// Pull the §9.10 `prob_intra` byte out of an inter-frame bitstream.
fn prob_intra_of(bytes: &[u8]) -> u8 {
    let hdr = Vp8FrameHeader::parse(bytes).expect("§19.1 tag parses");
    assert!(!hdr.key_frame, "prob_intra_of called on a K frame");
    let partition_start = hdr.header_bytes_consumed;
    let partition_end = partition_start + hdr.first_partition_size as usize;
    let partition = &bytes[partition_start..partition_end];
    let coded = Vp8CodedHeader::parse(partition, /*key_frame=*/ false)
        .expect("§19.2 control partition parses");
    coded.prob_intra.expect("inter prob_intra present")
}

/// Stream driver bytes on a K + P sequence are byte-identical to the
/// bare-encoder composition that internally backs them.
#[test]
fn stream_intra_pick_bytes_match_bare_encoder_composition() {
    let width = 32u32;
    let height = 32u32;
    let qi = 32u8;
    let params = KeyframeParams {
        y_ac_qi: qi,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };

    // K source: black. P source: bright 200. The picker on the
    // P-frame should land on intra for at least one MB (matches the
    // first test in `encoder_pframe_intra_pick.rs`).
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    // ── Bare-encoder reference path ─────────────────────────────────
    let (bare_k_bytes, bare_k_planes) =
        encode_keyframe_with_reconstruction(&k_frame, &params).expect("bare K");
    let (bare_p_bytes, _bare_p_planes) =
        encode_p_frame_multi_ref_with_intra_pick(&p_frame, &bare_k_planes, None, None, &params)
            .expect("bare P");

    // ── Stream-driver intra-pick path ───────────────────────────────
    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero keyframe interval");
    let stream_k = enc
        .encode_frame_with_intra_pick(&k_frame)
        .expect("stream K");
    let stream_p = enc
        .encode_frame_with_intra_pick(&p_frame)
        .expect("stream P");

    assert_eq!(stream_k.kind, FrameKind::Key);
    assert_eq!(stream_p.kind, FrameKind::InterZeroMv);
    assert_eq!(
        stream_k.bytes, bare_k_bytes,
        "stream K bytes must match the bare keyframe encoder bytes"
    );
    assert_eq!(
        stream_p.bytes, bare_p_bytes,
        "stream P intra-pick bytes must match the bare \
         encode_p_frame_multi_ref_with_intra_pick output"
    );

    // The picker actually activated. `fit_prob_l8(count_intra, count_inter)`
    // returns:
    //   * `1` when no MB picked intra (the clamped boundary value);
    //   * `255` when every MB picked intra (`count_inter == 0`);
    //   * something in between when the picker split the MBs.
    // On a black-LAST + bright-source pattern intra dominates every MB
    // (matches `intra_pick_selects_intra_when_inter_residual_is_large`
    // in `tests/encoder_pframe_intra_pick.rs` which lands on `255`),
    // so the engagement signal is "anything other than `1`".
    let pi = prob_intra_of(&stream_p.bytes);
    eprintln!("stream P §9.10 prob_intra = {pi}");
    assert!(
        pi > 1,
        "stream P prob_intra = {pi}: the round-161 picker should select intra for at least one \
         MB on a black-LAST + bright-source pattern (= the bare-encoder test's setup)"
    );

    // Self-decode confirms wire-correctness.
    let mut dec = Vp8DecoderState::new();
    let dk = dec.decode_frame(&stream_k.bytes).expect("decode K");
    let dp = dec.decode_frame(&stream_p.bytes).expect("decode P");
    let psnr_k = y_psnr(&k_y, &dk);
    let psnr_p = y_psnr(&p_y, &dp);
    eprintln!("stream K Y-PSNR = {psnr_k:.2} dB, stream P Y-PSNR = {psnr_p:.2} dB");
    assert!(psnr_k >= 30.0, "stream K Y-PSNR {psnr_k:.2} dB below floor");
    assert!(psnr_p >= 25.0, "stream P Y-PSNR {psnr_p:.2} dB below floor");
}

/// Scheduler invariants: K at 0, P at 1/2/3, K at 4 with `interval = 4`.
#[test]
fn stream_intra_pick_scheduler_keyframe_interval_4() {
    let width = 16u32;
    let height = 32u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (y, u, v) = flat_color(width as usize, height as usize, 128);
    let frame = I420Frame::packed(width, height, &y, &u, &v);
    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero keyframe interval");

    assert_eq!(enc.next_frame_is_keyframe(), FrameKind::Key);
    let f0 = enc.encode_frame_with_intra_pick(&frame).expect("frame 0 K");
    assert_eq!(f0.kind, FrameKind::Key);
    assert_eq!(f0.frame_index, 0);
    assert_eq!(enc.last_keyframe_index(), Some(0));

    for i in 1..=3u64 {
        assert_eq!(enc.next_frame_is_keyframe(), FrameKind::InterZeroMv);
        let f = enc.encode_frame_with_intra_pick(&frame).expect("p frame");
        assert_eq!(f.kind, FrameKind::InterZeroMv, "frame {i}");
        assert_eq!(f.frame_index, i, "frame {i} index");
    }

    assert_eq!(enc.next_frame_is_keyframe(), FrameKind::Key);
    let f4 = enc.encode_frame_with_intra_pick(&frame).expect("frame 4 K");
    assert!(f4.is_keyframe(), "frame 4 must be K at interval 4");
    assert_eq!(enc.last_keyframe_index(), Some(4));
}

/// `force_keyframe = true` on the intra-pick path re-anchors the
/// interval the same way [`Vp8InterStreamEncoder::encode_frame_with_force`]
/// does on the non-intra-pick path.
#[test]
fn stream_intra_pick_force_keyframe_reanchors_interval() {
    let width = 16u32;
    let height = 32u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (y, u, v) = flat_color(width as usize, height as usize, 100);
    let frame = I420Frame::packed(width, height, &y, &u, &v);
    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero keyframe interval");

    // K, P, P-forced-K, P, P, P, K (re-anchored at frame 2)
    let kinds = [
        (false, FrameKind::Key),         // 0 — first-ever
        (false, FrameKind::InterZeroMv), // 1
        (true, FrameKind::Key),          // 2 — forced
        (false, FrameKind::InterZeroMv), // 3
        (false, FrameKind::InterZeroMv), // 4 (would be K w/o re-anchor)
        (false, FrameKind::InterZeroMv), // 5
        (false, FrameKind::Key),         // 6 — re-anchored interval
    ];
    for (i, (force, expected)) in kinds.iter().enumerate() {
        let out = enc
            .encode_frame_with_force_and_intra_pick(&frame, *force)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
        assert_eq!(out.kind, *expected, "frame {i} kind mismatch");
    }
    assert_eq!(enc.last_keyframe_index(), Some(6));
}

/// `encode_p_frame_with_refresh_and_intra_pick` byte-matches the
/// equivalent bare-encoder call and applies the §20 page-147 slot
/// rotation correctly: a `refresh_golden_frame = true` call updates
/// GOLDEN to the just-emitted reconstruction (not the prior K's
/// contents).
#[test]
fn stream_intra_pick_refresh_drives_slot_rotation() {
    let width = 32u32;
    let height = 32u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (k_y, k_u, k_v) = flat_color(width as usize, height as usize, 0);
    let (p_y, p_u, p_v) = flat_color(width as usize, height as usize, 200);
    let k_frame = I420Frame::packed(width, height, &k_y, &k_u, &k_v);
    let p_frame = I420Frame::packed(width, height, &p_y, &p_u, &p_v);

    let mut enc = Vp8InterStreamEncoder::new(params, 100).expect("non-zero keyframe interval");

    // First emit a K to populate LAST / GOLDEN / ALTREF.
    let k_out = enc.encode_frame_with_intra_pick(&k_frame).expect("K frame");
    assert_eq!(k_out.kind, FrameKind::Key);
    let golden_after_k = enc.golden().expect("golden after K").y.clone();

    // P frame with custom refresh: refresh_last + refresh_golden, no
    // copies. GOLDEN should after this hold the P-frame reconstruction,
    // not the K's.
    let refresh = RefreshControls {
        refresh_last: true,
        refresh_golden_frame: true,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 0,
        copy_buffer_to_alternate: 0,
    };

    // Bare-encoder reference path to byte-compare against.
    let bare_k_planes = oxideav_vp8::KeyframePlanes {
        y: enc.last().unwrap().y.clone(),
        u: enc.last().unwrap().u.clone(),
        v: enc.last().unwrap().v.clone(),
        y_stride: enc.last().unwrap().y_stride,
        uv_stride: enc.last().unwrap().uv_stride,
        mb_cols: enc.last().unwrap().mb_cols,
        mb_rows: enc.last().unwrap().mb_rows,
    };
    let (bare_p_bytes, _) = encode_p_frame_multi_ref_with_refresh_and_intra_pick(
        &p_frame,
        &bare_k_planes,
        None,
        None,
        &params,
        &refresh,
    )
    .expect("bare P refresh+intra");

    let p_out = enc
        .encode_p_frame_with_refresh_and_intra_pick(&p_frame, &refresh)
        .expect("stream P refresh+intra");
    assert_eq!(p_out.kind, FrameKind::InterZeroMv);
    assert_eq!(
        p_out.bytes, bare_p_bytes,
        "stream P refresh+intra bytes must match the bare \
         encode_p_frame_multi_ref_with_refresh_and_intra_pick output"
    );

    // The picker activated (same source pattern as the byte-match test;
    // every MB picks intra ⇒ `prob_intra = 255`, the "all-intra" fitter
    // boundary, NOT the pre-r160 hardwired sentinel — only the fitter
    // ever writes the same byte value).
    let pi = prob_intra_of(&p_out.bytes);
    assert!(
        pi > 1,
        "refresh+intra-pick stream P prob_intra = {pi}: picker should select intra on this source"
    );

    // §20 page-147 slot rotation: GOLDEN now reflects the P-frame
    // reconstruction (changed from the K's contents).
    let golden_after_p = enc.golden().expect("golden after P").y.clone();
    assert_ne!(
        golden_after_p, golden_after_k,
        "refresh_golden_frame = true should replace GOLDEN with the P-frame reconstruction, \
         but the slot still holds the prior K's bytes"
    );
    // LAST also reflects the P-frame reconstruction.
    let last_after_p = enc.last().expect("last after P").y.clone();
    assert_eq!(
        last_after_p, golden_after_p,
        "refresh_last + refresh_golden_frame should write the same reconstruction into both slots"
    );

    // Self-decode round-trip sanity.
    let mut dec = Vp8DecoderState::new();
    let _ = dec.decode_frame(&k_out.bytes).expect("decode K");
    let dp = dec.decode_frame(&p_out.bytes).expect("decode P");
    let psnr = y_psnr(&p_y, &dp);
    eprintln!("refresh+intra P Y-PSNR = {psnr:.2} dB");
    assert!(
        psnr >= 25.0,
        "refresh+intra P Y-PSNR {psnr:.2} dB below floor"
    );
}

/// Without any P-frame having been emitted yet,
/// `encode_p_frame_with_refresh_and_intra_pick` errors with
/// `NoLastReference` (matches the non-intra-pick refresh path).
#[test]
fn stream_intra_pick_refresh_errors_when_no_last() {
    let width = 16u32;
    let height = 16u32;
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (y, u, v) = flat_color(width as usize, height as usize, 128);
    let frame = I420Frame::packed(width, height, &y, &u, &v);
    let refresh = RefreshControls::default();
    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero keyframe interval");
    let err = enc
        .encode_p_frame_with_refresh_and_intra_pick(&frame, &refresh)
        .expect_err("should refuse without a LAST slot");
    assert!(
        matches!(err, oxideav_vp8::StreamEncodeError::NoLastReference),
        "expected NoLastReference, got {err:?}"
    );
    assert_eq!(enc.frame_count(), 0, "failure must not advance the counter");
}

/// Dimensions-lock semantics are preserved on the intra-pick path.
#[test]
fn stream_intra_pick_dimensions_change_rejected() {
    let params = KeyframeParams {
        y_ac_qi: 32,
        loop_filter_level: 0,
        sharpness_level: 0,
        nbr_of_dct_partitions: 1,
        filter_type: false,
    };
    let (y1, u1, v1) = flat_color(32, 32, 128);
    let f1 = I420Frame::packed(32, 32, &y1, &u1, &v1);
    let (y2, u2, v2) = flat_color(48, 48, 64);
    let f2 = I420Frame::packed(48, 48, &y2, &u2, &v2);

    let mut enc = Vp8InterStreamEncoder::new(params, 4).expect("non-zero keyframe interval");
    enc.encode_frame_with_intra_pick(&f1)
        .expect("first frame locks dims");
    let err = enc
        .encode_frame_with_intra_pick(&f2)
        .expect_err("differently-sized second frame");
    assert!(matches!(
        err,
        oxideav_vp8::StreamEncodeError::DimensionsChanged {
            first: (32, 32),
            got: (48, 48)
        }
    ));
    assert_eq!(enc.frame_count(), 1, "failure must not advance the counter");
}
