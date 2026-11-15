#![no_main]

//! Fuzz: inter-frame encode options — search (the original exhaustive
//! one or an effort level) × thread budget × intra picker / loop-filter
//! search / token-probability fit / segmentation — in pixel lockstep
//! with the decoder at a second thread budget (RFC 6386 §9 / §13 / §15 /
//! §16 / §17 / §18).
//!
//! Each iteration encodes a key frame and two inter frames (the second
//! predicting from LAST and from the key frame as GOLDEN) and checks:
//!
//! 1. **Encoder thread-budget invariance** — bytes and reconstruction of
//!    a threaded encode equal the serial encode's.
//! 2. **Decoder lockstep** — a serial `Vp8DecoderState` reproduces the
//!    encoder's reconstruction of every frame over the visible window.
//! 3. **Decoder thread-budget invariance** — a threaded decoder returns
//!    the same pictures.
//! 4. No encode or decode error for in-range parameters.
//!
//! Input layout:
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]` | visible width `1 + (b % 80)` luma px |
//! | `[1]` | visible height `1 + (b % 80)` luma px |
//! | `[2]` | `y_ac_qi = b % 128` |
//! | `[3]` | `loop_filter_level = b % 64` |
//! | `[4]` | bits 0..1: `nbr_of_dct_partitions = 1 << n`; bits 2..4: encoder thread budget `2 + n`; bits 5..7: decoder thread budget `2 + n` |
//! | `[5]` | bits 0..2: search (`0` = original, `1..=4` = effort level 0..=3, else level `n % 4`); bit 3: `filter_type`; bit 4: intra picker; bit 5: `auto_loop_filter`; bit 6: fitted token probabilities; bit 7: segmentation |
//! | `[6]` | motion of the second and third pictures: low nibble horizontal, high nibble vertical shift (each `n − 8` px) |
//! | `[7..]` | I420 pixel payload of the first picture, tiled across the three planes; later pictures are shifted copies with payload-derived noise |
//!
//! Max area 80 × 80 luma px (25 macroblocks); hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::frame::KeyframePlanes;
use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_adaptive_quant,
    encode_p_frame_multi_ref_with_refresh_and_coding_options, EncodeEffort, I420Frame,
    InterCodingOptions, InterSegmentationConfig, KeyframeParams, RefreshControls, Vp8DecoderState,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 7;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }
    let width = 1u32 + u32::from(data[0] % 80);
    let height = 1u32 + u32::from(data[1] % 80);
    if !accept_dimensions(width, height) {
        return;
    }
    let params = KeyframeParams::default()
        .with_y_ac_qi(data[2] % 128)
        .with_loop_filter_level(data[3] % 64)
        .with_nbr_of_dct_partitions(1u8 << (data[4] & 0x03))
        .with_filter_type((data[5] & 0x08) != 0);
    let encoder_threads = 2 + usize::from((data[4] >> 2) & 0x07);
    let decoder_threads = 2 + usize::from(data[4] >> 5);
    let mut options = InterCodingOptions::default()
        .with_intra_pick((data[5] & 0x10) != 0)
        .with_auto_loop_filter((data[5] & 0x20) != 0)
        .with_fitted_token_prob_updates((data[5] & 0x40) != 0);
    match data[5] & 0x07 {
        0 => {}
        n => options = options.with_effort(EncodeEffort::from_level((n - 1) % 4)),
    }
    let segmentation = ((data[5] & 0x80) != 0).then(InterSegmentationConfig::default);
    let shift_x = i32::from(data[6] & 0x0F) - 8;
    let shift_y = i32::from(data[6] >> 4) - 8;

    let w = width as usize;
    let h = height as usize;
    let (uvw, uvh) = (w.div_ceil(2), h.div_ceil(2));
    let (y_len, uv_len) = (w * h, uvw * uvh);
    let payload = &data[HEADER_BYTES..];
    let sample = |i: usize| -> u8 {
        if payload.is_empty() {
            0
        } else {
            payload[i % payload.len()]
        }
    };
    // Picture `t`: the first picture shifted by `t` times the motion,
    // edge-clamped, plus a little payload-derived noise.
    let picture = |t: i32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let plane = |pw: usize, ph: usize, base: usize, sx: i32, sy: i32| -> Vec<u8> {
            let mut out = vec![0u8; pw * ph];
            for row in 0..ph {
                for col in 0..pw {
                    let x = (col as i32 + sx * t).clamp(0, pw as i32 - 1) as usize;
                    let y = (row as i32 + sy * t).clamp(0, ph as i32 - 1) as usize;
                    let noise = if t == 0 {
                        0
                    } else {
                        sample(base + row * pw + col + 977 * t as usize) & 3
                    };
                    out[row * pw + col] = sample(base + y * pw + x).wrapping_add(noise);
                }
            }
            out
        };
        (
            plane(w, h, 0, shift_x, shift_y),
            plane(uvw, uvh, y_len, shift_x / 2, shift_y / 2),
            plane(uvw, uvh, y_len + uv_len, shift_x / 2, shift_y / 2),
        )
    };

    let encode = |options: &InterCodingOptions| -> (Vec<Vec<u8>>, Vec<KeyframePlanes>) {
        let (y, u, v) = picture(0);
        let (key_bytes, key) =
            encode_keyframe_with_reconstruction(&I420Frame::packed(width, height, &y, &u, &v), &params)
                .unwrap_or_else(|e| panic!("key frame rejected: {e:?} ({width}x{height}, {params:?})"));
        let mut packets = vec![key_bytes];
        let mut recons = vec![key.clone()];
        let mut last = key.clone();
        for t in 1..=2 {
            let (y, u, v) = picture(t);
            let frame = I420Frame::packed(width, height, &y, &u, &v);
            let refresh = RefreshControls::default();
            let golden = (t == 2).then_some(&key);
            let result = match &segmentation {
                Some(seg) => encode_p_frame_multi_ref_adaptive_quant(
                    &frame, &last, golden, None, &params, &refresh, seg, options,
                ),
                None => encode_p_frame_multi_ref_with_refresh_and_coding_options(
                    &frame, &last, golden, None, &params, &refresh, options,
                ),
            };
            let (bytes, recon) = result.unwrap_or_else(|e| {
                panic!("inter frame rejected: {e:?} ({width}x{height}, {params:?}, {options:?})")
            });
            packets.push(bytes);
            recons.push(recon.clone());
            last = recon;
        }
        (packets, recons)
    };

    let (packets, recons) = encode(&options);

    // 1. Encoder thread-budget invariance.
    let (threaded_packets, threaded_recons) = encode(&options.with_threads(encoder_threads));
    assert_eq!(
        threaded_packets, packets,
        "bytes changed with the thread budget ({encoder_threads} threads, {width}x{height}, \
         {params:?}, {options:?})"
    );
    for (a, b) in threaded_recons.iter().zip(&recons) {
        assert!(
            a.y == b.y && a.u == b.u && a.v == b.v,
            "reconstruction changed with the thread budget ({encoder_threads} threads)"
        );
    }

    // 2 + 3. Decoder lockstep, serial and threaded.
    for threads in [1usize, decoder_threads] {
        let mut decoder = Vp8DecoderState::new().with_threads(threads);
        for (n, (packet, recon)) in packets.iter().zip(&recons).enumerate() {
            let decoded = decoder.decode_frame(packet).unwrap_or_else(|e| {
                panic!(
                    "own decoder ({threads} threads) rejected frame {n}: {e:?} \
                     ({width}x{height}, {params:?}, {options:?})"
                )
            });
            assert_eq!((decoded.width, decoded.height), (width, height));
            for row in 0..h {
                assert_eq!(
                    decoded.y[row * w..(row + 1) * w],
                    recon.y[row * recon.y_stride..row * recon.y_stride + w],
                    "frame {n} luma row {row} differs ({threads} decoder threads, \
                     {width}x{height}, {params:?}, {options:?})"
                );
            }
            for row in 0..uvh {
                let at = row * recon.uv_stride;
                assert_eq!(
                    decoded.u[row * uvw..(row + 1) * uvw],
                    recon.u[at..at + uvw],
                    "frame {n} U row {row} differs ({threads} decoder threads)"
                );
                assert_eq!(
                    decoded.v[row * uvw..(row + 1) * uvw],
                    recon.v[at..at + uvw],
                    "frame {n} V row {row} differs ({threads} decoder threads)"
                );
            }
        }
    }
});
