//! The decoder's thread budget never changes what it decodes.
//!
//! * every vendored test vector decodes to the same pictures at every
//!   budget, and those pictures are the vector's expected output;
//! * so do streams from this crate's encoder that exercise the paths the
//!   row units split up: several token partitions, intra macroblocks on
//!   inter frames, `SPLITMV`, golden / alt-ref prediction, invisible
//!   frames, both loop filters, segmentation;
//! * a damaged stream is accepted or refused alike at every budget, with
//!   the same pictures up to the refusal.

#[path = "support/photo.rs"]
mod photo;

#[path = "support/clip.rs"]
mod clip;

use oxideav_vp8::{
    AltrefStreamConfig, EncodeEffort, I420Frame, InterSegmentationConfig, KeyframeParams,
    Vp8AltrefStreamEncoder, Vp8DecoderState,
};

const BUDGETS: [usize; 4] = [2, 3, 4, 8];

fn ivf_frames(ivf: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&ivf[0..4], b"DKIF");
    let mut out = Vec::new();
    let mut cur = 32usize;
    while cur + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[cur..cur + 4].try_into().unwrap()) as usize;
        out.push(&ivf[cur + 12..cur + 12 + size]);
        cur += 12 + size;
    }
    out
}

/// Decode a packet sequence; each entry is the picture (planes back to
/// back, with the show flag) or `None` from the first refused packet on.
fn decode_all(packets: &[&[u8]], threads: usize) -> Vec<Option<(Vec<u8>, bool)>> {
    let mut dec = Vp8DecoderState::new().with_threads(threads);
    assert_eq!(dec.threads(), threads);
    packets
        .iter()
        .map(|p| {
            dec.decode_frame(p).ok().map(|d| {
                let mut out = d.y;
                out.extend_from_slice(&d.u);
                out.extend_from_slice(&d.v);
                (out, dec.last_frame_shown().unwrap_or(true))
            })
        })
        .collect()
}

#[test]
fn test_vectors_decode_identically_at_every_budget() {
    let vectors: [(&str, &[u8], &[u8]); 8] = [
        (
            "altref-arnr-on",
            include_bytes!("fixtures/altref-arnr-on/input.ivf"),
            include_bytes!("fixtures/altref-arnr-on/expected.yuv"),
        ),
        (
            "golden-update-cycle",
            include_bytes!("fixtures/golden-update-cycle/input.ivf"),
            include_bytes!("fixtures/golden-update-cycle/expected.yuv"),
        ),
        (
            "gradient-and-noise-128x128",
            include_bytes!("fixtures/gradient-and-noise-128x128/input.ivf"),
            include_bytes!("fixtures/gradient-and-noise-128x128/expected.yuv"),
        ),
        (
            "i-frame-then-p-frame-64x64",
            include_bytes!("fixtures/i-frame-then-p-frame-64x64/input.ivf"),
            include_bytes!("fixtures/i-frame-then-p-frame-64x64/expected.yuv"),
        ),
        (
            "i-only-64x64",
            include_bytes!("fixtures/i-only-64x64/input.ivf"),
            include_bytes!("fixtures/i-only-64x64/expected.yuv"),
        ),
        (
            "i-only-loopfilter-high",
            include_bytes!("fixtures/i-only-loopfilter-high/input.ivf"),
            include_bytes!("fixtures/i-only-loopfilter-high/expected.yuv"),
        ),
        (
            "segment-4-partitions",
            include_bytes!("fixtures/segment-4-partitions/input.ivf"),
            include_bytes!("fixtures/segment-4-partitions/expected.yuv"),
        ),
        (
            "vp8-with-loopfilter-mode-simple",
            include_bytes!("fixtures/vp8-with-loopfilter-mode-simple/input.ivf"),
            include_bytes!("fixtures/vp8-with-loopfilter-mode-simple/expected.yuv"),
        ),
    ];
    for (name, ivf, expected) in vectors {
        let packets = ivf_frames(ivf);
        let serial = decode_all(&packets, 1);
        // The shown pictures, back to back, are the vector's output.
        let shown: Vec<u8> = serial
            .iter()
            .map(|d| d.as_ref().expect("vector decodes"))
            .filter(|(_, shown)| *shown)
            .flat_map(|(picture, _)| picture.iter().copied())
            .collect();
        assert_eq!(shown.len(), expected.len(), "{name}: output size");
        assert!(shown == expected, "{name}: serial output");
        for threads in BUDGETS {
            assert!(
                decode_all(&packets, threads) == serial,
                "{name}: threads {threads}"
            );
        }
    }
}

/// The knobs of one lagged-stream encode of the moving clip.
#[derive(Clone, Copy)]
struct Case {
    w: usize,
    h: usize,
    frames: usize,
    partitions: u8,
    lf: u8,
    simple: bool,
    effort: EncodeEffort,
    segmentation: bool,
}

fn encode(case: Case) -> Vec<Vec<u8>> {
    let Case { w, h, .. } = case;
    let source = clip::moving_clip(w, h, case.frames);
    let params = KeyframeParams::default()
        .with_y_ac_qi(30)
        .with_nbr_of_dct_partitions(case.partitions)
        .with_loop_filter_level(case.lf)
        .with_filter_type(case.simple);
    let config = AltrefStreamConfig::default()
        .with_params(params)
        .with_keyframe_interval(0)
        .with_altref_window(3)
        .with_intra_pick(true)
        .with_fitted_token_prob_updates(true)
        .with_segmentation(case.segmentation.then(InterSegmentationConfig::default))
        .with_effort(EncodeEffort::Fast)
        .with_inter_effort(case.effort)
        .with_threads(4);
    let mut enc = Vp8AltrefStreamEncoder::new(config).unwrap();
    let mut packets = Vec::new();
    for (y, u, v) in &source {
        let frame = I420Frame::packed(w as u32, h as u32, y, u, v);
        packets.extend(enc.push_frame(&frame).unwrap());
    }
    packets.extend(enc.finish().unwrap());
    packets.into_iter().map(|p| p.bytes).collect()
}

#[test]
fn encoded_streams_decode_identically_at_every_budget() {
    let case = |w, h, partitions, lf, simple, effort, segmentation| Case {
        w,
        h,
        frames: if effort == EncodeEffort::Thorough {
            4
        } else {
            7
        },
        partitions,
        lf,
        simple,
        effort,
        segmentation,
    };
    let cases = [
        case(64, 48, 1, 0, false, EncodeEffort::Balanced, false),
        case(176, 144, 1, 20, false, EncodeEffort::Balanced, false),
        case(176, 144, 2, 14, true, EncodeEffort::Fast, false),
        case(178, 98, 4, 24, false, EncodeEffort::Balanced, true),
        case(322, 182, 8, 10, false, EncodeEffort::Fastest, false),
        case(96, 64, 4, 30, false, EncodeEffort::Thorough, false),
        case(16, 208, 8, 12, false, EncodeEffort::Balanced, false),
        case(240, 16, 2, 12, false, EncodeEffort::Balanced, false),
    ];
    for case in cases {
        let stream = encode(case);
        let packets: Vec<&[u8]> = stream.iter().map(Vec::as_slice).collect();
        let serial = decode_all(&packets, 1);
        assert!(
            serial.iter().all(Option::is_some),
            "{}x{}: decodes",
            case.w,
            case.h
        );
        for threads in BUDGETS {
            assert!(
                decode_all(&packets, threads) == serial,
                "{}x{} partitions {} {:?}: threads {threads}",
                case.w,
                case.h,
                case.partitions,
                case.effort
            );
        }
    }
}

#[test]
fn damaged_streams_fare_alike_at_every_budget() {
    let stream = encode(Case {
        w: 176,
        h: 144,
        frames: 5,
        partitions: 4,
        lf: 16,
        simple: false,
        effort: EncodeEffort::Balanced,
        segmentation: false,
    });
    let mut lcg = 0x0BAD_5EEDu32;
    let mut next = move || {
        lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (lcg >> 8) as usize
    };
    for trial in 0..160 {
        let mut damaged = stream.clone();
        let victim = next() % damaged.len();
        let packet = &mut damaged[victim];
        match trial % 4 {
            // Flip a byte anywhere.
            0 => {
                let at = next() % packet.len();
                packet[at] ^= 1 << (next() % 8);
            }
            // Flip a byte in the header / mode layer.
            1 => {
                let at = next() % packet.len().min(40);
                packet[at] ^= (next() % 255 + 1) as u8;
            }
            // Truncate.
            2 => {
                let keep = next() % packet.len();
                packet.truncate(keep);
            }
            // Overwrite a run with noise.
            _ => {
                let at = next() % packet.len();
                let end = (at + 1 + next() % 24).min(packet.len());
                for b in &mut packet[at..end] {
                    *b = next() as u8;
                }
            }
        }
        let packets: Vec<&[u8]> = damaged.iter().map(Vec::as_slice).collect();
        let serial = decode_all(&packets, 1);
        for threads in [2usize, 4] {
            assert!(
                decode_all(&packets, threads) == serial,
                "trial {trial}: threads {threads}"
            );
        }
    }
}
