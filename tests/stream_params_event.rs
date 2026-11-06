//! Stream-parameter contract of the registry [`Vp8Decoder`]:
//! `output_params()`, `output_params_generation()` and the in-band
//! [`Frame::StreamParams`] event, plus the encoders' answer to the event.
//!
//! Only a key frame can change the decoder's output description — the
//! §9.1 size words and the §9.2 `color_space` bit are key-frame-only. What
//! this pins:
//!
//! 1. the first description is announced by `output_params()` (before the
//!    first frame, from the queued key frame's headers) — no event;
//! 2. a constant stream never emits an event and keeps generation 0, even
//!    across a repeated same-size key frame;
//! 3. each size switch emits exactly ONE event, immediately before the
//!    first picture of the new size, with payload == `output_params()`
//!    and the generation already bumped (1:1 with events);
//! 4. the same ordering holds while draining after `flush()`, and after
//!    `reset()` the comparison is against the last picture returned
//!    before the reset (the DoS cap survives the reset too);
//! 5. §9.1 `show_frame = 0` pictures are never returned, and an invisible
//!    key frame of a new size defers the event to the first SHOWN picture
//!    of that size;
//! 6. the staged conformance vectors decode byte-identically through the
//!    trait path with no event;
//! 7. both framework encoders follow a size-changing event with a fresh
//!    §9.1 key frame, and reject what they cannot follow as `Unsupported`.

#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Decoder, DecoderLimits, Encoder, Error, Frame, MatrixCoefficients,
    MediaType, OutputParamsWatch, Packet, PixelFormat, VideoFrame,
};
use oxideav_vp8::{Vp8Decoder, Vp8EncoderConfig, make_encoder, make_encoder_with_config};

const A: (u32, u32) = (64, 48);
const B: (u32, u32) = (80, 56);

fn params(w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("vp8"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

/// Textured drifting I420 source picture `f` of `w × h`, pts `f`.
fn picture(w: u32, h: u32, f: usize) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let (uvw, uvh) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let sc = c + 2 * f;
            y[r * w + c] = (50 + ((r * 3 + sc * 2) % 140) as i32 + (((sc * 7) % 19) as i32 - 9))
                .clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        pts: Some(f as i64),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: uvw,
                data: vec![112; uvw * uvh],
            },
            VideoPlane {
                stride: uvw,
                data: vec![140; uvw * uvh],
            },
        ],
    }
}

fn drain(enc: &mut Box<dyn Encoder>) -> Vec<Packet> {
    let mut out = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => out.push(p),
            Err(Error::NeedMore | Error::Eof) => return out,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
}

/// One K + (n-1) P stream of `(w, h)` with no invisible frames; pts
/// start at `pts0`.
fn kp_stream((w, h): (u32, u32), n: usize, pts0: usize) -> Vec<Packet> {
    let config = Vp8EncoderConfig {
        qindex: 40,
        alt_ref_interval: 0,
        golden_interval: 0,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = make_encoder_with_config(&params(w, h), config).expect("encoder");
    let mut out = Vec::new();
    for f in 0..n {
        let mut pic = picture(w, h, f);
        pic.pts = Some((pts0 + f) as i64);
        enc.send_frame(&Frame::Video(pic)).expect("send");
        out.extend(drain(&mut enc));
    }
    enc.flush().expect("flush");
    out.extend(drain(&mut enc));
    assert_eq!(out.len(), n);
    out
}

/// What one `receive_frame` produced, reduced to what the contract talks
/// about.
#[derive(Debug, PartialEq)]
enum Out {
    Pic { w: u32, h: u32, pts: Option<i64> },
    Event { w: u32, h: u32 },
}

/// Pull everything currently available, checking the event invariants on
/// the way: payload == `output_params()`, generation bumped exactly once
/// per event, `OutputParamsWatch` (both forms) reporting exactly the
/// events, and each picture's plane shape matching `output_params()`.
struct Harness {
    dec: Vp8Decoder,
    watch: OutputParamsWatch,
    event_watch: OutputParamsWatch,
    stream: CodecParameters,
    event_stream: CodecParameters,
    last_gen: u64,
    watch_changes: usize,
    out: Vec<Out>,
}

impl Harness {
    fn new(dec: Vp8Decoder) -> Self {
        Self {
            dec,
            watch: OutputParamsWatch::new(),
            event_watch: OutputParamsWatch::new(),
            stream: CodecParameters::video(CodecId::new("vp8")),
            event_stream: CodecParameters::video(CodecId::new("vp8")),
            last_gen: 0,
            watch_changes: 0,
            out: Vec::new(),
        }
    }

    fn events(&self) -> usize {
        self.out
            .iter()
            .filter(|o| matches!(o, Out::Event { .. }))
            .count()
    }

    /// Pull until `NeedMore` / `Eof`; returns the terminal error.
    fn pull(&mut self) -> Error {
        loop {
            let frame = match self.dec.receive_frame() {
                Ok(f) => f,
                Err(e @ (Error::NeedMore | Error::Eof)) => return e,
                Err(e) => panic!("receive_frame: {e:?}"),
            };
            let op = self.dec.output_params().expect("known once a frame is out");
            let generation = self.dec.output_params_generation();
            let first = self.out.is_empty();
            let changed = self.watch.observe(&self.dec, &mut self.stream);
            match &frame {
                Frame::StreamParams(p) => {
                    assert!(!first, "no event for the first shape");
                    assert!(
                        frame.is_stream_params() && frame.pts().is_none(),
                        "event carries no media"
                    );
                    assert_eq!(p.width, op.width, "payload == output_params (width)");
                    assert_eq!(p.height, op.height, "payload == output_params (height)");
                    assert_eq!(p.pixel_format, op.pixel_format);
                    assert_eq!(p.color_signal, op.color_signal);
                    assert_eq!(p.media_type, MediaType::Video);
                    assert_eq!(
                        generation,
                        self.last_gen.wrapping_add(1),
                        "one bump per event, visible with it"
                    );
                    assert!(changed, "the counter watch sees the event's change");
                    assert!(
                        self.event_watch
                            .observe_frame(&frame, &mut self.event_stream),
                        "the event watch sees the change"
                    );
                    assert!(
                        !self.watch.observe(&self.dec, &mut self.stream),
                        "a second observe reports nothing"
                    );
                    self.watch_changes += 1;
                    self.out.push(Out::Event {
                        w: p.width.unwrap(),
                        h: p.height.unwrap(),
                    });
                }
                Frame::Video(v) => {
                    assert_eq!(generation, self.last_gen, "pictures never bump");
                    if first {
                        assert!(changed, "first observe resolves the description");
                        self.event_stream.merge_output_params(&op);
                    } else {
                        assert!(!changed, "a picture never reports a change itself");
                    }
                    let (w, h) = (op.width.unwrap(), op.height.unwrap());
                    assert_eq!(v.planes.len(), 3);
                    assert_eq!(v.planes[0].stride, w as usize);
                    assert_eq!(v.planes[0].data.len(), (w * h) as usize);
                    let (uvw, uvh) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
                    assert_eq!(v.planes[1].data.len(), uvw * uvh);
                    assert_eq!(v.planes[2].data.len(), uvw * uvh);
                    assert!(
                        !self
                            .event_watch
                            .observe_frame(&frame, &mut self.event_stream)
                    );
                    self.out.push(Out::Pic { w, h, pts: v.pts });
                }
                other => panic!("unexpected frame kind {other:?}"),
            }
            assert_eq!(self.stream.width, op.width);
            assert_eq!(self.stream.height, op.height);
            assert_eq!(self.event_stream.width, op.width);
            assert_eq!(self.event_stream.height, op.height);
            self.last_gen = generation;
        }
    }

    /// Feed packets one at a time, pulling after each.
    fn feed(&mut self, packets: &[Packet]) {
        for p in packets {
            self.dec.send_packet(p).expect("send_packet");
            assert!(matches!(self.pull(), Error::NeedMore));
        }
    }
}

fn pics(shape: (u32, u32), pts: std::ops::Range<i64>) -> Vec<Out> {
    pts.map(|t| Out::Pic {
        w: shape.0,
        h: shape.1,
        pts: Some(t),
    })
    .collect()
}

fn ev(shape: (u32, u32)) -> Vec<Out> {
    vec![Out::Event {
        w: shape.0,
        h: shape.1,
    }]
}

fn new_decoder() -> Vp8Decoder {
    Vp8Decoder::new(CodecId::new("vp8"))
}

#[test]
fn output_params_is_known_before_the_first_frame() {
    let mut dec = new_decoder();
    assert!(dec.output_params().is_none(), "nothing queued yet");
    let packets = kp_stream(A, 2, 0);
    dec.send_packet(&packets[0]).unwrap();
    let p = dec
        .output_params()
        .expect("peeked from the queued key frame");
    assert_eq!((p.width, p.height), (Some(A.0), Some(A.1)));
    assert_eq!(p.pixel_format, Some(PixelFormat::Yuv420P));
    assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P));
    // §9.2 color_space = 0 → the BT.601 matrix; nothing else is stated.
    assert_eq!(p.color_signal.matrix, MatrixCoefficients::BT601_525);
    assert!(p.color_signal.range.is_unspecified());
    assert!(p.color_signal.primaries.is_unspecified());
    assert!(p.color_signal.transfer.is_unspecified());
    assert_eq!(dec.output_params_generation(), 0);

    let mut h = Harness::new(dec);
    assert!(matches!(h.pull(), Error::NeedMore));
    h.feed(&packets[1..]);
    assert_eq!(h.out, pics(A, 0..2));
    assert_eq!(h.dec.output_params_generation(), 0);
    assert_eq!(h.dec.clamping_type(), Some(false));
    assert_eq!(
        h.dec.upscale(),
        Some((oxideav_vp8::ScaleCode::None, oxideav_vp8::ScaleCode::None))
    );
}

#[test]
fn constant_stream_emits_no_event_even_across_a_same_size_key_frame() {
    let mut packets = kp_stream(A, 4, 0);
    packets.extend(kp_stream(A, 3, 4)); // second key frame, same size
    let mut h = Harness::new(new_decoder());
    h.feed(&packets);
    assert_eq!(h.out, pics(A, 0..7));
    assert_eq!(h.events(), 0);
    assert_eq!(h.dec.output_params_generation(), 0);
}

#[test]
fn each_resolution_switch_emits_exactly_one_event_before_its_first_picture() {
    let mut packets = kp_stream(A, 3, 0);
    packets.extend(kp_stream(B, 3, 3));
    packets.extend(kp_stream(A, 2, 6));
    let mut h = Harness::new(new_decoder());
    h.feed(&packets);
    let mut want = pics(A, 0..3);
    want.extend(ev(B));
    want.extend(pics(B, 3..6));
    want.extend(ev(A));
    want.extend(pics(A, 6..8));
    assert_eq!(h.out, want);
    assert_eq!(h.dec.output_params_generation(), 2);
    assert_eq!(h.watch_changes, 2);
}

#[test]
fn flush_drain_keeps_old_frames_event_new_frames_order() {
    let mut packets = kp_stream(A, 2, 0);
    packets.extend(kp_stream(B, 2, 2));
    let mut h = Harness::new(new_decoder());
    for p in &packets {
        h.dec.send_packet(p).unwrap();
    }
    h.dec.flush().unwrap();
    assert!(matches!(h.pull(), Error::Eof));
    let mut want = pics(A, 0..2);
    want.extend(ev(B));
    want.extend(pics(B, 2..4));
    assert_eq!(h.out, want);
    assert!(matches!(h.dec.receive_frame(), Err(Error::Eof)));
}

#[test]
fn reset_compares_against_the_last_picture_returned() {
    let a = kp_stream(A, 2, 0);
    let b = kp_stream(B, 2, 10);
    let mut h = Harness::new(new_decoder());
    h.feed(&a);
    // A seek into the B segment: the first B picture is announced.
    h.dec.reset().unwrap();
    h.feed(&b);
    // A seek back into B again: same shape, no event.
    h.dec.reset().unwrap();
    h.feed(&b);
    // A seek into A: announced again.
    h.dec.reset().unwrap();
    h.feed(&a);
    let mut want = pics(A, 0..2);
    want.extend(ev(B));
    want.extend(pics(B, 10..12));
    want.extend(pics(B, 10..12));
    want.extend(ev(A));
    want.extend(pics(A, 0..2));
    assert_eq!(h.out, want);
    assert_eq!(h.dec.output_params_generation(), 2);
}

#[test]
fn reset_drops_a_held_picture_and_queued_packets() {
    let mut packets = kp_stream(A, 1, 0);
    packets.extend(kp_stream(B, 1, 1));
    let mut h = Harness::new(new_decoder());
    for p in &packets {
        h.dec.send_packet(p).unwrap();
    }
    assert!(matches!(h.dec.receive_frame(), Ok(Frame::Video(_))));
    assert!(matches!(h.dec.receive_frame(), Ok(Frame::StreamParams(_))));
    // The B picture is held behind the event; a seek discards it.
    h.dec.reset().unwrap();
    assert!(matches!(h.dec.receive_frame(), Err(Error::NeedMore)));
    // The description stays the one just announced (B).
    let p = h.dec.output_params().unwrap();
    assert_eq!((p.width, p.height), (Some(B.0), Some(B.1)));
}

#[test]
fn pixel_cap_survives_reset() {
    let limits = DecoderLimits::default().with_max_pixels_per_frame(u64::from(A.0 * A.1));
    let mut dec = Vp8Decoder::with_limits(CodecId::new("vp8"), limits);
    let b = kp_stream(B, 1, 0);
    dec.send_packet(&b[0]).unwrap();
    assert!(matches!(
        dec.receive_frame(),
        Err(Error::ResourceExhausted(_))
    ));
    dec.reset().unwrap();
    dec.send_packet(&b[0]).unwrap();
    assert!(
        matches!(dec.receive_frame(), Err(Error::ResourceExhausted(_))),
        "the cap must still apply after reset"
    );
}

/// §9.1 bit 4 of byte 0 is `show_frame`.
fn hide(p: &Packet) -> Packet {
    let mut p = p.clone();
    p.data[0] &= !0x10;
    p
}

#[test]
fn invisible_key_frame_defers_the_event_to_the_first_shown_picture() {
    let a = kp_stream(A, 2, 0);
    let b = kp_stream(B, 3, 2);
    let mut h = Harness::new(new_decoder());
    h.feed(&a);
    // An invisible key frame of the new size: decoded (it installs the
    // reference slots the B interframes predict from), but nothing is
    // shown, so nothing is announced yet.
    h.dec.send_packet(&hide(&b[0])).unwrap();
    assert!(matches!(h.pull(), Error::NeedMore));
    assert_eq!(h.dec.output_params_generation(), 0);
    let p = h.dec.output_params().unwrap();
    assert_eq!(
        (p.width, p.height),
        (Some(A.0), Some(A.1)),
        "output_params still describes the last picture returned"
    );
    h.feed(&b[1..]);
    let mut want = pics(A, 0..2);
    want.extend(ev(B));
    want.extend(pics(B, 3..5));
    assert_eq!(h.out, want);
    assert_eq!(h.dec.output_params_generation(), 1);
}

#[test]
fn invisible_anchor_frames_are_never_returned_and_never_announce() {
    // The lagged encoder ships one invisible ARNR anchor per completed
    // multi-frame group.
    let anchored = |shape: (u32, u32), pts0: usize| {
        let config = Vp8EncoderConfig {
            qindex: 40,
            alt_ref_interval: 3,
            lookahead_window: 3,
            golden_interval: 0,
            ..Vp8EncoderConfig::default()
        };
        let mut enc = make_encoder_with_config(&params(shape.0, shape.1), config).unwrap();
        let mut out = Vec::new();
        for f in 0..6 {
            let mut pic = picture(shape.0, shape.1, f);
            pic.pts = Some((pts0 + f) as i64);
            enc.send_frame(&Frame::Video(pic)).unwrap();
            out.extend(drain(&mut enc));
        }
        enc.flush().unwrap();
        out.extend(drain(&mut enc));
        out
    };
    let mut packets = anchored(A, 0);
    let a_len = packets.len();
    packets.extend(anchored(B, 6));
    assert!(
        packets.iter().filter(|p| p.data[0] & 0x10 == 0).count() >= 2,
        "the fixture must carry invisible frames"
    );
    assert!(packets.len() > a_len + 6);
    let mut h = Harness::new(new_decoder());
    h.feed(&packets);
    let mut want = pics(A, 0..6);
    want.extend(ev(B));
    want.extend(pics(B, 6..12));
    assert_eq!(h.out, want);
}

// ───────────────────── staged conformance vectors ─────────────────────

fn ivf_frames(ivf: &[u8]) -> Vec<&[u8]> {
    oxideav_vp8::ivf::parse_header(ivf).expect("IVF header");
    let mut off = oxideav_vp8::ivf::IVF_HEADER_LEN;
    let mut out = Vec::new();
    while off < ivf.len() {
        let fh = oxideav_vp8::ivf::parse_frame_header(&ivf[off..]).expect("frame header");
        off += oxideav_vp8::ivf::IVF_FRAME_HEADER_LEN;
        out.push(&ivf[off..off + fh.size as usize]);
        off += fh.size as usize;
    }
    out
}

macro_rules! vector {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("fixtures/", $name, "/input.ivf")).as_slice(),
            include_bytes!(concat!("fixtures/", $name, "/expected.yuv")).as_slice(),
        )
    };
}

#[test]
fn staged_vectors_decode_byte_identically_with_no_event() {
    let vectors = [
        vector!("altref-arnr-on"),
        vector!("golden-update-cycle"),
        vector!("gradient-and-noise-128x128"),
        vector!("i-frame-then-p-frame-64x64"),
        vector!("i-only-64x64"),
        vector!("i-only-loopfilter-high"),
        vector!("i-only-loopfilter-off"),
        vector!("partition-padding-16x16-4parts"),
        vector!("q-high"),
        vector!("q-low"),
        vector!("segment-4-partitions"),
        vector!("tiny-i-only-16x16"),
        vector!("vp8-with-loopfilter-mode-simple"),
    ];
    for (name, ivf, expected) in vectors {
        let hdr = oxideav_vp8::ivf::parse_header(ivf).unwrap();
        let mut h = Harness::new(new_decoder());
        let packets: Vec<Packet> = ivf_frames(ivf)
            .into_iter()
            .map(|f| Packet::new(0, oxideav_core::TimeBase::new(1, 30), f.to_vec()))
            .collect();
        h.feed(&packets);
        assert_eq!(h.events(), 0, "{name}: constant stream");
        assert_eq!(h.dec.output_params_generation(), 0, "{name}");
        let p = h.dec.output_params().unwrap();
        assert_eq!((p.width, p.height), (Some(hdr.width), Some(hdr.height)));

        // Re-decode collecting the bytes (the harness keeps only shapes).
        let mut dec = new_decoder();
        let mut got = Vec::with_capacity(expected.len());
        for p in &packets {
            dec.send_packet(p).unwrap();
            while let Ok(f) = dec.receive_frame() {
                let Frame::Video(v) = f else {
                    panic!("{name}: event in a constant stream")
                };
                for plane in &v.planes {
                    got.extend_from_slice(&plane.data);
                }
            }
        }
        assert_eq!(got.len(), expected.len(), "{name}: shown-frame byte count");
        assert!(got == expected, "{name}: byte-identical to expected.yuv");
    }
}

// ───────────────────────────── encoders ─────────────────────────────

fn event(p: CodecParameters) -> Frame {
    Frame::StreamParams(Box::new(p))
}

/// Decode `packets` through the harness and return its trace.
fn decode_trace(packets: &[Packet]) -> Vec<Out> {
    let mut h = Harness::new(new_decoder());
    h.feed(packets);
    h.out
}

#[test]
fn keyframe_encoder_follows_a_size_event() {
    let mut enc = make_encoder(&params(A.0, A.1)).unwrap();
    let mut packets = Vec::new();
    for f in 0..2 {
        enc.send_frame(&Frame::Video(picture(A.0, A.1, f))).unwrap();
    }
    packets.extend(drain(&mut enc));
    enc.send_frame(&event(params(B.0, B.1))).unwrap();
    assert_eq!(enc.output_params().width, Some(B.0));
    assert_eq!(enc.output_params().height, Some(B.1));
    for f in 2..4 {
        enc.send_frame(&Frame::Video(picture(B.0, B.1, f))).unwrap();
    }
    packets.extend(drain(&mut enc));
    assert!(packets.iter().all(|p| p.flags.keyframe));
    let mut want = pics(A, 0..2);
    want.extend(ev(B));
    want.extend(pics(B, 2..4));
    assert_eq!(decode_trace(&packets), want);
}

#[test]
fn lagged_encoder_rekeys_on_a_size_event() {
    let config = Vp8EncoderConfig {
        qindex: 40,
        alt_ref_interval: 4,
        lookahead_window: 4,
        golden_interval: 0,
        ..Vp8EncoderConfig::default()
    };
    let mut enc = make_encoder_with_config(&params(A.0, A.1), config).unwrap();
    for f in 0..3 {
        enc.send_frame(&Frame::Video(picture(A.0, A.1, f))).unwrap();
    }
    // A same-size event is a no-op (nothing is flushed early).
    enc.send_frame(&event(params(A.0, A.1))).unwrap();
    assert!(drain(&mut enc).is_empty(), "group still filling");
    // The size event closes the A group(s) at the old size.
    enc.send_frame(&event(params(B.0, B.1))).unwrap();
    let mut packets = drain(&mut enc);
    assert_eq!(
        packets.iter().filter(|p| p.pts.is_some()).count(),
        3,
        "the three buffered A pictures are emitted by the event"
    );
    assert_eq!(enc.output_params().width, Some(B.0));
    let a_len = packets.len();
    for f in 3..6 {
        enc.send_frame(&Frame::Video(picture(B.0, B.1, f))).unwrap();
    }
    enc.flush().unwrap();
    packets.extend(drain(&mut enc));
    assert!(
        packets[a_len].flags.keyframe,
        "the first packet after the event is a key frame"
    );
    let mut want = pics(A, 0..3);
    want.extend(ev(B));
    want.extend(pics(B, 3..6));
    assert_eq!(decode_trace(&packets), want);
}

#[test]
fn encoders_reject_an_event_they_cannot_follow() {
    let config = Vp8EncoderConfig {
        alt_ref_interval: 0,
        ..Vp8EncoderConfig::default()
    };
    let mut bad_fmt = params(B.0, B.1);
    bad_fmt.pixel_format = Some(PixelFormat::Rgb24);
    let too_big = params(0x4000, 16);
    let zero = params(0, 16);
    let mut audio = CodecParameters::audio(CodecId::new("vp8"));
    audio.sample_rate = Some(48_000);
    for mut enc in [
        make_encoder(&params(A.0, A.1)).unwrap(),
        make_encoder_with_config(&params(A.0, A.1), config).unwrap(),
    ] {
        for p in [&bad_fmt, &too_big, &zero, &audio] {
            assert!(
                matches!(
                    enc.send_frame(&event(p.clone())),
                    Err(Error::Unsupported(_))
                ),
                "{p:?} must be Unsupported"
            );
        }
        // Rejected events leave the encoder at the old size.
        assert_eq!(enc.output_params().width, Some(A.0));
        enc.send_frame(&Frame::Video(picture(A.0, A.1, 0))).unwrap();
        enc.flush().unwrap();
        assert_eq!(drain(&mut enc).len(), 1);
    }
}
