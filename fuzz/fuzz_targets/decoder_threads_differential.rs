#![no_main]

//! Fuzz: the decoder's thread budget on arbitrary packets (RFC 6386 §9 /
//! §13 / §15 / §16 / §19).
//!
//! The input is cut into packets and fed to two `Vp8DecoderState`s, one
//! serial and one with a thread budget. With a budget the frame is
//! decoded by a different driver: the mode layer is parsed up front,
//! token rows / reconstruction rows run as units on scoped workers and
//! the loop filter runs in row bands. For every packet the two decoders
//! must agree on whether it decodes and, if it does, on every sample of
//! the picture and on the show flag — and neither may panic.
//!
//! To reach the inter-frame paths without first having to synthesise a
//! valid key frame, each iteration starts both decoders from the same
//! small key frame produced by this crate's encoder (its size and
//! partition count picked by the first input byte), then feeds the
//! fuzzed packets. A second mode replays an encoded key + inter pair
//! with fuzz-chosen bytes overwritten, so the mutations land in real
//! mode / token / motion-vector syntax.
//!
//! Input layout:
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]` | bits 0..1: token partitions `1 << n` of the seed frames; bits 2..3: seed picture size class; bit 4: mode (`0` = raw packets, `1` = mutated encoded pair); bits 5..7: thread budget `2 + n` |
//! | `[1]` | seed `y_ac_qi = b % 128`; bit 7 also turns the loop filter on |
//! | raw mode | then repeated `[len_lo, len_hi, bytes…]` packets (length `% 512`) |
//! | mutation mode | then repeated `[offset_lo, offset_hi, value]` overwrites applied to the concatenated key + inter packets |
//!
//! Hard input cap 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    encode_keyframe_with_reconstruction, encode_p_frame_multi_ref_with_refresh_and_coding_options,
    EncodeEffort, I420Frame, InterCodingOptions, KeyframeParams, RefreshControls, Vp8DecoderState,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;

fn pattern(w: usize, h: usize, phase: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y = (0..w * h)
        .map(|i| {
            let (x, r) = (i % w + phase, i / w);
            ((x * 7 + r * 5) ^ ((x / 9) * 31 + (r / 6) * 17)) as u8
        })
        .collect();
    let u = (0..cw * ch).map(|i| (96 + (i % cw) * 3 + phase) as u8).collect();
    let v = (0..cw * ch).map(|i| (160 - (i / cw) * 2) as u8).collect();
    (y, u, v)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }
    let partitions = 1u8 << (data[0] & 0x03);
    let (w, h) = [(48usize, 32usize), (80, 64), (33, 97), (112, 48)][usize::from((data[0] >> 2) & 0x03)];
    let mutate = (data[0] & 0x10) != 0;
    let threads = 2 + usize::from(data[0] >> 5);
    let params = KeyframeParams::default()
        .with_y_ac_qi(data[1] % 128)
        .with_loop_filter_level(if data[1] & 0x80 != 0 { 18 } else { 0 })
        .with_nbr_of_dct_partitions(partitions);

    let (y, u, v) = pattern(w, h, 0);
    let (key, key_planes) =
        encode_keyframe_with_reconstruction(&I420Frame::packed(w as u32, h as u32, &y, &u, &v), &params)
            .expect("seed key frame encodes");
    let (y, u, v) = pattern(w, h, 3);
    let (inter, _) = encode_p_frame_multi_ref_with_refresh_and_coding_options(
        &I420Frame::packed(w as u32, h as u32, &y, &u, &v),
        &key_planes,
        Some(&key_planes),
        None,
        &params,
        &RefreshControls::default(),
        &InterCodingOptions::default()
            .with_effort(EncodeEffort::Fast)
            .with_intra_pick(true),
    )
    .expect("seed inter frame encodes");

    let body = &data[2..];
    let mut packets: Vec<Vec<u8>> = Vec::new();
    if mutate {
        let mut key = key.clone();
        let mut inter = inter.clone();
        let total = key.len() + inter.len();
        for edit in body.chunks_exact(3) {
            let at = (usize::from(edit[0]) | usize::from(edit[1]) << 8) % total;
            if at < key.len() {
                key[at] = edit[2];
            } else {
                inter[at - key.len()] = edit[2];
            }
        }
        packets.push(key);
        packets.push(inter.clone());
        packets.push(inter);
    } else {
        packets.push(key.clone());
        let mut rest = body;
        while rest.len() >= 2 && packets.len() < 12 {
            let len = (usize::from(rest[0]) | usize::from(rest[1]) << 8) % 512;
            rest = &rest[2..];
            let take = len.min(rest.len());
            packets.push(rest[..take].to_vec());
            rest = &rest[take..];
        }
        packets.push(inter.clone());
    }

    let mut serial = Vp8DecoderState::new();
    let mut threaded = Vp8DecoderState::new().with_threads(threads);
    for (n, packet) in packets.iter().enumerate() {
        let a = serial.decode_frame(packet);
        let b = threaded.decode_frame(packet);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert!(
                    a.width == b.width && a.height == b.height && a.y == b.y && a.u == b.u && a.v == b.v,
                    "packet {n}: pictures differ between 1 and {threads} threads"
                );
                assert_eq!(serial.last_frame_shown(), threaded.last_frame_shown());
            }
            (Err(_), Err(_)) => {}
            (a, b) => panic!(
                "packet {n}: serial {:?} but {threads} threads {:?}",
                a.map(|_| ()),
                b.map(|_| ())
            ),
        }
    }
});
