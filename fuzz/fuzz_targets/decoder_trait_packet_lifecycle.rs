#![no_main]

//! Fuzz: the `oxideav_core::Decoder` trait driver (`Vp8Decoder`).
//!
//! Every other decode target in this crate enters through the direct
//! API — `decode_vp8`, `Vp8DecoderState::decode_frame`, or one of the
//! per-stage kernels. This target drives the *framework* entry point
//! instead: the `oxideav_core::Decoder` impl exposed via the `registry`
//! feature, walked through its full `send_packet` / `receive_frame` /
//! `flush` / `reset` lifecycle.
//!
//! That path is distinct in two ways no direct-API target reaches:
//!
//! First, the packet/frame plumbing. `send_packet` clones each
//! `Packet` into the internal `VecDeque`; `receive_frame` pops one,
//! feeds its `data` to `Vp8DecoderState::decode_frame`, and on success
//! wraps the result. `flush` flips the EOF latch so a later empty
//! `receive_frame` returns `Error::Eof` rather than `Error::NeedMore`.
//! `reset` rebuilds the inner state and clears the queue. The exact
//! `NeedMore` / `Eof` / decoded-frame interleaving depends on the
//! order and count of sent packets, which the fuzzer drives.
//!
//! Second, the `From<Vp8DecodedFrame> for VideoFrame` conversion. On a
//! successful `receive_frame` the decoded planes are moved into a
//! `VideoFrame` with computed luma / chroma strides
//! (`uv_stride = ceil(width / 2)`). This conversion — including the
//! `pts` copy from the source packet — runs only on the trait path,
//! so it is exercised nowhere else in the fuzz suite.
//!
//! Input format: repeated `[u16-LE len][len payload bytes]` packets
//! (see `oxideav_vp8_fuzz::split_packets`), each payload becoming one
//! `Packet` fed through the trait. Mutating the crate's own
//! fixture-derived seeds keeps a leading keyframe structurally valid
//! while corrupting later packets, so the queue carries a mix of
//! decodable and rejected packets — the realistic shape a container
//! demuxer would hand the trait.
//!
//! Oracles beyond panic-freedom. Plane materialisation: every byte of
//! every plane of every `Frame::Video` produced is folded into an
//! FNV-1a accumulator, so a stride/length mismatch in the conversion
//! (a short plane buffer paired with an over-large declared stride)
//! surfaces under ASan here rather than going unread. Lifecycle
//! invariant: after `flush`, once the queue has drained,
//! `receive_frame` must report `Eof`, never `NeedMore`; after `reset`
//! it must report `NeedMore` again (queue empty, EOF latch cleared). A
//! divergence panics.
//!
//! Caps: input <= 16 KiB; at most 12 packets per iteration; the §9.1
//! dimension cap from [`oxideav_vp8_fuzz::accept_dimensions`] applied
//! to every packet that parses as a key frame so a wire-legal 14-bit
//! width/height pair can't drive a ~384 MiB allocation.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_vp8::decoder::{Vp8Decoder, VP8_CODEC_ID};
use oxideav_vp8::frame_header::Vp8FrameHeader;
use oxideav_vp8_fuzz::{accept_dimensions, split_packets};

const MAX_INPUT_BYTES: usize = 16 * 1024;
const MAX_PACKETS_PER_ITER: usize = 12;

/// Fold a byte slice into a running FNV-1a hash. Forces the harness to
/// read every plane byte the conversion claims to have produced.
fn fnv1a(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= u64::from(b);
        *acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Reject a packet whose key-frame header declares dimensions outside
/// the per-frame luma budget. Inter frames (and unparseable bytes)
/// pass through: an inter frame inherits the locked geometry, which
/// was itself capped when the key frame was accepted.
fn packet_within_budget(payload: &[u8]) -> bool {
    match Vp8FrameHeader::parse(payload) {
        Ok(h) if h.key_frame => {
            let w = u32::from(h.width.unwrap_or(0));
            let height = u32::from(h.height.unwrap_or(0));
            accept_dimensions(w, height)
        }
        _ => true,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let packets = split_packets(data, MAX_PACKETS_PER_ITER);
    if packets.is_empty() {
        return;
    }

    let mut dec = Vp8Decoder::new(CodecId::new(VP8_CODEC_ID));
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;

    // Feed packets one at a time, draining decoded frames after each
    // send. send_packet only queues, so receive_frame is what actually
    // runs the decode; interleaving keeps the queue shallow.
    for (i, pkt) in packets.iter().enumerate() {
        if !packet_within_budget(pkt) {
            continue;
        }
        let packet = Packet::new(i as u32, TimeBase::new(1, 1000), pkt.to_vec());
        if dec.send_packet(&packet).is_err() {
            continue;
        }
        // Pull at most one frame per send to mirror the typical 1:1
        // demux cadence; on Ok, materialise the planes.
        if let Ok(Frame::Video(vf)) = dec.receive_frame() {
            for plane in &vf.planes {
                fnv1a(&mut hash, &plane.data);
            }
        }
    }

    // Flush, then drain whatever is still queued. After the queue is
    // empty, the contract is Eof (not NeedMore).
    let _ = dec.flush();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                for plane in &vf.planes {
                    fnv1a(&mut hash, &plane.data);
                }
            }
            Ok(_) => {}
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => {
                // Post-flush with an empty queue must surface Eof.
                panic!("receive_frame returned NeedMore after flush + drain");
            }
            Err(_) => break,
        }
    }

    // reset clears the EOF latch and the queue; a subsequent empty
    // receive must report NeedMore again, not Eof.
    if dec.reset().is_ok() {
        match dec.receive_frame() {
            Err(Error::NeedMore) => {}
            Err(Error::Eof) => panic!("receive_frame returned Eof after reset"),
            _ => {}
        }
    }

    std::hint::black_box(hash);
});
