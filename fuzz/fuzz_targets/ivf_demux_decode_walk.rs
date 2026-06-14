#![no_main]

//! Fuzz: the IVF container demux walk feeding the stateful VP8 decoder.
//!
//! The crate ships an IVF reader / writer surface (`ivf::parse_header`,
//! `ivf::parse_frame_header`, `ivf::write_header`, `ivf::write_frame`)
//! so downstream consumers can carry VP8 elementary streams in the
//! de-facto-standard single-codec container. Every other decode target
//! in this suite enters *below* that layer:
//!
//!   * `parse_headers` calls `ivf::parse_header` and
//!     `ivf::parse_frame_header` once each on the raw front bytes — it
//!     never advances a cursor, never walks a second record, and never
//!     feeds an extracted payload anywhere.
//!   * `panic_free_decoder_state` and `decoder_trait_packet_lifecycle`
//!     drive `Vp8DecoderState::decode_frame` from a synthetic
//!     `[u16-LE len][payload]` packetiser (`split_packets`) that
//!     deliberately inherits NO container framing — its `len` is a
//!     16-bit count it bounds-checks itself.
//!
//! That leaves the actual container demux loop — the one a real caller
//! writes to play an `.ivf` file — completely un-fuzzed. This target is
//! that loop, driven on attacker bytes:
//!
//!   1. `ivf::parse_header(data)` for the 32-byte global header.
//!   2. A frame-record walk from offset `IVF_HEADER_LEN`: at each step
//!      `ivf::parse_frame_header(&data[off..])` yields a 32-bit
//!      attacker-controlled payload `size`; the loop must then carve
//!      `data[off + 12 .. off + 12 + size]` and advance the cursor past
//!      it. That slice arithmetic is the integer-overflow / out-of-range
//!      surface a hostile `size` field targets — `off + 12 + size` can
//!      overflow `usize` on a 32-bit host and can exceed `data.len()` on
//!      any host. The walk uses checked arithmetic and a saturating
//!      bounds clamp; the contract under test is that a corrupt `size`
//!      produces a clean stop (or a clamped final record), never a
//!      panic / slice-out-of-bounds / arithmetic-overflow abort.
//!   3. Each carved payload is fed to `Vp8DecoderState::decode_frame`.
//!      A single `Vp8DecoderState` persists across the whole walk, so
//!      the §9.7 LAST / GOLDEN / ALTREF reference ladder is exercised
//!      exactly as it would be when playing a multi-frame `.ivf`: the
//!      first structurally-valid keyframe locks the geometry and every
//!      later record is an inter frame that may reference garbage. Any
//!      `Ok` / `Err` is fine; only a panic fails the target.
//!
//! Oracle beyond panic-freedom: every byte of every successfully
//! decoded frame's Y / U / V planes is folded into an FNV-1a
//! accumulator, so a short-write or stride miscount in the decode path
//! reached only through the container walk surfaces as an ASan / debug
//! out-of-bounds rather than silently passing. The accumulator is
//! consumed via `std::hint::black_box` so the fold is not optimised out.
//!
//! Hard caps (defence-in-depth against OOM / runaway wall time):
//!   * input  ≤ 8 KiB,
//!   * at most 64 frame records decoded per iteration,
//!   * decoded frames whose visible geometry exceeds the shared
//!     `MAX_LUMA_PIXELS` budget are dropped before the plane fold (the
//!     decoder already produced them; the fold is what we skip).

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::ivf::{parse_frame_header, parse_header, IVF_FRAME_HEADER_LEN, IVF_HEADER_LEN};
use oxideav_vp8::Vp8DecoderState;
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 8 * 1024;
const MAX_FRAMES_PER_ITER: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §1: the 32-byte global header. A buffer that fails this gate is
    // not an IVF file at all — nothing further to walk.
    if parse_header(data).is_err() {
        return;
    }

    let mut state = Vp8DecoderState::new();
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis.
    let mut off = IVF_HEADER_LEN;
    let mut frames = 0usize;

    while frames < MAX_FRAMES_PER_ITER {
        // Need at least one 12-byte frame header remaining.
        let rest = match data.get(off..) {
            Some(r) if r.len() >= IVF_FRAME_HEADER_LEN => r,
            _ => break,
        };
        let fh = match parse_frame_header(rest) {
            Ok(fh) => fh,
            Err(_) => break,
        };

        // Cursor past the 12-byte record header. `off + 12` cannot
        // overflow here: `off <= data.len() <= 8 KiB`.
        let payload_start = off + IVF_FRAME_HEADER_LEN;

        // Attacker-controlled payload length. The end offset is the
        // overflow-prone term — compute it with checked arithmetic and
        // clamp to the buffer end. A `size` that runs past the buffer
        // yields the truncated tail (the demux's best effort) and then
        // the next iteration's `data.get(off..)` returns `None`, ending
        // the walk cleanly.
        let size = fh.size as usize;
        let payload_end = payload_start
            .checked_add(size)
            .map(|e| e.min(data.len()))
            .unwrap_or(data.len());

        // `payload_start` can itself sit past the buffer end if a prior
        // record's clamped advance landed exactly at `data.len()`; guard
        // the slice so the range is always well-ordered.
        let payload = data.get(payload_start..payload_end).unwrap_or(&[][..]);

        if let Ok(frame) = state.decode_frame(payload) {
            // Drop the plane fold for pathologically large geometries
            // (the decoder already produced the planes; we only skip the
            // per-byte walk to bound wall time). `accept_dimensions`
            // also rejects a zero axis.
            if accept_dimensions(frame.width, frame.height) {
                for plane in [&frame.y, &frame.u, &frame.v] {
                    for &b in plane.iter() {
                        acc ^= u64::from(b);
                        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
        }

        frames += 1;

        // Advance to the next record. If the declared `size` overflowed
        // or ran past the buffer, `payload_end` is `data.len()`, so the
        // next `data.get(off..)` returns a sub-12-byte (or empty) slice
        // and the loop terminates.
        let next = match payload_start.checked_add(size) {
            Some(n) => n,
            None => break,
        };
        if next <= off {
            // A zero-size record must still make forward progress past
            // its own 12-byte header (already guaranteed by `+ 12`); a
            // non-advancing cursor would loop forever, so bail.
            break;
        }
        off = next;
    }

    std::hint::black_box(acc);
});
