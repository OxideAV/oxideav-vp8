#![no_main]

//! Fuzz: the IVF writer ↔ parser round-trip (`ivf::write_header` /
//! `ivf::write_frame` vs `ivf::parse_header` / `ivf::parse_frame_header`).
//!
//! `parse_headers` throws raw attacker bytes at the two parse
//! functions and `ivf_demux_decode_walk` fuzzes the demux cursor walk,
//! but the *writer* half of the module had no coverage at all — and
//! with it, the round-trip contract: a file assembled from
//! `write_header` + N × `write_frame` must re-parse to exactly the
//! fields that were written. That lockstep is where a field-offset
//! skew, an endianness slip, or a width/height wire-truncation
//! surprise would hide: each half can be self-consistently wrong.
//!
//! Legs and oracles:
//!
//! 1. **Header round-trip** — an `IvfHeader` with attacker-chosen
//!    16-bit dimensions and raw 32-bit framerate / frame-count fields
//!    is serialised and re-parsed; every field must match, the buffer
//!    must be exactly `IVF_HEADER_LEN`, and the declared header-length
//!    word must equal 32. Dimensions are drawn from the full u16 wire
//!    range (the IVF width/height fields are 16-bit on the wire; every
//!    §9.1-legal VP8 dimension fits).
//! 2. **Frame-record round-trip** — up to 8 records with
//!    attacker-chosen pts values (cliff values u64::MAX / 0 included)
//!    and payload slices carved from the fuzz input are appended with
//!    `write_frame`; a cursor walk re-parses each record, asserting
//!    size, pts, and payload byte equality, and lands exactly at the
//!    buffer end.
//! 3. **Mutation leg** — one attacker-chosen byte of the assembled
//!    file is flipped and the same demux walk re-runs with checked
//!    cursor arithmetic: any outcome is fine except a panic or an
//!    out-of-range slice.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::ivf::{
    parse_frame_header, parse_header, write_frame, write_header, IvfHeader, IVF_FRAME_HEADER_LEN,
    IVF_HEADER_LEN, IVF_VP8_FOURCC,
};

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 24;
const MAX_RECORDS: usize = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let width = u32::from(u16::from_le_bytes([data[0], data[1]]));
    let height = u32::from(u16::from_le_bytes([data[2], data[3]]));
    let framerate_num = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let framerate_den = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let frame_count_field = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let record_count = usize::from(data[16]) % (MAX_RECORDS + 1);
    let pts_seed = u64::from_le_bytes([
        data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[16],
    ]);
    let payload_pool = &data[HEADER_BYTES..];

    // ---- Leg 1: header round-trip ------------------------------------
    let mut header = IvfHeader::vp8(width, height, framerate_num, framerate_den);
    header.frame_count = frame_count_field;
    let mut file = write_header(&header);
    assert_eq!(
        file.len(),
        IVF_HEADER_LEN,
        "header must serialise to 32 bytes"
    );
    assert_eq!(
        u16::from_le_bytes([file[6], file[7]]),
        IVF_HEADER_LEN as u16,
        "declared header length must be 32"
    );
    let parsed = parse_header(&file).expect("own header must re-parse");
    assert_eq!(parsed.width, width, "width round-trip drift");
    assert_eq!(parsed.height, height, "height round-trip drift");
    assert_eq!(parsed.framerate_num, framerate_num, "framerate_num drift");
    assert_eq!(parsed.framerate_den, framerate_den, "framerate_den drift");
    assert_eq!(parsed.frame_count, frame_count_field, "frame_count drift");
    assert_eq!(parsed.fourcc, IVF_VP8_FOURCC, "fourcc drift");

    // ---- Leg 2: frame-record round-trip --------------------------------
    // Records carve consecutive chunks off the payload pool; pts values
    // mix the seed with cliff values.
    let mut written: Vec<(u64, &[u8])> = Vec::with_capacity(record_count);
    let mut pool_cursor = 0usize;
    for i in 0..record_count {
        let chunk = if payload_pool.is_empty() {
            &payload_pool[0..0]
        } else {
            let len = usize::from(payload_pool[pool_cursor % payload_pool.len()]) % 64;
            let start = pool_cursor.min(payload_pool.len());
            let end = (start + len).min(payload_pool.len());
            pool_cursor = end + 1;
            &payload_pool[start..end]
        };
        let pts = match i % 4 {
            0 => pts_seed.wrapping_add(i as u64),
            1 => u64::MAX - i as u64,
            2 => 0,
            _ => pts_seed.rotate_left(i as u32),
        };
        write_frame(&mut file, pts, chunk);
        written.push((pts, chunk));
    }

    let mut cursor = IVF_HEADER_LEN;
    for (i, (pts, chunk)) in written.iter().enumerate() {
        let fh = parse_frame_header(&file[cursor..])
            .unwrap_or_else(|e| panic!("own frame header {i} must re-parse: {e:?}"));
        assert_eq!(fh.size as usize, chunk.len(), "record {i} size drift");
        assert_eq!(fh.pts, *pts, "record {i} pts drift");
        let start = cursor + IVF_FRAME_HEADER_LEN;
        assert_eq!(
            &file[start..start + chunk.len()],
            *chunk,
            "record {i} payload byte drift"
        );
        cursor = start + chunk.len();
    }
    assert_eq!(cursor, file.len(), "record walk must land on the file end");

    // ---- Leg 3: single-byte mutation demux walk -------------------------
    if !file.is_empty() {
        let flip = usize::from(data[16]) * 31 % file.len();
        file[flip] ^= 1 << (data[17] % 8);
        let _ = parse_header(&file);
        let mut cursor = IVF_HEADER_LEN.min(file.len());
        let mut steps = 0;
        while cursor < file.len() && steps < MAX_RECORDS + 2 {
            let Ok(fh) = parse_frame_header(&file[cursor..]) else {
                break;
            };
            let Some(next) = cursor
                .checked_add(IVF_FRAME_HEADER_LEN)
                .and_then(|c| c.checked_add(fh.size as usize))
            else {
                break;
            };
            if next > file.len() {
                break;
            }
            cursor = next;
            steps += 1;
        }
    }
});
