#![no_main]

//! Fuzz: the §9.5 multi-DCT-partition layout walk on the decode side —
//! `decoder::carve_dct_partitions` (the 3-byte little-endian size-table
//! split + per-partition slice bounds) and the §20.4 row-interleaved
//! token descent that reads macroblock row `r` from partition
//! `r % nbr_of_dct_partitions`.
//!
//! ## Why this path is hard to reach from raw bytes
//!
//! `decode_vp8` only carves more than one DCT partition when the §19.2
//! first-partition control header decodes far enough to yield
//! `log2_nbr_of_dct_partitions > 0` AND the subsequent size table points
//! at byte ranges that lie inside the supplied buffer. Random libFuzzer
//! bytes essentially never survive the §9.1 start-code / §19.2 bool-coded
//! header validation long enough to set `nbr_of_dct_partitions` to 2 / 4
//! / 8 with a self-consistent size table, so the `panic_free_*` decode
//! targets exercise `carve_dct_partitions` almost exclusively on its
//! early `num_partitions == 1` short-circuit. The multi-partition
//! truncation / overshoot branches (`TruncatedPartitionSizes`,
//! `TruncatedDctPartition { index, .. }`) and the round-robin partition
//! selection in the residual decode go essentially unfuzzed.
//!
//! ## Construction
//!
//! This target builds a **structurally valid** multi-partition key frame
//! with the crate's own `encode_keyframe` (partition count drawn from
//! `{2, 4, 8}` so the §9.5 size table is always present), confirms it
//! round-trips through `decode_vp8` as a sanity oracle, then applies
//! fuzzer-driven mutations *targeted at the size table and the partition
//! body boundaries* before feeding each mutant back to `decode_vp8`:
//!
//! 1. **Size-table corruption** — overwrite one 3-byte LE size word with
//!    fuzz bytes. Drives the `consumed > body.len()` overshoot scan and
//!    the per-slot `off + sz > body.len()` bounds check, plus the legal
//!    case where a smaller-than-real size shifts every following
//!    partition boundary (re-routing which bytes each `BoolDecoder`
//!    instance reads, exercising the §7.3 out-of-data renormalisation
//!    tail in several partitions at once).
//! 2. **Tail truncation** — cut the buffer at a fuzz-chosen offset inside
//!    the DCT section, so a declared size now overshoots the available
//!    body (`TruncatedDctPartition`) or the size table itself is cut
//!    short (`TruncatedPartitionSizes`).
//! 3. **`first_partition_size` field rewrite** — perturb the 19-bit
//!    field in the §9.1 uncompressed header so the DCT section offset
//!    moves, mis-aligning where the size table is read from (the
//!    `TruncatedFirstPartition` guard and a shifted, still-in-bounds
//!    section both get hit).
//!
//! The locations to mutate are computed from the freshly parsed
//! `Vp8FrameHeader` (`header_bytes_consumed`, `first_partition_size`),
//! so the harness always knows exactly where the §9.5 table lives without
//! re-implementing any container framing.
//!
//! ## Oracle
//!
//! Beyond panic-freedom: the clean (unmutated) encode MUST decode back
//! to the source dimensions — a regression in multi-partition assembly
//! that broke even the happy path would surface immediately rather than
//! hiding behind the mutation noise. Every decoded plane byte of the
//! clean decode is folded into an FNV-1a accumulator so a stale-length /
//! short-write bug is observed under ASan. Mutant decodes may legally
//! `Err` (that is the whole point); we only fail on a Rust panic.
//!
//! OOM / wall-time caps: dimensions clamped to ≤ 64×64 luma pixels (well
//! inside the shared `accept_dimensions` budget — multi-partition layout
//! needs several macroblock rows to be meaningful but the per-iteration
//! cost is dominated by the repeated decodes, so the frame stays small),
//! at most a handful of mutants per iteration, input ≤ 4 KiB.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    decode_vp8, encode_keyframe, frame_header::Vp8FrameHeader, I420Frame, KeyframeParams,
};
use oxideav_vp8_fuzz::accept_dimensions;

const MAX_INPUT_BYTES: usize = 4 * 1024;
const HEADER_BYTES: usize = 6;

/// FNV-1a 64-bit fold — forces every decoded plane byte to be read so a
/// short-write / stale-length bug surfaces under ASan.
fn fnv1a(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= u64::from(b);
        *acc = acc.wrapping_mul(0x100_0000_01b3);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // §9.1 dimensions in macroblock units. Cap at 4 MBs/axis (64 px) so a
    // multi-partition layout spans enough rows to route across all eight
    // partitions while staying tiny enough to decode many times per
    // iteration. `1 + (b % 4)` lands in 1..=4 MBs i.e. 16..=64 luma px.
    let mb_w = 1u32 + u32::from(data[0] % 4);
    let mb_h = 1u32 + u32::from(data[1] % 4);
    let width = mb_w * 16;
    let height = mb_h * 16;
    if !accept_dimensions(width, height) {
        return;
    }

    // §9.5 partition count drawn from {2, 4, 8} so the size table is
    // always present (the 1-partition layout has no table and is already
    // covered by the panic_free_* decode targets).
    let nbr_of_dct_partitions = match data[2] % 3 {
        0 => 2u8,
        1 => 4u8,
        _ => 8u8,
    };

    let params = KeyframeParams {
        y_ac_qi: 4 + (data[3] % 124), // 4..=127, a non-degenerate quantiser so partitions carry real tokens
        loop_filter_level: data[4] % 64,
        sharpness_level: data[5] % 8,
        nbr_of_dct_partitions,
        filter_type: (data[5] & 0x80) != 0,
        trellis_strength: oxideav_vp8::TrellisStrength::DEFAULT,
    };

    // I420 plane lengths for tightly-packed strides at the chosen
    // dimensions.
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Tile the payload tail across the three planes so a short input still
    // produces populated buffers (modular indexing stays in bounds even
    // for an empty payload because we only index when non-empty).
    let payload = &data[HEADER_BYTES..];
    let mut y_plane = vec![0u8; y_len];
    let mut u_plane = vec![0u8; uv_len];
    let mut v_plane = vec![0u8; uv_len];
    if !payload.is_empty() {
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = payload[i % payload.len()];
        }
        for (i, slot) in u_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len) % payload.len()];
        }
        for (i, slot) in v_plane.iter_mut().enumerate() {
            *slot = payload[(i + y_len + uv_len) % payload.len()];
        }
    }

    let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
    let Ok(stream) = encode_keyframe(&frame, &params) else {
        // A parameter the encoder rejects is a Result, not a panic — the
        // dedicated encode target covers that surface. Nothing to carve.
        return;
    };

    // Sanity oracle: the unmutated multi-partition stream MUST decode and
    // reproduce the source geometry. A happy-path regression in §9.5
    // assembly fails here before any mutation can mask it.
    let clean = decode_vp8(&stream).expect("clean multi-partition keyframe failed to decode");
    assert_eq!(
        (clean.width, clean.height),
        (width, height),
        "§9.1 dimension drift on a clean multi-partition decode"
    );
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    fnv1a(&mut acc, &clean.y);
    fnv1a(&mut acc, &clean.u);
    fnv1a(&mut acc, &clean.v);
    std::hint::black_box(acc);

    // Locate the §9.5 DCT-partition size table from the parsed header so
    // mutations land precisely on the carve path. `header_bytes_consumed`
    // is 10 for a key frame; `first_partition_size` is the §19.2 control
    // partition length; the size table immediately follows it.
    let Ok(header) = Vp8FrameHeader::parse(&stream) else {
        return;
    };
    let first_part_offset = header.header_bytes_consumed;
    let first_part_size = header.first_partition_size as usize;
    let table_offset = first_part_offset.saturating_add(first_part_size);
    // (N - 1) 3-byte LE size words precede the partition bodies.
    let table_len = (nbr_of_dct_partitions as usize - 1) * 3;

    // Derive a few independent mutation knobs from the payload tail (or
    // zeros when it is empty) so a single corpus entry can drive several
    // distinct mutants without growing the harness past a handful of
    // decodes per iteration.
    let knob = |i: usize| -> u8 {
        if payload.is_empty() {
            0
        } else {
            payload[i % payload.len()]
        }
    };

    // --- Mutant 1: corrupt one 3-byte size word in place. -------------
    if table_offset + table_len <= stream.len() && table_len >= 3 {
        let mut m = stream.clone();
        // Pick one of the (N-1) size words.
        let word = (knob(0) as usize) % (nbr_of_dct_partitions as usize - 1);
        let at = table_offset + word * 3;
        m[at] = knob(1);
        m[at + 1] = knob(2);
        m[at + 2] = knob(3) & 0x0F; // keep magnitudes plausible (≤ ~1 MiB)
        let _ = decode_vp8(&m);
    }

    // --- Mutant 2: truncate the buffer inside the DCT section. ---------
    // Cut somewhere in [table_offset, stream.len()) so either the size
    // table is cut short (`TruncatedPartitionSizes`) or a declared body
    // overshoots (`TruncatedDctPartition`).
    if table_offset < stream.len() {
        let span = stream.len() - table_offset;
        let cut = table_offset + (knob(4) as usize % span);
        let _ = decode_vp8(&stream[..cut]);
    }

    // --- Mutant 3: perturb the 19-bit first_partition_size field. ------
    // §9.1 layout: a little-endian 24-bit word at the start of a key
    // frame holds key_frame(1) | version(3) | show(1) | first_part(19).
    // Rewriting the high bytes moves the DCT section offset, exercising
    // both the `TruncatedFirstPartition` guard and a shifted-but-in-bounds
    // section whose size table is now read from the wrong place.
    if stream.len() >= 3 {
        let mut m = stream.clone();
        // The 19-bit field spans bits 5..24 of the 24-bit word, i.e. all
        // of bytes [1] and [2] plus the top 3 bits of byte [0]. Mutate
        // bytes [1] / [2] only so the key-frame / start-code-bearing low
        // byte stays intact and the header still parses.
        m[1] = knob(5);
        m[2] = knob(6);
        let _ = decode_vp8(&m);
    }

    // --- Mutant 4: shrink one declared size to a smaller legal value. --
    // A size smaller than the real partition length keeps every boundary
    // in-bounds but re-routes which bytes each per-partition BoolDecoder
    // reads, driving several §7.3 out-of-data renormalisation tails in a
    // single decode. This is a valid-layout stress, not a truncation.
    if table_offset + table_len <= stream.len() && table_len >= 3 {
        let mut m = stream.clone();
        let word = (knob(7) as usize) % (nbr_of_dct_partitions as usize - 1);
        let at = table_offset + word * 3;
        // Read the existing size, halve it, write it back.
        let cur = (m[at] as usize) | ((m[at + 1] as usize) << 8) | ((m[at + 2] as usize) << 16);
        let shrunk = cur / 2;
        m[at] = (shrunk & 0xFF) as u8;
        m[at + 1] = ((shrunk >> 8) & 0xFF) as u8;
        m[at + 2] = ((shrunk >> 16) & 0xFF) as u8;
        let _ = decode_vp8(&m);
    }
});
