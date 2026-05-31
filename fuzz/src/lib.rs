//! Shared helpers for `oxideav-vp8`'s fuzz targets.
//!
//! Kept deliberately tiny: the dimension cap below is the single
//! cross-target guard against OOM-by-design (a wire-legal §9.1
//! 14-bit width/height pair multiplies to ~256 Mpx, whose I420 raster
//! is ~384 MiB — instant OOM on the fuzz runner). Each per-target
//! harness consults [`accept_dimensions`] before letting bytes flow
//! into the dequant / reconstruct / loop-filter pipeline.

/// Maximum visible-luma-pixel budget per decoded frame.
///
/// 256 × 256 = 65 536 luma pixels; the matching I420 raster is
/// 65 536 + 2 × 16 384 ≈ 96 KiB, comfortably inside the fuzz
/// runner's per-iteration memory cap. The §9.1 wire range is
/// 14 bits per axis (so up to 16 383 × 16 383 ≈ 268 Mpx) — gating
/// on this cap rejects the pathological extremes without
/// suppressing meaningful coverage of the parse + early-decode
/// paths, which are exercised before any allocation by the
/// `parse_headers` target.
pub const MAX_LUMA_PIXELS: u64 = 256 * 256;

/// Return `true` iff `width × height` fits inside [`MAX_LUMA_PIXELS`]
/// and neither axis is zero. Targets that allocate decoded planes
/// short-circuit on `false`.
pub fn accept_dimensions(width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    let pixels = u64::from(width) * u64::from(height);
    pixels <= MAX_LUMA_PIXELS
}

/// Split `data` into a sequence of length-prefixed packets.
///
/// Format: repeated `[u16 len][len bytes payload]`. Used by the
/// stateful-decoder target to materialise a multi-packet stream from
/// flat libfuzzer bytes without inheriting any container's framing
/// rules. Returns at most `max_packets` packets; the tail is
/// silently dropped once the cap is reached so a long input doesn't
/// blow per-iteration wall time.
pub fn split_packets(data: &[u8], max_packets: usize) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor + 2 <= data.len() && out.len() < max_packets {
        let len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + len > data.len() {
            break;
        }
        out.push(&data[cursor..cursor + len]);
        cursor += len;
    }
    out
}
