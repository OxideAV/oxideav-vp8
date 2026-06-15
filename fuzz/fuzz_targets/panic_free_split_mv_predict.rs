#![no_main]

//! Fuzz: panic-freedom + write-equivalence of the §16.4 / §18 whole-MB
//! SPLITMV prediction synthesiser `predict_split_mv`.
//!
//! `predict_split_mv` builds the full §18.2/§18.3 prediction buffer for a
//! SPLITMV-coded inter macroblock: sixteen luma 4×4 sub-blocks, each
//! interpolated under *its own* §18.1-doubled quarter-pixel vector, plus
//! four chroma 4×4 sub-blocks per plane under the §18.1 four-vector
//! average ([`split_chroma_mvs`]). Per §18.1 page 114 it performs **no**
//! secondary clamp, so any sub-block vector may point arbitrarily outside
//! the clamping zone and the §20.14 edge-replication inside
//! `filter_block_4x4` is the only thing keeping the gather in-bounds.
//!
//! No existing harness reaches this surface with attacker-shaped input:
//!
//! * The stream / IVF decode fuzzers (`decode_stream_token_descent`,
//!   `ivf_demux_decode_walk`, `inter_stream_encode_decode_sequence`) only
//!   reach `predict_split_mv` through SPLITMV mode decode, whose sixteen
//!   vectors are bounded by the §16.4 sub-MV-mode prediction tree — they
//!   never present the function with a freely-chosen full-`i16` vector per
//!   sub-block.
//! * `panic_free_filter_block_into` and `panic_free_sixtap_subpel` drive
//!   the single-sub-block primitives, not the whole-MB SPLITMV assembly
//!   (the chroma-MV §18.1 averaging, the per-sub-block doubling, and the
//!   four-row scratch-copy raster pack).
//! * The in-tree equivalence test
//!   `strided_into_assembly_matches_predict_split_mv` drives only fixed
//!   inputs on a single mid-plane geometry.
//!
//! This target drives `predict_split_mv` directly with an attacker-shaped
//! `(MB grid geometry, target MB position, sixteen luma vectors, version /
//! full-pixel flag)` envelope and, on every iteration, asserts the
//! synthesiser's documented contract by recomputing the same prediction
//! independently from the per-sub-block primitives:
//!
//! * Each of the sixteen luma sub-blocks must equal, byte for byte, the
//!   `[u8; 16]` block `filter_block_4x4` returns for that sub-block's
//!   `stored_luma_mv` (with the version-3 `apply_full_pixel` truncation
//!   applied when set) at its `(mb_col*16 + sc*4, mb_row*16 + sb*4)`
//!   origin — a mismatch (wrong doubling, wrong raster offset, wrong
//!   four-row copy) panics here.
//! * Each of the four U and four V chroma sub-blocks must equal the
//!   `filter_block_4x4` block for the §18.1-averaged vector
//!   ([`split_chroma_mvs`]) at its chroma origin — a regression in the
//!   averaging, the chroma-slot mapping, or the chroma raster pack is
//!   caught.
//!
//! The harness's reference assembly deliberately recomputes
//! `split_chroma_mvs` and the per-sub-block `filter_block_4x4` calls so a
//! drift between `predict_split_mv`'s packing and the primitives it is
//! documented to compose surfaces as an `assert_eq!`.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes      | Meaning |
//! |-----------:|---------|
//! | `[0]`      | `mb_cols = data[0] % 4 + 1` (∈ 1..=4) |
//! | `[1]`      | `mb_rows = data[1] % 4 + 1` (∈ 1..=4) |
//! | `[2]`      | `mb_col` target column, saturated into `0..mb_cols` |
//! | `[3]`      | `mb_row` target row, saturated into `0..mb_rows` |
//! | `[4]`      | `version` byte → `filter_set_for_version` (0 = six-tap, else bilinear) |
//! | `[5]`      | flags: bit0 forces the version-3 `full_pixel` chroma truncation; bits 1..=2 select an MV-shaping class |
//! | `[6..70]`  | sixteen luma vectors, 4 bytes each: `[row_lo, row_hi, col_lo, col_hi]` little-endian `i16` |
//! | `[70..]`   | reference-plane payload tiled into the Y / U / V planes |
//!
//! MV-shaping class (`flags >> 1 & 3`) keeps a fraction of the corpus on
//! in-bounds mid-plane vectors (fast §18.3 paths) while still letting the
//! adversarial class park sub-block support anywhere in the `i16`
//! envelope so the §20.14 clamp is exercised densely:
//!
//! * 0 — raw `i16` vectors verbatim (adversarial; full clamp coverage).
//! * 1 — mask each component to a small signed range (mostly mid-plane).
//! * 2 — whole-pixel vectors only (`& !7`; the §18.3 copy branch).
//! * 3 — all sixteen vectors equal (degenerate-SPLITMV; chroma avg = the
//!   shared vector).
//!
//! Hard caps: input ≤ 8 KiB; MB grid ≤ 4 × 4, so the largest reference is
//! a 64 × 64 luma + two 32 × 32 chroma planes (≤ 6 KiB of plane buffers);
//! all heap buffers bounded; iteration is the fixed 16 + 4 + 4 sub-block
//! compare.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::motion_comp::{
    apply_full_pixel, filter_block_4x4, filter_set_for_version, predict_split_mv, split_chroma_mvs,
    stored_luma_mv, ReferencePlanes,
};
use oxideav_vp8::motion_vector::Mv;

const MAX_INPUT_BYTES: usize = 8 * 1024;
const HEADER_BYTES: usize = 6;
const MV_BYTES: usize = 64; // 16 vectors × 4 bytes
const PAYLOAD_START: usize = HEADER_BYTES + MV_BYTES;

/// Shape one raw `i16` vector pair per the MV-shaping class (see module
/// docs). Class 3 (all-equal) is applied by the caller after the per-MV
/// shape, so here only classes 0..=2 vary.
fn shape_mv(raw: Mv, class: u8) -> Mv {
    match class {
        // Adversarial: verbatim full-`i16` vector.
        0 => raw,
        // Small signed range, mostly mid-plane.
        1 => Mv {
            row: (raw.row % 32) - 16,
            col: (raw.col % 32) - 16,
        },
        // Whole-pixel only: drop the eighth-pixel fraction bits.
        2 => Mv {
            row: raw.row & !7,
            col: raw.col & !7,
        },
        // Class 3 handled by the caller; treat like verbatim here.
        _ => raw,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < PAYLOAD_START || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let mb_cols = (data[0] as usize % 4) + 1;
    let mb_rows = (data[1] as usize % 4) + 1;
    let mb_col = data[2] as usize % mb_cols;
    let mb_row = data[3] as usize % mb_rows;

    let filter_set = filter_set_for_version(data[4]);
    let filters = filter_set.taps();

    let flags = data[5];
    let full_pixel = flags & 1 != 0;
    let mv_class = (flags >> 1) & 0b11;

    // Sixteen per-sub-block luma vectors from the fixed 64-byte window.
    let mut luma_mvs = [Mv::default(); 16];
    for (i, m) in luma_mvs.iter_mut().enumerate() {
        let off = HEADER_BYTES + i * 4;
        let raw = Mv {
            row: i16::from_le_bytes([data[off], data[off + 1]]),
            col: i16::from_le_bytes([data[off + 2], data[off + 3]]),
        };
        *m = shape_mv(raw, mv_class);
    }
    // Class 3: degenerate SPLITMV — every sub-block shares the first vector.
    if mv_class == 3 {
        let shared = luma_mvs[0];
        for m in luma_mvs.iter_mut() {
            *m = shared;
        }
    }

    // Reference I420 planes sized to the MB grid. Tile the payload (or a
    // flags-seeded ramp when empty) so the convolution landscape is
    // non-degenerate but fully deterministic.
    let lw = mb_cols * 16;
    let lh = mb_rows * 16;
    let cw = mb_cols * 8;
    let ch = mb_rows * 8;
    let payload = &data[PAYLOAD_START..];
    let fill = |len: usize, salt: u8| -> Vec<u8> {
        if payload.is_empty() {
            (0..len)
                .map(|i| (i as u8).wrapping_add(flags).wrapping_add(salt))
                .collect()
        } else {
            (0..len)
                .map(|i| payload[i % payload.len()].wrapping_add(salt))
                .collect()
        }
    };
    let y = fill(lw * lh, 0);
    let u = fill(cw * ch, 0x40);
    let v = fill(cw * ch, 0x80);

    let reference = ReferencePlanes {
        y: &y,
        u: &u,
        v: &v,
        y_stride: lw,
        uv_stride: cw,
        mb_cols,
        mb_rows,
    };

    // ---- drive the whole-MB SPLITMV synthesiser ---------------------
    let out = predict_split_mv(&reference, mb_col, mb_row, &luma_mvs, full_pixel, filters);

    // ---- independent reference assembly ------------------------------
    // Recompute the same prediction from the documented primitives:
    // sixteen `filter_block_4x4` luma blocks under the doubled (and
    // optionally full-pixel-truncated) vectors, and four chroma blocks per
    // plane under the §18.1 averaged vectors. Byte-identity is the
    // §16.4/§18 contract `predict_split_mv` ships.
    let y_x0 = mb_col * 16;
    let y_y0 = mb_row * 16;
    for sb in 0..4 {
        for sc in 0..4 {
            let b = sb * 4 + sc;
            let mut ymv = stored_luma_mv(luma_mvs[b]);
            if full_pixel {
                ymv = apply_full_pixel(ymv);
            }
            let blk = filter_block_4x4(
                reference.y,
                reference.y_stride,
                lw,
                lh,
                y_x0 + sc * 4,
                y_y0 + sb * 4,
                ymv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 16 + sc * 4;
                assert_eq!(
                    &out.y[dst..dst + 4],
                    &blk[r * 4..r * 4 + 4],
                    "luma mismatch sub=({sb},{sc}) row={r} mv={:?} mb=({mb_col},{mb_row})",
                    luma_mvs[b]
                );
            }
        }
    }

    let chroma = split_chroma_mvs(&luma_mvs);
    let uv_x0 = mb_col * 8;
    let uv_y0 = mb_row * 8;
    for sb in 0..2 {
        for sc in 0..2 {
            let c = sb * 2 + sc;
            let mut uvmv = chroma[c];
            if full_pixel {
                uvmv = apply_full_pixel(uvmv);
            }
            let ublk = filter_block_4x4(
                reference.u,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            let vblk = filter_block_4x4(
                reference.v,
                reference.uv_stride,
                cw,
                ch,
                uv_x0 + sc * 4,
                uv_y0 + sb * 4,
                uvmv,
                filters,
            );
            for r in 0..4 {
                let dst = (sb * 4 + r) * 8 + sc * 4;
                assert_eq!(
                    &out.u[dst..dst + 4],
                    &ublk[r * 4..r * 4 + 4],
                    "U mismatch sub=({sb},{sc}) row={r} mv={uvmv:?} mb=({mb_col},{mb_row})"
                );
                assert_eq!(
                    &out.v[dst..dst + 4],
                    &vblk[r * 4..r * 4 + 4],
                    "V mismatch sub=({sb},{sc}) row={r} mv={uvmv:?} mb=({mb_col},{mb_row})"
                );
            }
        }
    }
});
