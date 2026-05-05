//! Inter-frame prediction — RFC 6386 §14 / §18.
//!
//! Given a reconstructed reference frame and a motion vector in 1/8-pel
//! units, reconstruct a block of predicted samples. Sub-pel positions use
//! the 6-tap filter for luma (RFC 6386 §18.3 `sixtap_filters`) and a
//! bilinear filter for chroma (RFC 6386 §18.3 `bilinear_filters`).
//!
//! VP8 replicates the reference frame edge when a motion vector points
//! outside the frame; we do that explicitly by clamping integer sample
//! reads against the frame bounds.

use crate::tables::mv::{BILINEAR_FILTERS, SIXTAP_FILTERS};

/// Reference plane descriptor. `stride` is the allocated stride in bytes,
/// while `width`/`height` are the display bounds used to clamp reads into
/// the buffer (VP8 replicates the edge pixels).
pub struct RefPlane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> RefPlane<'a> {
    #[inline]
    fn sample(&self, x: i32, y: i32) -> u8 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[yc * self.stride + xc]
    }
}

/// Apply a 6-tap horizontal filter at a single integer row `src_y` and
/// filter column (fractional offset). `src_x` is the integer pixel just
/// before the sub-pel sample. Returns the rounded filtered value
/// (unclamped 32-bit, caller is responsible for the 128-round + >>7 and
/// clamping when forming the final output).
#[inline]
fn sixtap_h(plane: &RefPlane<'_>, src_x: i32, src_y: i32, fx: usize) -> i32 {
    let taps = &SIXTAP_FILTERS[fx];
    (plane.sample(src_x - 2, src_y) as i32) * taps[0]
        + (plane.sample(src_x - 1, src_y) as i32) * taps[1]
        + (plane.sample(src_x, src_y) as i32) * taps[2]
        + (plane.sample(src_x + 1, src_y) as i32) * taps[3]
        + (plane.sample(src_x + 2, src_y) as i32) * taps[4]
        + (plane.sample(src_x + 3, src_y) as i32) * taps[5]
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 6-tap luma sub-pel interpolation. Produces a `bw × bh` output at
/// position `(dst_x, dst_y)` in `dst`, referencing pixels at
/// `(ref_x_fp, ref_y_fp)` in 1/8-pel units on `plane`.
///
/// The VP8 6-tap filter is symmetric in H/V. When the fractional
/// component in a direction is 0, a simple integer-sample copy is used
/// (i.e. the zero-phase taps [0,0,128,0,0,0] are equivalent to a copy).
pub fn sixtap_predict(
    plane: &RefPlane<'_>,
    ref_x_fp: i32,
    ref_y_fp: i32,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    bw: usize,
    bh: usize,
) {
    // Integer + fractional split. For a negative fractional we still keep
    // fx/fy in [0..8) by adjusting the integer part.
    let int_x = ref_x_fp >> 3;
    let fx = (ref_x_fp & 7) as usize;
    let int_y = ref_y_fp >> 3;
    let fy = (ref_y_fp & 7) as usize;

    if fx == 0 && fy == 0 {
        // Integer copy.
        for j in 0..bh {
            for i in 0..bw {
                let v = plane.sample(int_x + i as i32, int_y + j as i32);
                dst[(dst_y + j) * dst_stride + dst_x + i] = v;
            }
        }
        return;
    }

    // When fy == 0, run horizontal only.
    if fy == 0 {
        for j in 0..bh {
            for i in 0..bw {
                let v = sixtap_h(plane, int_x + i as i32, int_y + j as i32, fx);
                dst[(dst_y + j) * dst_stride + dst_x + i] = clip_u8((v + 64) >> 7);
            }
        }
        return;
    }
    // When fx == 0, run vertical only.
    if fx == 0 {
        let taps = &SIXTAP_FILTERS[fy];
        for j in 0..bh {
            for i in 0..bw {
                let x = int_x + i as i32;
                let base_y = int_y + j as i32;
                let v = (plane.sample(x, base_y - 2) as i32) * taps[0]
                    + (plane.sample(x, base_y - 1) as i32) * taps[1]
                    + (plane.sample(x, base_y) as i32) * taps[2]
                    + (plane.sample(x, base_y + 1) as i32) * taps[3]
                    + (plane.sample(x, base_y + 2) as i32) * taps[4]
                    + (plane.sample(x, base_y + 3) as i32) * taps[5];
                dst[(dst_y + j) * dst_stride + dst_x + i] = clip_u8((v + 64) >> 7);
            }
        }
        return;
    }

    // Two-pass: horizontal then vertical. The intermediate buffer needs
    // bh + 5 rows (extra for the 6-tap vertical support).
    //
    // VP8 motion-comp block sizes are 4 / 8 / 16 (chroma 4 / 8) — bw
    // never exceeds 16, and bh + 5 never exceeds 21 (16 + 5). So a
    // stack-resident `[i32; 21 * 16]` covers every legal call and
    // avoids the per-block heap allocation that the prior `vec![]` did.
    // The indexing keeps `bw` as the row stride exactly as before, so
    // the inner-loop arithmetic is unchanged.
    const MAX_TMP_BW: usize = 16;
    const MAX_TMP_H: usize = 16 + 5;
    debug_assert!(bw <= MAX_TMP_BW, "sixtap_predict: bw {bw} > {MAX_TMP_BW}");
    let tmp_h = bh + 5;
    debug_assert!(
        tmp_h <= MAX_TMP_H,
        "sixtap_predict: tmp_h {tmp_h} > {MAX_TMP_H}"
    );
    // Per RFC 6386 §20.13 (`sixtap_2d` / `sixtap_horiz`), the intermediate
    // buffer is `unsigned char temp[16*(16+5)]` — i.e. each horizontal-pass
    // result is `CLAMP_255((v + 64) >> 7)` before being fed into the
    // vertical pass. With sub-pel filters whose taps can be negative
    // (e.g. position 1: [0, -6, 123, 12, -1, 0]) the unclamped
    // `(v + 64) >> 7` overshoots 0..255 for some input neighbourhoods,
    // and the absent clamp made our two-pass output drift from libvpx /
    // ffmpeg by a few luma units per inter MB. Found by a per-MB
    // pixel-diff against the trace's reconstructed frame; explained by
    // the §20.13 reference's `CLAMP_255(temp); reference++;` pair.
    let mut tmp = [0i32; MAX_TMP_BW * MAX_TMP_H];
    let row_stride = bw;
    for j in 0..tmp_h {
        let yy = int_y + j as i32 - 2;
        for i in 0..bw {
            let v = sixtap_h(plane, int_x + i as i32, yy, fx);
            tmp[j * row_stride + i] = ((v + 64) >> 7).clamp(0, 255);
        }
    }
    let taps = &SIXTAP_FILTERS[fy];
    for j in 0..bh {
        for i in 0..bw {
            // tmp row for center = j + 2.
            let base = j + 2;
            let v = tmp[(base - 2) * row_stride + i] * taps[0]
                + tmp[(base - 1) * row_stride + i] * taps[1]
                + tmp[base * row_stride + i] * taps[2]
                + tmp[(base + 1) * row_stride + i] * taps[3]
                + tmp[(base + 2) * row_stride + i] * taps[4]
                + tmp[(base + 3) * row_stride + i] * taps[5];
            dst[(dst_y + j) * dst_stride + dst_x + i] = clip_u8((v + 64) >> 7);
        }
    }
}

/// Bilinear sub-pel interpolation (chroma, 2-tap).
pub fn bilinear_predict(
    plane: &RefPlane<'_>,
    ref_x_fp: i32,
    ref_y_fp: i32,
    dst: &mut [u8],
    dst_stride: usize,
    dst_x: usize,
    dst_y: usize,
    bw: usize,
    bh: usize,
) {
    let int_x = ref_x_fp >> 3;
    let fx = (ref_x_fp & 7) as usize;
    let int_y = ref_y_fp >> 3;
    let fy = (ref_y_fp & 7) as usize;

    if fx == 0 && fy == 0 {
        for j in 0..bh {
            for i in 0..bw {
                let v = plane.sample(int_x + i as i32, int_y + j as i32);
                dst[(dst_y + j) * dst_stride + dst_x + i] = v;
            }
        }
        return;
    }

    let hx = &BILINEAR_FILTERS[fx];
    let hy = &BILINEAR_FILTERS[fy];

    // Horizontal first produces bh+1 rows.
    //
    // VP8 chroma bilinear blocks are at most 8 wide / 8 tall (so
    // tmp_h <= 9). A stack-resident `[i32; 16 * 17]` (sized for the
    // worst-case luma path that may also dispatch through this fn)
    // removes the per-block heap allocation. Same indexing scheme.
    const MAX_TMP_BW: usize = 16;
    const MAX_TMP_H: usize = 16 + 1;
    debug_assert!(bw <= MAX_TMP_BW, "bilinear_predict: bw {bw} > {MAX_TMP_BW}");
    let tmp_h = bh + 1;
    debug_assert!(
        tmp_h <= MAX_TMP_H,
        "bilinear_predict: tmp_h {tmp_h} > {MAX_TMP_H}"
    );
    let mut tmp = [0i32; MAX_TMP_BW * MAX_TMP_H];
    let row_stride = bw;
    for j in 0..tmp_h {
        let yy = int_y + j as i32;
        for i in 0..bw {
            let v = (plane.sample(int_x + i as i32, yy) as i32) * hx[0]
                + (plane.sample(int_x + i as i32 + 1, yy) as i32) * hx[1];
            tmp[j * row_stride + i] = (v + 64) >> 7;
        }
    }
    for j in 0..bh {
        for i in 0..bw {
            let v = tmp[j * row_stride + i] * hy[0] + tmp[(j + 1) * row_stride + i] * hy[1];
            dst[(dst_y + j) * dst_stride + dst_x + i] = clip_u8((v + 64) >> 7);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_copy() {
        let data: Vec<u8> = (0..16 * 16).map(|i| (i as u8).wrapping_mul(3)).collect();
        let plane = RefPlane {
            data: &data,
            stride: 16,
            width: 16,
            height: 16,
        };
        let mut dst = vec![0u8; 16 * 16];
        sixtap_predict(&plane, 4 * 8, 4 * 8, &mut dst, 16, 0, 0, 4, 4);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[j * 16 + i], data[(4 + j) * 16 + (4 + i)]);
            }
        }
    }

    #[test]
    fn constant_image_stays_constant() {
        let data = vec![128u8; 16 * 16];
        let plane = RefPlane {
            data: &data,
            stride: 16,
            width: 16,
            height: 16,
        };
        let mut dst = vec![0u8; 16 * 16];
        // Various sub-pel offsets.
        for fx in 0..8 {
            for fy in 0..8 {
                sixtap_predict(&plane, 4 * 8 + fx, 4 * 8 + fy, &mut dst, 16, 0, 0, 4, 4);
                for v in &dst[0..4] {
                    assert_eq!(*v, 128, "constant image should filter to constant");
                }
            }
        }
    }

    #[test]
    fn bilinear_constant() {
        let data = vec![200u8; 16 * 16];
        let plane = RefPlane {
            data: &data,
            stride: 16,
            width: 16,
            height: 16,
        };
        let mut dst = vec![0u8; 16 * 16];
        bilinear_predict(&plane, 4 * 8 + 3, 4 * 8 + 5, &mut dst, 16, 0, 0, 4, 4);
        for v in &dst[0..4] {
            assert_eq!(*v, 200);
        }
    }
}
