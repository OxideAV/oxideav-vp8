//! VP8 loop filter — RFC 6386 §15.
//!
//! Only key-frame relevant pieces are wired up: the simple and normal
//! modes are both supported for MB- and sub-block-edge filtering. The
//! filter operates in place on the reconstructed YUV planes after
//! intra prediction + IDCT residue has been added.
//!
//! For an I-frame all macroblocks share `loop_filter_level` because
//! ref-frame deltas only matter for predictions involving inter / golden
//! / altref references; on keyframes they default to zero.
//!
//! ## SIMD
//!
//! With the `simd` cargo feature on nightly Rust, the
//! `filter_simple_horizontal` path uses `std::simd` to process 16
//! horizontal pixels per chunk via `Simd<u8, 16>` (Y MB row width is
//! exactly 16). The vectorised path is bit-exact with the scalar
//! reference — every per-pixel arithmetic op (clamp, signed shift,
//! mask-then-replace) has a direct lane-wise simd equivalent — and
//! falls back to scalar for the trailing `width % 16` pixels. The
//! more elaborate `normal_filter` (with its MB-edge wide branch and
//! per-pixel HEV gate) is left for a follow-up round; vertical edges
//! need a transposed-load layout that is also queued.

#[inline]
fn clamp(v: i32) -> i32 {
    v.clamp(-128, 127)
}

#[inline]
fn u_to_i(v: u8) -> i32 {
    v as i32 - 128
}

#[inline]
fn i_to_u(v: i32) -> u8 {
    (v + 128).clamp(0, 255) as u8
}

#[inline]
fn abs_diff(a: u8, b: u8) -> i32 {
    (a as i32 - b as i32).abs()
}

#[inline]
fn simple_threshold(p1: u8, p0: u8, q0: u8, q1: u8, edge_limit: i32) -> bool {
    let mask = (abs_diff(p0, q0) * 2 + abs_diff(p1, q1) / 2) <= edge_limit;
    mask
}

#[inline]
fn normal_threshold(
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
    edge_limit: i32,
    interior_limit: i32,
) -> bool {
    if !simple_threshold(p1, p0, q0, q1, edge_limit) {
        return false;
    }
    abs_diff(p3, p2) <= interior_limit
        && abs_diff(p2, p1) <= interior_limit
        && abs_diff(p1, p0) <= interior_limit
        && abs_diff(q1, q0) <= interior_limit
        && abs_diff(q2, q1) <= interior_limit
        && abs_diff(q3, q2) <= interior_limit
}

#[inline]
fn high_edge_variance(p1: u8, p0: u8, q0: u8, q1: u8, hev: i32) -> bool {
    abs_diff(p1, p0) > hev || abs_diff(q1, q0) > hev
}

/// Simple-mode 4-tap filter on a single edge crossing (`p1 p0 | q0 q1`).
#[inline]
fn simple_filter(p1: u8, p0: u8, q0: u8, q1: u8) -> (u8, u8) {
    let p0i = u_to_i(p0);
    let q0i = u_to_i(q0);
    let p1i = u_to_i(p1);
    let q1i = u_to_i(q1);
    let mut a = 3 * (q0i - p0i);
    a += clamp(p1i - q1i);
    a = clamp(a);
    let b = clamp(a + 3) >> 3;
    let a = clamp(a + 4) >> 3;
    let new_q0 = i_to_u(q0i - a);
    let new_p0 = i_to_u(p0i + b);
    (new_p0, new_q0)
}

/// Normal-mode filter (RFC §15.4) — adjusts up to 3 px on each side
/// depending on HEV / interior masks.
#[inline]
fn normal_filter(
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    is_mb_edge: bool,
    hev_threshold: i32,
) -> (u8, u8, u8, u8, u8, u8) {
    let hev = high_edge_variance(p1, p0, q0, q1, hev_threshold);
    let p0i = u_to_i(p0);
    let q0i = u_to_i(q0);
    let p1i = u_to_i(p1);
    let q1i = u_to_i(q1);
    let p2i = u_to_i(p2);
    let q2i = u_to_i(q2);

    let mut a = clamp(p1i - q1i);
    if !hev {
        // No HEV: use the smoothing branch.
        a = 0;
    }
    let mut a = clamp(3 * (q0i - p0i) + a);

    if is_mb_edge && !hev {
        // MB-edge wide smoothing per RFC 6386 §15.3 (MBfilter, !hev branch).
        //   w  = c(c(p1 - q1) + 3*(q0 - p0))
        //   a3 = c((27*w + 63) >> 7); P0/Q0 += ±a3
        //   a2 = c((18*w + 63) >> 7); P1/Q1 += ±a2
        //   a1 = c(( 9*w + 63) >> 7); P2/Q2 += ±a1
        // The clamp goes around the SHIFTED expression — clamping the
        // pre-shift value first yields ~0 in the common case and breaks
        // every wide-filter pass.
        let w = clamp(clamp(p1i - q1i) + 3 * (q0i - p0i));
        let a3 = clamp((27 * w + 63) >> 7);
        let a2 = clamp((18 * w + 63) >> 7);
        let a1 = clamp((9 * w + 63) >> 7);
        let new_p0 = i_to_u(p0i + a3);
        let new_q0 = i_to_u(q0i - a3);
        let new_p1 = i_to_u(p1i + a2);
        let new_q1 = i_to_u(q1i - a2);
        let new_p2 = i_to_u(p2i + a1);
        let new_q2 = i_to_u(q2i - a1);
        return (new_p2, new_p1, new_p0, new_q0, new_q1, new_q2);
    }

    let b = clamp(a + 3) >> 3;
    a = clamp(a + 4) >> 3;
    let new_q0 = i_to_u(q0i - a);
    let new_p0 = i_to_u(p0i + b);

    let (new_p1, new_q1) = if !hev {
        let a2 = (a + 1) >> 1;
        (i_to_u(p1i + a2), i_to_u(q1i - a2))
    } else {
        (p1, q1)
    };

    (p2, new_p1, new_p0, new_q0, new_q1, q2)
}

/// Filter `level` parameters helper — derives the three thresholds
/// per RFC 6386 §15.4.
///
///   interior_limit = loop_filter_level, then if sharpness > 0
///       interior_limit >>= sharpness > 4 ? 2 : 1
///       interior_limit = min(interior_limit, 9 - sharpness)
///       interior_limit = max(interior_limit, 1)
///
///   mbedge_limit   = ((loop_filter_level + 2) * 2) + interior_limit
///   sub_bedge_limit = (loop_filter_level * 2) + interior_limit
///
///   hev_threshold (key frames):
///       level >= 40 → 2
///       level >= 15 → 1
///       else → 0
///   hev_threshold (interframes — note the extra "level >= 20 → 2"
///   tier and the "level >= 40 → 3" cap):
///       level >= 40 → 3
///       level >= 20 → 2
///       level >= 15 → 1
///       else → 0
///
/// `mb_edge=true` returns the inter-macroblock variant; `false` returns
/// the inter-subblock variant.
#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    pub edge_limit: i32,
    pub interior_limit: i32,
    pub hev_threshold: i32,
}

impl FilterParams {
    /// Convenience for key-frame callers (the in-tree decoder/encoder
    /// historically only filtered keyframes through this path).
    pub fn for_mb(level: u8, sharpness: u8, mb_edge: bool) -> Self {
        Self::for_mb_typed(level, sharpness, mb_edge, true)
    }

    pub fn for_mb_typed(level: u8, sharpness: u8, mb_edge: bool, key_frame: bool) -> Self {
        let l = level as i32;
        let mut interior = l;
        if sharpness > 0 {
            interior >>= if sharpness > 4 { 2 } else { 1 };
            let cap = 9 - sharpness as i32;
            if interior > cap {
                interior = cap;
            }
        }
        if interior < 1 {
            interior = 1;
        }
        let edge = if mb_edge {
            ((l + 2) * 2) + interior
        } else {
            (l * 2) + interior
        };
        let hev = if key_frame {
            if l >= 40 {
                2
            } else if l >= 15 {
                1
            } else {
                0
            }
        } else if l >= 40 {
            3
        } else if l >= 20 {
            2
        } else if l >= 15 {
            1
        } else {
            0
        };
        Self {
            edge_limit: edge,
            interior_limit: interior,
            hev_threshold: hev,
        }
    }
}

/// Apply the simple-mode loop filter to a MB-edge column at `(x, y)`
/// with `width × height` boundary. `simple` mode only filters the four
/// pixels closest to the edge.
pub fn filter_simple_vertical(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    width: usize,
    height: usize,
    params: FilterParams,
) {
    if x < 2 || x + 2 > width {
        return;
    }
    for j in 0..height {
        let row = j * stride;
        let p1 = plane[row + x - 2];
        let p0 = plane[row + x - 1];
        let q0 = plane[row + x];
        let q1 = plane[row + x + 1];
        if simple_threshold(p1, p0, q0, q1, params.edge_limit) {
            let (np0, nq0) = simple_filter(p1, p0, q0, q1);
            plane[row + x - 1] = np0;
            plane[row + x] = nq0;
        }
    }
}

/// Apply the simple-mode loop filter to a MB-edge row at `(x, y)`.
pub fn filter_simple_horizontal(
    plane: &mut [u8],
    stride: usize,
    y: usize,
    width: usize,
    height: usize,
    params: FilterParams,
) {
    if y < 2 || y + 2 > height {
        return;
    }
    // SIMD body: process `width` in blocks of 16 then fall through to
    // the scalar tail. The simd variant is bit-exact with the scalar
    // `simple_threshold` + `simple_filter` pair. The cfg branch is
    // written as separate `let` bindings so the default-stable build
    // doesn't see an unused `mut`.
    #[cfg(feature = "simd")]
    let start = simd::filter_simple_horizontal_simd(plane, stride, y, width, params);
    #[cfg(not(feature = "simd"))]
    let start = 0usize;
    for i in start..width {
        let p1 = plane[(y - 2) * stride + i];
        let p0 = plane[(y - 1) * stride + i];
        let q0 = plane[y * stride + i];
        let q1 = plane[(y + 1) * stride + i];
        if simple_threshold(p1, p0, q0, q1, params.edge_limit) {
            let (np0, nq0) = simple_filter(p1, p0, q0, q1);
            plane[(y - 1) * stride + i] = np0;
            plane[y * stride + i] = nq0;
        }
    }
}

/// Apply the normal-mode loop filter to a vertical edge.
pub fn filter_normal_vertical(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    width: usize,
    height: usize,
    params: FilterParams,
    is_mb_edge: bool,
) {
    if x < 4 || x + 4 > width {
        return;
    }
    for j in 0..height {
        let row = j * stride;
        let p3 = plane[row + x - 4];
        let p2 = plane[row + x - 3];
        let p1 = plane[row + x - 2];
        let p0 = plane[row + x - 1];
        let q0 = plane[row + x];
        let q1 = plane[row + x + 1];
        let q2 = plane[row + x + 2];
        let q3 = plane[row + x + 3];
        if !normal_threshold(
            p3,
            p2,
            p1,
            p0,
            q0,
            q1,
            q2,
            q3,
            params.edge_limit,
            params.interior_limit,
        ) {
            continue;
        }
        let (np2, np1, np0, nq0, nq1, nq2) =
            normal_filter(p2, p1, p0, q0, q1, q2, is_mb_edge, params.hev_threshold);
        plane[row + x - 3] = np2;
        plane[row + x - 2] = np1;
        plane[row + x - 1] = np0;
        plane[row + x] = nq0;
        plane[row + x + 1] = nq1;
        plane[row + x + 2] = nq2;
    }
}

#[cfg(feature = "simd")]
mod simd {
    //! Vectorised loop-filter helpers. Each function processes a slab
    //! of 16 contiguous horizontal pixels per call and returns the
    //! number of pixels handled (the caller fills the trailing
    //! `width % 16` with the scalar implementation).
    //!
    //! The arithmetic is bit-exact with the scalar reference. Each
    //! per-pixel step in `simple_threshold` / `simple_filter` lifts
    //! to a lane-wise simd op:
    //!   * `(v as i32 - 128)`     → `v.cast::<i16>() - splat(128)`
    //!   * `clamp(-128, 127)`    → `simd_clamp(splat(-128), splat(127))`
    //!   * arithmetic shift `>>3` → `>>` on `Simd<i16, 16>` (i16 is
    //!     signed so `>>` lowers to SAR on x86 / sshr on ARM)
    //!   * `(v + 128).clamp(0, 255) as u8` → `(v + splat(128))
    //!     .simd_clamp(splat(0), splat(255)).cast::<u8>()`
    //!   * `mask.select(filtered, original)` writes the filtered
    //!     value only where `simple_threshold` passes
    //!
    //! Loads use `Simd::from_slice` (panics on short slice — caller
    //! ensures the chunk is in-bounds via the `start + 16 <= width`
    //! check below). Stores use `copy_to_slice`.

    use core::simd::cmp::{SimdOrd, SimdPartialOrd};
    use core::simd::num::SimdInt;
    use core::simd::{Mask, Simd};

    use super::FilterParams;

    const N: usize = 16;

    /// Returns the count of pixels handled by SIMD (caller fills the
    /// rest with the scalar path). When width < 16 returns 0.
    pub(super) fn filter_simple_horizontal_simd(
        plane: &mut [u8],
        stride: usize,
        y: usize,
        width: usize,
        params: FilterParams,
    ) -> usize {
        if width < N {
            return 0;
        }
        let edge_limit: Simd<i16, N> = Simd::splat(params.edge_limit as i16);
        let c128: Simd<i16, N> = Simd::splat(128);
        let c0: Simd<i16, N> = Simd::splat(0);
        let c255: Simd<i16, N> = Simd::splat(255);
        let cmin: Simd<i16, N> = Simd::splat(-128);
        let cmax: Simd<i16, N> = Simd::splat(127);

        let mut handled = 0usize;
        let row_pm2 = (y - 2) * stride;
        let row_pm1 = (y - 1) * stride;
        let row_q0 = y * stride;
        let row_q1 = (y + 1) * stride;

        let mut i = 0;
        while i + N <= width {
            // --- loads (immutable) ---
            let p1u: Simd<u8, N> = Simd::from_slice(&plane[row_pm2 + i..row_pm2 + i + N]);
            let p0u: Simd<u8, N> = Simd::from_slice(&plane[row_pm1 + i..row_pm1 + i + N]);
            let q0u: Simd<u8, N> = Simd::from_slice(&plane[row_q0 + i..row_q0 + i + N]);
            let q1u: Simd<u8, N> = Simd::from_slice(&plane[row_q1 + i..row_q1 + i + N]);

            // u8 → i16 (zero-extend then arithmetic).
            let p1: Simd<i16, N> = p1u.cast();
            let p0: Simd<i16, N> = p0u.cast();
            let q0: Simd<i16, N> = q0u.cast();
            let q1: Simd<i16, N> = q1u.cast();

            // simple_threshold:
            //   abs_diff(p0,q0) * 2 + abs_diff(p1,q1) / 2 <= edge_limit
            // abs_diff via i16: |p0 - q0|.
            let ad_p0q0 = (p0 - q0).abs();
            let ad_p1q1 = (p1 - q1).abs();
            let s1: Simd<i16, N> = Simd::splat(1);
            let thr_lhs = ad_p0q0 + ad_p0q0 + (ad_p1q1 >> s1);
            // SimdPartialOrd::simd_le returns Mask<i16, N>.
            let mask: Mask<i16, N> = thr_lhs.simd_le(edge_limit);

            // simple_filter (bit-exact with scalar):
            //   p0i = p0 - 128; q0i = q0 - 128; ... (signed i16 lanes)
            let p0i = p0 - c128;
            let q0i = q0 - c128;
            let p1i = p1 - c128;
            let q1i = q1 - c128;

            //   a = clamp(3*(q0-p0) + clamp(p1-q1))
            let inner = (p1i - q1i).simd_clamp(cmin, cmax);
            let a0 = (q0i - p0i) * Simd::splat(3) + inner;
            let a0 = a0.simd_clamp(cmin, cmax);
            //   b = clamp(a + 3) >> 3;  a = clamp(a + 4) >> 3;
            // Note: portable_simd's `Shr` impl is `Simd<T,N> >> Simd<T,N>`,
            // so the shift count must be a splat — a bare integer
            // literal does not coerce.
            let s3: Simd<i16, N> = Simd::splat(3);
            let b = (a0 + s3).simd_clamp(cmin, cmax) >> s3;
            let a = (a0 + Simd::splat(4)).simd_clamp(cmin, cmax) >> s3;
            //   new_q0 = i_to_u(q0i - a); new_p0 = i_to_u(p0i + b);
            let new_q0i = (q0i - a + c128).simd_clamp(c0, c255);
            let new_p0i = (p0i + b + c128).simd_clamp(c0, c255);

            // Mask-select: keep originals where threshold rejects.
            let out_p0_i = mask.select(new_p0i, p0);
            let out_q0_i = mask.select(new_q0i, q0);
            // i16 → u8 (lanes are clamped 0..=255 already).
            let out_p0: Simd<u8, N> = out_p0_i.cast();
            let out_q0: Simd<u8, N> = out_q0_i.cast();

            // --- stores ---
            out_p0.copy_to_slice(&mut plane[row_pm1 + i..row_pm1 + i + N]);
            out_q0.copy_to_slice(&mut plane[row_q0 + i..row_q0 + i + N]);

            i += N;
            handled += N;
        }
        handled
    }
}

/// Apply the normal-mode loop filter to a horizontal edge.
pub fn filter_normal_horizontal(
    plane: &mut [u8],
    stride: usize,
    y: usize,
    width: usize,
    height: usize,
    params: FilterParams,
    is_mb_edge: bool,
) {
    if y < 4 || y + 4 > height {
        return;
    }
    for i in 0..width {
        let p3 = plane[(y - 4) * stride + i];
        let p2 = plane[(y - 3) * stride + i];
        let p1 = plane[(y - 2) * stride + i];
        let p0 = plane[(y - 1) * stride + i];
        let q0 = plane[y * stride + i];
        let q1 = plane[(y + 1) * stride + i];
        let q2 = plane[(y + 2) * stride + i];
        let q3 = plane[(y + 3) * stride + i];
        if !normal_threshold(
            p3,
            p2,
            p1,
            p0,
            q0,
            q1,
            q2,
            q3,
            params.edge_limit,
            params.interior_limit,
        ) {
            continue;
        }
        let (np2, np1, np0, nq0, nq1, nq2) =
            normal_filter(p2, p1, p0, q0, q1, q2, is_mb_edge, params.hev_threshold);
        plane[(y - 3) * stride + i] = np2;
        plane[(y - 2) * stride + i] = np1;
        plane[(y - 1) * stride + i] = np0;
        plane[y * stride + i] = nq0;
        plane[(y + 1) * stride + i] = nq1;
        plane[(y + 2) * stride + i] = nq2;
    }
}
