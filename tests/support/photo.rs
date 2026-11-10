//! Deterministic photo-like I420 test picture, shared by the keyframe
//! effort tests, the `keyframe_encode_photo` bench and the
//! `keyframe_timing` example (each pulls this file in with `#[path]`).
//!
//! Integer-only so every platform synthesises the identical bytes: a
//! vertical brightness ramp, four octaves of bilinear value noise (soft
//! large-scale structure), a band of fine two-octave texture, hard-edged
//! rectangles with a regular grid of dark insets, a pair of diagonal
//! strokes, and ±2 of per-pixel grain. `scale` stretches every feature
//! (1 below 2400 px wide, 2 at 2400 px wide) so a larger picture is the same
//! scene sampled more densely, as a higher-resolution capture would be.

#![allow(dead_code)]

/// 32-bit integer mix of a lattice coordinate and a seed.
fn lattice_hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    h
}

/// Bilinear value noise in `-128..=127`, lattice pitch `cell` pixels.
fn value_noise(x: usize, y: usize, cell: usize, seed: u32) -> i32 {
    let cx = (x / cell) as u32;
    let cy = (y / cell) as u32;
    let fx = ((x % cell) * 256 / cell) as i32;
    let fy = ((y % cell) * 256 / cell) as i32;
    let corner = |dx: u32, dy: u32| (lattice_hash(cx + dx, cy + dy, seed) & 0xFF) as i32 - 128;
    let top = corner(0, 0) * (256 - fx) + corner(1, 0) * fx;
    let bottom = corner(0, 1) * (256 - fx) + corner(1, 1) * fx;
    (top * (256 - fy) + bottom * fy) >> 16
}

/// One full-resolution sample of the scene: `(luma, cb, cr)`.
fn scene(x: usize, y: usize, w: usize, h: usize, scale: usize) -> (i32, i32, i32) {
    let s = scale.max(1);
    // Brightness ramp: bright top, darker bottom.
    let mut luma = 200 - (y * 110 / h.max(1)) as i32;
    // Soft structure.
    luma += value_noise(x, y, 64 * s, 1) * 48 / 128;
    luma += value_noise(x, y, 32 * s, 2) * 24 / 128;
    luma += value_noise(x, y, 16 * s, 3) * 12 / 128;
    luma += value_noise(x, y, 8 * s, 4) * 6 / 128;
    let mut cb = 128 + value_noise(x, y, 96 * s, 5) * 30 / 128;
    let mut cr = 128 + value_noise(x, y, 80 * s, 6) * 30 / 128;
    // Fine texture across the lower third.
    if y * 3 >= h * 2 {
        luma += value_noise(x, y, 4 * s, 7) * 28 / 128;
        luma += value_noise(x, y, 2 * s, 8) * 14 / 128;
        cb -= 12;
        cr -= 6;
    }
    // Hard-edged rectangles with a grid of dark insets.
    for (i, &(rx, ry, rw, rh)) in [
        (10usize, 20usize, 18usize, 40usize),
        (45, 10, 25, 52),
        (78, 30, 14, 30),
    ]
    .iter()
    .enumerate()
    {
        let (x0, y0) = (rx * w / 100, ry * h / 100);
        let (x1, y1) = ((rx + rw) * w / 100, (ry + rh) * h / 100);
        if x >= x0 && x < x1 && y >= y0 && y < y1 {
            luma = 90 + 35 * i as i32 + value_noise(x, y, 24 * s, 9 + i as u32) * 10 / 128;
            cb = 118 + 6 * i as i32;
            cr = 136 - 5 * i as i32;
            let (gx, gy) = ((x - x0) / s % 12, (y - y0) / s % 16);
            if (3..9).contains(&gx) && (4..12).contains(&gy) {
                luma -= 55;
            }
        }
    }
    // Two diagonal strokes.
    let d = (x + 2 * y) / s;
    if d % 211 < 3 {
        luma -= 70;
    }
    let e = (3 * x + h * 3 - 3 * y.min(h)) / s;
    if e % 397 < 4 {
        luma += 60;
    }
    // Grain.
    luma += (lattice_hash(x as u32, y as u32, 99) % 5) as i32 - 2;
    (luma, cb, cr)
}

/// Synthesise the `w × h` picture as tightly packed I420 planes
/// `(y, u, v)`; chroma is the 2×2 box average of the scene chroma.
pub fn photo_like_i420(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let scale = (w / 1200).max(1);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0u8; w * h];
    let mut cb_sum = vec![0i32; cw * ch];
    let mut cr_sum = vec![0i32; cw * ch];
    let mut count = vec![0i32; cw * ch];
    for row in 0..h {
        for col in 0..w {
            let (l, cb, cr) = scene(col, row, w, h, scale);
            y[row * w + col] = l.clamp(0, 255) as u8;
            let ci = (row / 2) * cw + col / 2;
            cb_sum[ci] += cb;
            cr_sum[ci] += cr;
            count[ci] += 1;
        }
    }
    let u = cb_sum
        .iter()
        .zip(&count)
        .map(|(&s, &n)| ((s + n / 2) / n).clamp(0, 255) as u8)
        .collect();
    let v = cr_sum
        .iter()
        .zip(&count)
        .map(|(&s, &n)| ((s + n / 2) / n).clamp(0, 255) as u8)
        .collect();
    (y, u, v)
}

/// 64-bit FNV-1a digest — the pin the byte-identity tests compare.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Luma PSNR in dB between two equally sized planes.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sse: u64 = a
        .iter()
        .zip(b)
        .map(|(&p, &q)| {
            let d = p as i64 - q as i64;
            (d * d) as u64
        })
        .sum();
    if sse == 0 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 * a.len() as f64 / sse as f64).log10()
}
