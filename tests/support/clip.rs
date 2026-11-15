//! Deterministic moving test clip, shared by the inter-effort tests, the
//! `inter_encode_clip` bench and the `inter_timing` example (each pulls
//! this file in with `#[path]`, next to `photo.rs`).
//!
//! The scene is the photo-like picture of `photo.rs` rendered on a canvas
//! twice the clip size and box-filtered 2×2 down to it, so motion comes
//! out in half-pixel steps and edges are band-limited the way a camera
//! delivers them:
//!
//! * the whole background pans by (1.5, 0.5) pixels per frame;
//! * three textured rectangles move over it with their own velocities
//!   (uncovering background behind them);
//! * every frame carries fresh ±1 grain.
//!
//! Integer-only, so every platform synthesises identical bytes.

use super::photo;

/// One I420 frame as tightly packed `(y, u, v)` planes.
pub type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// A rectangle of canvas texture travelling over the background: its
/// size, where its texture is taken from, where it starts and how far it
/// moves per frame (all in canvas pixels, i.e. half clip pixels).
struct Sprite {
    w: i64,
    h: i64,
    src_x: i64,
    src_y: i64,
    x0: i64,
    y0: i64,
    dx: i64,
    dy: i64,
}

fn hash3(x: u32, y: u32, t: u32) -> u32 {
    let mut h = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(t.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h
}

/// Synthesise `frames` frames of the `w × h` clip (`w`, `h` even).
pub fn moving_clip(w: usize, h: usize, frames: usize) -> Vec<Planes> {
    assert!(
        w.is_multiple_of(2) && h.is_multiple_of(2),
        "even clip dimensions"
    );
    // Canvas: twice the clip plus room for the pan.
    let margin = 4 * frames + 16;
    let (cw, ch) = (2 * w + margin, 2 * h + margin);
    let (cw, ch) = (cw + cw % 2, ch + ch % 2);
    let (cy, cu, cv) = photo::photo_like_i420(cw, ch);
    let (ccw, cch) = (cw / 2, ch / 2);

    let (cwi, chi) = (cw as i64, ch as i64);
    let sprites = [
        Sprite {
            w: cwi / 6,
            h: chi / 5,
            src_x: cwi / 2,
            src_y: chi / 8,
            x0: cwi / 8,
            y0: chi / 6,
            dx: 8,
            dy: 0,
        },
        Sprite {
            w: cwi / 8,
            h: chi / 4,
            src_x: cwi / 10,
            src_y: chi / 2,
            x0: cwi / 2,
            y0: chi / 3,
            dx: -5,
            dy: 2,
        },
        Sprite {
            w: cwi / 5,
            h: chi / 7,
            src_x: cwi / 3,
            src_y: (chi * 2) / 3,
            x0: cwi / 3,
            y0: (chi * 3) / 5,
            dx: 2,
            dy: -3,
        },
    ];

    // Canvas coordinate → texture coordinate at frame `t` (the sprite on
    // top wins; otherwise the panned background).
    let lookup = |x: i64, y: i64, t: i64| -> (usize, usize) {
        for s in &sprites {
            let (sx, sy) = (s.x0 + s.dx * t, s.y0 + s.dy * t);
            if x >= sx && x < sx + s.w && y >= sy && y < sy + s.h {
                let tx = (s.src_x + (x - sx)).clamp(0, cwi - 1);
                let ty = (s.src_y + (y - sy)).clamp(0, chi - 1);
                return (tx as usize, ty as usize);
            }
        }
        let tx = (x + 3 * t).clamp(0, cwi - 1);
        let ty = (y + t).clamp(0, chi - 1);
        (tx as usize, ty as usize)
    };

    (0..frames)
        .map(|t| {
            let ti = t as i64;
            let mut y = vec![0u8; w * h];
            let mut u = vec![0u8; (w / 2) * (h / 2)];
            let mut v = vec![0u8; (w / 2) * (h / 2)];
            for row in 0..h {
                for col in 0..w {
                    let mut sum = 0u32;
                    for (oy, ox) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (tx, ty) = lookup(2 * col as i64 + ox, 2 * row as i64 + oy, ti);
                        sum += cy[ty * cw + tx] as u32;
                    }
                    let grain = (hash3(col as u32, row as u32, t as u32) % 3) as i32 - 1;
                    y[row * w + col] = (((sum + 2) / 4) as i32 + grain).clamp(0, 255) as u8;
                }
            }
            for row in 0..h / 2 {
                for col in 0..w / 2 {
                    // One chroma sample covers a 4×4 canvas-luma patch,
                    // i.e. 2×2 canvas-chroma samples.
                    let (mut su, mut sv) = (0u32, 0u32);
                    for (oy, ox) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
                        let (tx, ty) = lookup(4 * col as i64 + ox, 4 * row as i64 + oy, ti);
                        let ci = (ty / 2).min(cch - 1) * ccw + (tx / 2).min(ccw - 1);
                        su += cu[ci] as u32;
                        sv += cv[ci] as u32;
                    }
                    u[row * (w / 2) + col] = ((su + 2) / 4) as u8;
                    v[row * (w / 2) + col] = ((sv + 2) / 4) as u8;
                }
            }
            (y, u, v)
        })
        .collect()
}

/// Read up to `frames` frames of a raw I420 file (`w × h`, frames back
/// to back) — lets the timing example run on an arbitrary clip.
#[allow(dead_code)]
pub fn read_i420_file(path: &str, w: usize, h: usize, frames: usize) -> Vec<Planes> {
    let data = std::fs::read(path).expect("readable raw I420 file");
    let (ys, cs) = (w * h, (w / 2) * (h / 2));
    data.chunks_exact(ys + 2 * cs)
        .take(frames)
        .map(|f| {
            (
                f[..ys].to_vec(),
                f[ys..ys + cs].to_vec(),
                f[ys + cs..].to_vec(),
            )
        })
        .collect()
}
