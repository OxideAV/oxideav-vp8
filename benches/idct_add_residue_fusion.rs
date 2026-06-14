//! Round 286 — A/B micro-bench of the §14.4 IDCT + §14.5 add-clamp
//! residue pass, isolating the fusion landed this round.
//!
//! The r285 bench round ranked `reconstruct_inter_mb` the #1 decoder
//! self-time symbol (≈24 %), with the §14.4/§14.5 residue pass — 24
//! per-MB inverse DCTs whose output is summed with the strided
//! prediction raster — the dominant share. Round 286 fuses the IDCT
//! second pass with the add-clamp store directly into the strided
//! plane (`inverse_dct_4x4_add_into`), removing the prior four-buffer
//! round-trip (`inverse_dct_4x4` → extract 4×4 → `add_residue_4x4` →
//! insert 4×4).
//!
//! This harness times BOTH shapes in the SAME binary and SAME run, so
//! the A/B is immune to the cross-run machine-load drift that
//! contaminates a before/after of `reconstruct_inter_mb` itself when
//! the box is shared. Each iteration performs the full 24-sub-block MB
//! residue pass (16 luma at stride 16 + 4 U + 4 V at stride 8) over a
//! fresh predictor copy.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::inverse_transform::{add_residue_4x4, inverse_dct_4x4, inverse_dct_4x4_add_into};

/// Deterministic dequantized-magnitude coefficient block (the shape
/// §14.1 dequantization produces from small tokens at mid quantisers).
fn coeff_block(seed: i16) -> [i16; 16] {
    let mut out = [0i16; 16];
    out[0] = 40 + (seed % 7) * 9;
    out[1] = -12 + seed % 5;
    out[4] = 17 - seed % 3;
    out[5] = 6 + seed % 4;
    out[2] = -(4 + seed % 2);
    out
}

/// A DC-only block — every AC coefficient zero, only `out[0]` set. This
/// is the residue shape `inverse_dct_4x4_add_into` short-circuits
/// through its §14.4 DC-only fast path (`(input[0] + 4) >> 3` added
/// uniformly, no butterfly), and the dominant residue shape a real
/// decode sees: a well-predicted sub-block contributes only its DC
/// correction.
fn dc_only_block(seed: i16) -> [i16; 16] {
    let mut out = [0i16; 16];
    out[0] = 40 + (seed % 7) * 9;
    out
}

fn make_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            buf[r * w + c] = ((r as u32 * 7)
                .wrapping_add(c as u32 * 11)
                .wrapping_add(seed.wrapping_mul(13))
                ^ (r as u32 * c as u32)) as u8;
        }
    }
    buf
}

#[inline]
fn extract_4x4(plane: &[u8], stride: usize, sr: usize, sc: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let (y0, x0) = (sr * 4, sc * 4);
    for r in 0..4 {
        let src = (y0 + r) * stride + x0;
        out[r * 4..r * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    out
}

#[inline]
fn insert_4x4(plane: &mut [u8], stride: usize, sr: usize, sc: usize, src: &[u8; 16]) {
    let (y0, x0) = (sr * 4, sc * 4);
    for r in 0..4 {
        let dst = (y0 + r) * stride + x0;
        plane[dst..dst + 4].copy_from_slice(&src[r * 4..r * 4 + 4]);
    }
}

/// The prior four-buffer sequence, one MB worth (24 sub-blocks).
fn unfused_mb(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    yc: &[[i16; 16]; 16],
    uc: &[[i16; 16]; 4],
    vc: &[[i16; 16]; 4],
) {
    for i in 0..4 {
        for j in 0..4 {
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&yc[i * 4 + j], &mut residue);
            let pred = extract_4x4(y, 16, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(y, 16, i, j, &summed);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            let mut residue = [0i16; 16];
            inverse_dct_4x4(&uc[idx], &mut residue);
            let pred = extract_4x4(u, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(u, 8, i, j, &summed);

            let mut residue = [0i16; 16];
            inverse_dct_4x4(&vc[idx], &mut residue);
            let pred = extract_4x4(v, 8, i, j);
            let mut summed = [0u8; 16];
            add_residue_4x4(&pred, &residue, &mut summed);
            insert_4x4(v, 8, i, j, &summed);
        }
    }
}

/// The round-286 fused helper, one MB worth (24 sub-blocks).
fn fused_mb(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    yc: &[[i16; 16]; 16],
    uc: &[[i16; 16]; 4],
    vc: &[[i16; 16]; 4],
) {
    for i in 0..4 {
        for j in 0..4 {
            inverse_dct_4x4_add_into(&yc[i * 4 + j], y, 16, i, j);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            inverse_dct_4x4_add_into(&uc[idx], u, 8, i, j);
            inverse_dct_4x4_add_into(&vc[idx], v, 8, i, j);
        }
    }
}

fn bench_fusion(c: &mut Criterion) {
    let yc: [[i16; 16]; 16] = core::array::from_fn(|i| coeff_block(2 + i as i16));
    let uc: [[i16; 16]; 4] = core::array::from_fn(|i| coeff_block(20 + i as i16));
    let vc: [[i16; 16]; 4] = core::array::from_fn(|i| coeff_block(30 + i as i16));
    let y0 = make_plane(16, 16, 2);
    let u0 = make_plane(8, 8, 3);
    let v0 = make_plane(8, 8, 4);

    let mut g = c.benchmark_group("idct_add_residue_fusion");

    g.bench_function("unfused_mb", |b| {
        b.iter(|| {
            let mut y = y0.clone();
            let mut u = u0.clone();
            let mut v = v0.clone();
            unfused_mb(&mut y, &mut u, &mut v, &yc, &uc, &vc);
            black_box(&y)[0]
        });
    });

    g.bench_function("fused_mb", |b| {
        b.iter(|| {
            let mut y = y0.clone();
            let mut u = u0.clone();
            let mut v = v0.clone();
            fused_mb(&mut y, &mut u, &mut v, &yc, &uc, &vc);
            black_box(&y)[0]
        });
    });

    g.finish();
}

/// Round 297 — DC-only A/B. A whole-keyframe decode profile (`sample`
/// over a tight `decode_vp8` loop on a 320×240 qi=32 keyframe) ranked
/// `inverse_dct_4x4_scalar` the #1 top-of-stack symbol of the decode
/// path. The existing `bench_fusion` above measures the full
/// (high-AC) transform; this group isolates the DC-only residue shape,
/// which is what well-predicted sub-blocks actually produce and what
/// `inverse_dct_4x4_add_into`'s §14.4 DC-only fast path
/// short-circuits. The fast path's `(input[0]+4)>>3` uniform add
/// should be far cheaper than running the full two-pass butterfly, so
/// this A/B exposes the speedup the fast path already buys and gives a
/// stable target for any future DC-path tuning. Both shapes run in the
/// SAME binary so the comparison is immune to cross-run machine-load
/// drift on a shared box.
fn bench_dc_only(c: &mut Criterion) {
    // DC-only coefficient blocks — only out[0] non-zero.
    let yc: [[i16; 16]; 16] = core::array::from_fn(|i| dc_only_block(2 + i as i16));
    let uc: [[i16; 16]; 4] = core::array::from_fn(|i| dc_only_block(20 + i as i16));
    let vc: [[i16; 16]; 4] = core::array::from_fn(|i| dc_only_block(30 + i as i16));
    // Full (high-AC) blocks for the contrasting cell.
    let yc_ac: [[i16; 16]; 16] = core::array::from_fn(|i| coeff_block(2 + i as i16));
    let uc_ac: [[i16; 16]; 4] = core::array::from_fn(|i| coeff_block(20 + i as i16));
    let vc_ac: [[i16; 16]; 4] = core::array::from_fn(|i| coeff_block(30 + i as i16));
    let y0 = make_plane(16, 16, 2);
    let u0 = make_plane(8, 8, 3);
    let v0 = make_plane(8, 8, 4);

    let mut g = c.benchmark_group("idct_dc_only");

    // Fast path: 24 DC-only sub-blocks per MB through `add_into`.
    g.bench_function("dc_only_mb", |b| {
        b.iter(|| {
            let mut y = y0.clone();
            let mut u = u0.clone();
            let mut v = v0.clone();
            fused_mb(&mut y, &mut u, &mut v, &yc, &uc, &vc);
            black_box(&y)[0]
        });
    });

    // Contrast: the same 24 sub-blocks but with AC present, so each one
    // runs the full two-pass §14.4 butterfly.
    g.bench_function("full_ac_mb", |b| {
        b.iter(|| {
            let mut y = y0.clone();
            let mut u = u0.clone();
            let mut v = v0.clone();
            fused_mb(&mut y, &mut u, &mut v, &yc_ac, &uc_ac, &vc_ac);
            black_box(&y)[0]
        });
    });

    g.finish();
}

criterion_group!(benches, bench_fusion, bench_dc_only);
criterion_main!(benches);
