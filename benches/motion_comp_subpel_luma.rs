//! Micro-bench of the §18.3 luma sub-pel six-tap motion compensation.
//!
//! The 4×4 `filter_block_4x4` is the per-sub-block dispatcher; on a
//! sub-pel motion vector it pulls a 9×9 edge-replicated halo and runs
//! `sixtap_2d` (a horizontal pass of `cols=4, rows=9` followed by a
//! vertical pass of `cols=4, rows=4`). Per-MB this fires 16 times for
//! Y and 4 times for each chroma plane. The macroblock-scale variant
//! sums 16 calls on a 16×16 luma synthetic reference under a fixed
//! `(mx, my) = (3, 5)` quarter-pel vector so the §18.3 sub-pel path
//! is exercised (a whole-pel vector would short-circuit to the
//! `fetch_block_whole_pixel` copy path).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use oxideav_vp8::motion_comp::{
    fetch_block_whole_pixel, fetch_chroma_mb_halo, fetch_chroma_mb_whole_pixel, fetch_luma_mb_halo,
    fetch_luma_mb_whole_pixel, filter_block_4x4, filter_block_4x4_into, filter_set_for_version,
    sixtap_2d, sixtap_mb_chroma, sixtap_mb_luma, split_chroma_mvs, stored_luma_mv, FilterSet,
    ReferencePlanes,
};
use oxideav_vp8::motion_vector::Mv;
use oxideav_vp8::ReconstructedMb;

/// Build a 64×64 8-bit luma plane with a deterministic gradient
/// pattern. The plane is bigger than the MB so the §20.14 edge-replication
/// path stays cold (we measure the inner-block cost, not the clamp).
fn make_plane(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Mixed-frequency synthetic pattern — keeps the six-tap
            // partial sums non-trivial across the block.
            buf[r * w + c] = ((r * 7).wrapping_add(c * 11) ^ (r * c)) as u8;
        }
    }
    buf
}

fn bench_filter_block_4x4_subpel(c: &mut Criterion) {
    let w = 64;
    let h = 64;
    let plane = make_plane(w, h);
    // (mx, my) = (3, 5) ⇒ raw §17 vector with col & 7 = 3, row & 7 = 5.
    // Place the block well inside the plane so we cost the inner-block
    // path, not the clamp.
    let mv = Mv { row: 5, col: 3 };
    let filters = filter_set_for_version(0).taps();
    let mut g = c.benchmark_group("motion_comp_subpel_luma");
    g.bench_function("filter_block_4x4_sub3x5", |b| {
        b.iter(|| {
            let out = filter_block_4x4(
                black_box(&plane),
                w,
                w,
                h,
                /* blk_x = */ 24,
                /* blk_y = */ 24,
                black_box(mv),
                black_box(filters),
            );
            black_box(out)
        });
    });
    g.finish();
}

fn bench_mb_sixtap_2d(c: &mut Criterion) {
    // 16-sub-block "MB-scale" workload: do the 16 sixtap_2d calls a
    // single inter MB needs. Reuses the same prebuilt halo per inner
    // sub-block to isolate the convolution cost from the gather cost.
    let halo = {
        let plane = make_plane(64, 64);
        oxideav_vp8::motion_comp::fetch_block_halo(
            &plane,
            64,
            64,
            64,
            24,
            24,
            Mv { row: 5, col: 3 },
        )
    };
    let filters: &[[i32; 6]; 8] = match filter_set_for_version(0) {
        FilterSet::Sixtap => &oxideav_vp8::motion_comp::SIXTAP_FILTERS,
        FilterSet::Bilinear => &oxideav_vp8::motion_comp::BILINEAR_FILTERS,
    };
    let mut g = c.benchmark_group("motion_comp_subpel_luma");
    g.bench_function("mb_sixtap_2d_16x4x4", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for _ in 0..16 {
                let out = sixtap_2d(black_box(&halo), 3, 5, black_box(filters));
                acc = acc.wrapping_add(out[0] as u32);
            }
            black_box(acc)
        });
    });
    g.finish();
}

fn bench_mb_luma_batched(c: &mut Criterion) {
    // Round-270 MB-scale §18.3 batching: the whole 16×16 luma block of a
    // non-SPLITMV inter MB synthesised in one pass off a single 21×21
    // halo, versus the 16 separate `sixtap_2d` calls (each off its own
    // 9×9 halo) the per-sub-block path issues. Both produce byte-identical
    // output; this measures the amortised-fetch / wider-lane win.
    let plane = make_plane(64, 64);
    let mv = Mv { row: 5, col: 3 }; // (mx, my) = (3, 5) sub-pixel
    let filters: &[[i32; 6]; 8] = match filter_set_for_version(0) {
        FilterSet::Sixtap => &oxideav_vp8::motion_comp::SIXTAP_FILTERS,
        FilterSet::Bilinear => &oxideav_vp8::motion_comp::BILINEAR_FILTERS,
    };
    let mb_halo = fetch_luma_mb_halo(&plane, 64, 64, 64, 16, 16, mv);
    let sub_halo = oxideav_vp8::motion_comp::fetch_block_halo(&plane, 64, 64, 64, 24, 24, mv);

    let mut g = c.benchmark_group("motion_comp_subpel_luma");
    // Batched whole-MB synthesis: one 21×21 halo → one 16×16 luma block.
    g.bench_function("mb_luma_batched_16x16", |b| {
        b.iter(|| {
            let out = sixtap_mb_luma(black_box(&mb_halo), 3, 5, black_box(filters));
            black_box(out[0])
        });
    });
    // Single-axis fractions: the §18.3 whole-pixel filter row on the
    // other axis is the exact identity, so the r451 identity-pass
    // elision runs only one §20.14 pass. Same call shape as the (3, 5)
    // workload; the delta against it is the elided pass's cost. These
    // are the shapes the §17 half-pixel refinement probes four times
    // per center (axis-aligned candidates).
    g.bench_function("mb_luma_batched_16x16_horiz_only_3x0", |b| {
        b.iter(|| {
            let out = sixtap_mb_luma(black_box(&mb_halo), 3, 0, black_box(filters));
            black_box(out[0])
        });
    });
    g.bench_function("mb_luma_batched_16x16_vert_only_0x5", |b| {
        b.iter(|| {
            let out = sixtap_mb_luma(black_box(&mb_halo), 0, 5, black_box(filters));
            black_box(out[0])
        });
    });
    // Per-sub-block partner on the same workload: 16 sixtap_2d calls.
    g.bench_function("mb_luma_per_subblock_16x16", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for _ in 0..16 {
                let out = sixtap_2d(black_box(&sub_halo), 3, 5, black_box(filters));
                acc = acc.wrapping_add(out[0] as u32);
            }
            black_box(acc)
        });
    });
    g.finish();
}

fn bench_mb_chroma_batched(c: &mut Criterion) {
    // Round-271 MB-scale §18.3 chroma batching: the whole 8×8 chroma block
    // of a non-SPLITMV inter MB synthesised in one pass off a single 13×13
    // halo, versus the 4 separate `sixtap_2d` calls (each off its own 9×9
    // halo) the per-sub-block path issues. Both produce byte-identical
    // output; this measures the amortised-fetch / wider-lane win.
    let plane = make_plane(64, 64);
    let mv = Mv { row: 5, col: 3 }; // (mx, my) = (3, 5) sub-pixel
    let filters: &[[i32; 6]; 8] = match filter_set_for_version(0) {
        FilterSet::Sixtap => &oxideav_vp8::motion_comp::SIXTAP_FILTERS,
        FilterSet::Bilinear => &oxideav_vp8::motion_comp::BILINEAR_FILTERS,
    };
    let mb_halo = fetch_chroma_mb_halo(&plane, 64, 64, 64, 16, 16, mv);
    let sub_halo = oxideav_vp8::motion_comp::fetch_block_halo(&plane, 64, 64, 64, 20, 20, mv);

    let mut g = c.benchmark_group("motion_comp_subpel_luma");
    // Batched whole-MB synthesis: one 13×13 halo → one 8×8 chroma block.
    g.bench_function("mb_chroma_batched_8x8", |b| {
        b.iter(|| {
            let out = sixtap_mb_chroma(black_box(&mb_halo), 3, 5, black_box(filters));
            black_box(out[0])
        });
    });
    // Per-sub-block partner on the same workload: 4 sixtap_2d calls.
    g.bench_function("mb_chroma_per_subblock_8x8", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for _ in 0..4 {
                let out = sixtap_2d(black_box(&sub_halo), 3, 5, black_box(filters));
                acc = acc.wrapping_add(out[0] as u32);
            }
            black_box(acc)
        });
    });
    g.finish();
}

fn bench_mb_whole_pixel_batched(c: &mut Criterion) {
    // Round-272 whole-pixel non-SPLITMV MB batching: when the shared §18.1
    // vector is whole-pixel, the §18.3 prediction is a pure copy (no
    // convolution). The whole 16×16 luma / 8×8 chroma block is one
    // contiguous source region, fetched in one pass
    // (`fetch_luma_mb_whole_pixel` / `fetch_chroma_mb_whole_pixel`) instead
    // of sixteen / four 4×4 `fetch_block_whole_pixel` copies. Both produce
    // byte-identical output; this measures the gather amortisation.
    let plane = make_plane(64, 64);
    // Whole-pixel vector: integer offset (1, 2), no fractional bits.
    let mv = Mv { row: 8, col: 16 };

    let mut g = c.benchmark_group("motion_comp_subpel_luma");

    // Batched whole-MB luma fetch: one 16×16 contiguous copy.
    g.bench_function("mb_luma_whole_pixel_batched_16x16", |b| {
        b.iter(|| {
            let out =
                fetch_luma_mb_whole_pixel(black_box(&plane), 64, 64, 64, 16, 16, black_box(mv));
            black_box(out[0])
        });
    });
    // Per-sub-block partner: sixteen 4×4 `fetch_block_whole_pixel` copies
    // assembled into a 16×16 block (the pre-round-272 luma whole-pixel
    // path).
    g.bench_function("mb_luma_whole_pixel_per_subblock_16x16", |b| {
        b.iter(|| {
            let mut out = [0u8; 256];
            for sb in 0..4 {
                for sc in 0..4 {
                    let blk = fetch_block_whole_pixel(
                        black_box(&plane),
                        64,
                        64,
                        64,
                        16 + sc * 4,
                        16 + sb * 4,
                        black_box(mv),
                    );
                    for r in 0..4 {
                        let dst = (sb * 4 + r) * 16 + sc * 4;
                        out[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
                    }
                }
            }
            black_box(out[0])
        });
    });

    // Batched whole-MB chroma fetch: one 8×8 contiguous copy.
    g.bench_function("mb_chroma_whole_pixel_batched_8x8", |b| {
        b.iter(|| {
            let out =
                fetch_chroma_mb_whole_pixel(black_box(&plane), 64, 64, 64, 8, 8, black_box(mv));
            black_box(out[0])
        });
    });
    // Per-sub-block partner: four 4×4 `fetch_block_whole_pixel` copies.
    g.bench_function("mb_chroma_whole_pixel_per_subblock_8x8", |b| {
        b.iter(|| {
            let mut out = [0u8; 64];
            for sb in 0..2 {
                for sc in 0..2 {
                    let blk = fetch_block_whole_pixel(
                        black_box(&plane),
                        64,
                        64,
                        64,
                        8 + sc * 4,
                        8 + sb * 4,
                        black_box(mv),
                    );
                    for r in 0..4 {
                        let dst = (sb * 4 + r) * 8 + sc * 4;
                        out[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
                    }
                }
            }
            black_box(out[0])
        });
    });
    g.finish();
}

/// Round-274 SPLITMV write-strategy A/B. SPLITMV sub-blocks carry sixteen
/// distinct luma vectors (plus four chroma), so the MB-scale shared-halo
/// batch (rounds 270–272) cannot apply — the only freedom left is *how* the
/// per-sub-block synthesis lands in the MB raster. This bench measures two
/// strategies that produce byte-identical output:
///
/// * **scratch_copy** — `filter_block_4x4` builds a contiguous `[u8; 16]`
///   block, then four contiguous 4-byte rows are copied into the stride-16
///   / stride-8 raster (the form `predict_split_mv` ships).
/// * **strided_write** — `filter_block_4x4_into` writes each synthesised
///   sub-block directly into the destination at its strided offset, with no
///   intermediate `[u8; 16]`.
///
/// Counter-intuitively the scratch_copy form wins (~+23 % on Apple M4):
/// the contiguous block lets the compiler vectorise the per-row writes,
/// where the scattered strided writes into a stride-16 raster can't. This
/// is the measured reason `predict_split_mv` keeps the scratch path and the
/// long-standing "SPLITMV sub-block batching" candidate is closed negative.
fn bench_splitmv_predict(c: &mut Criterion) {
    // 3×3 MB grid so the centre MB's per-sub-block reads stay in-bounds.
    let lw = 48;
    let lh = 48;
    let cw = 24;
    let ch = 24;
    let y = make_plane(lw, lh);
    let u = make_plane(cw, ch);
    let v = make_plane(cw, ch);
    let reference = ReferencePlanes {
        y: &y,
        u: &u,
        v: &v,
        y_stride: lw,
        uv_stride: cw,
        mb_cols: 3,
        mb_rows: 3,
    };
    let filters = filter_set_for_version(0).taps();

    // Sixteen distinct vectors mixing whole-pixel and sub-pixel fractions.
    let mut mvs = [Mv { row: 0, col: 0 }; 16];
    for (i, m) in mvs.iter_mut().enumerate() {
        let frac_r = if i % 2 == 0 { 0 } else { (i as i16 % 7) + 1 };
        let frac_c = if i % 3 == 0 { 0 } else { (i as i16 % 5) + 1 };
        *m = Mv {
            row: ((i as i16 % 3) - 1) * 8 + frac_r,
            col: ((i as i16 % 4) - 2) * 8 + frac_c,
        };
    }

    let mut g = c.benchmark_group("motion_comp_subpel_luma");

    // Strided-write strategy: `filter_block_4x4_into` writes each sub-block
    // directly into the raster (no intermediate `[u8; 16]`).
    g.bench_function("splitmv_predict_strided_write", |b| {
        b.iter(|| {
            let mvs = black_box(&mvs);
            let mut out = ReconstructedMb::default();
            for sb in 0..4 {
                for sc in 0..4 {
                    let ymv = stored_luma_mv(mvs[sb * 4 + sc]);
                    filter_block_4x4_into(
                        &mut out.y,
                        16,
                        sc * 4,
                        sb * 4,
                        reference.y,
                        reference.y_stride,
                        lw,
                        lh,
                        16 + sc * 4,
                        16 + sb * 4,
                        ymv,
                        filters,
                    );
                }
            }
            let chroma = split_chroma_mvs(mvs);
            for sb in 0..2 {
                for sc in 0..2 {
                    let uvmv = chroma[sb * 2 + sc];
                    filter_block_4x4_into(
                        &mut out.u,
                        8,
                        sc * 4,
                        sb * 4,
                        reference.u,
                        reference.uv_stride,
                        cw,
                        ch,
                        8 + sc * 4,
                        8 + sb * 4,
                        uvmv,
                        filters,
                    );
                    filter_block_4x4_into(
                        &mut out.v,
                        8,
                        sc * 4,
                        sb * 4,
                        reference.v,
                        reference.uv_stride,
                        cw,
                        ch,
                        8 + sc * 4,
                        8 + sb * 4,
                        uvmv,
                        filters,
                    );
                }
            }
            black_box(out.y[0])
        });
    });

    // Scratch-copy strategy (the form `predict_split_mv` ships):
    // `filter_block_4x4` into a `[u8; 16]` scratch +
    // four-row strided copy into the MB raster, for every sub-block.
    g.bench_function("splitmv_predict_scratch_copy", |b| {
        b.iter(|| {
            let mvs = black_box(&mvs);
            let mut out = ReconstructedMb::default();
            for sb in 0..4 {
                for sc in 0..4 {
                    let ymv = stored_luma_mv(mvs[sb * 4 + sc]);
                    let blk = filter_block_4x4(
                        reference.y,
                        reference.y_stride,
                        lw,
                        lh,
                        16 + sc * 4,
                        16 + sb * 4,
                        ymv,
                        filters,
                    );
                    for r in 0..4 {
                        let dst = (sb * 4 + r) * 16 + sc * 4;
                        out.y[dst..dst + 4].copy_from_slice(&blk[r * 4..r * 4 + 4]);
                    }
                }
            }
            let chroma = split_chroma_mvs(mvs);
            for sb in 0..2 {
                for sc in 0..2 {
                    let uvmv = chroma[sb * 2 + sc];
                    let ublk = filter_block_4x4(
                        reference.u,
                        reference.uv_stride,
                        cw,
                        ch,
                        8 + sc * 4,
                        8 + sb * 4,
                        uvmv,
                        filters,
                    );
                    let vblk = filter_block_4x4(
                        reference.v,
                        reference.uv_stride,
                        cw,
                        ch,
                        8 + sc * 4,
                        8 + sb * 4,
                        uvmv,
                        filters,
                    );
                    for r in 0..4 {
                        let dst = (sb * 4 + r) * 8 + sc * 4;
                        out.u[dst..dst + 4].copy_from_slice(&ublk[r * 4..r * 4 + 4]);
                        out.v[dst..dst + 4].copy_from_slice(&vblk[r * 4..r * 4 + 4]);
                    }
                }
            }
            black_box(out.y[0])
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_filter_block_4x4_subpel,
    bench_mb_sixtap_2d,
    bench_mb_luma_batched,
    bench_mb_chroma_batched,
    bench_mb_whole_pixel_batched,
    bench_splitmv_predict
);
criterion_main!(benches);
