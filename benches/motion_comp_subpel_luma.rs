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
    fetch_chroma_mb_halo, fetch_luma_mb_halo, filter_block_4x4, filter_set_for_version, sixtap_2d,
    sixtap_mb_chroma, sixtap_mb_luma, FilterSet,
};
use oxideav_vp8::motion_vector::Mv;

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

criterion_group!(
    benches,
    bench_filter_block_4x4_subpel,
    bench_mb_sixtap_2d,
    bench_mb_luma_batched,
    bench_mb_chroma_batched
);
criterion_main!(benches);
