//! Whole-frame §15 loop-filter bench — `filter_frame` (key-frame path)
//! and `filter_inter_frame` (§15.1 inter path) over a 320×240 frame
//! (20×15 macroblocks). The round-170 `loop_filter_normal` and
//! round-260 `loop_filter_mb_edge` benches measure the per-edge
//! primitives in isolation; this bench measures the full §20.6 frame
//! pass that the decoder runs once per frame — per-MB level resolution,
//! the §15.1 skip rules, and the raster-ordered MB-edge + sub-block
//! edge cascade across all three planes. Every macroblock carries a
//! coded coefficient so steps 2 and 4 (the interior sub-block edges)
//! run on every MB — the fully-coded worst case, matching what a
//! mid-quantiser keyframe or dense P-frame produces.
//!
//! The filter mutates the planes in place, so each measured iteration
//! runs on a fresh clone via `iter_batched` (the clone is unmeasured).

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use oxideav_vp8::loop_filter::filter_inter_frame;
use oxideav_vp8::{
    filter_frame, FrameFilterConfig, InterMode, IntraUvMode, IntraYMode, KeyframePlanes,
    MacroblockModes, MbCoeffs, RefFrame,
};

const MB_COLS: usize = 20;
const MB_ROWS: usize = 15;

/// Build a 320×240 reconstructed-frame stand-in: a luma gradient with a
/// per-MB checkerboard DC offset so every MB edge carries a real
/// discontinuity for the filters to act on (a flat plane would still
/// pay the full mask arithmetic but never write).
fn synthesise_planes() -> KeyframePlanes {
    let ys = MB_COLS * 16;
    let cs = MB_COLS * 8;
    let yh = MB_ROWS * 16;
    let ch = MB_ROWS * 8;

    let mut y = vec![0u8; ys * yh];
    for row in 0..yh {
        for col in 0..ys {
            let base = ((col * 256 / ys + row * 256 / yh) / 2) as i32;
            let offset = if ((row / 16) + (col / 16)) % 2 == 0 {
                6
            } else {
                -6
            };
            y[row * ys + col] = (base + offset).clamp(0, 255) as u8;
        }
    }
    let mut u = vec![0u8; cs * ch];
    let mut v = vec![0u8; cs * ch];
    for row in 0..ch {
        for col in 0..cs {
            let offset = if ((row / 8) + (col / 8)) % 2 == 0 {
                4
            } else {
                -4
            };
            u[row * cs + col] = (120 + (col * 16 / cs) as i32 + offset).clamp(0, 255) as u8;
            v[row * cs + col] = (130 + (row * 16 / ch) as i32 + offset).clamp(0, 255) as u8;
        }
    }
    KeyframePlanes {
        y,
        u,
        v,
        y_stride: ys,
        uv_stride: cs,
        mb_cols: MB_COLS,
        mb_rows: MB_ROWS,
    }
}

/// Per-MB records: DC-mode, not skipped, one coded AC coefficient so
/// the §15.1 interior sub-block steps run on every macroblock.
fn synthesise_mb_records() -> (Vec<MacroblockModes>, Vec<MbCoeffs>) {
    let n = MB_COLS * MB_ROWS;
    let modes = vec![
        MacroblockModes {
            segment_id: None,
            mb_skip_coeff: false,
            y_mode: IntraYMode::Dc,
            subblock_modes: None,
            uv_mode: IntraUvMode::Dc,
        };
        n
    ];
    let mut coeffs = vec![MbCoeffs::default(); n];
    for c in &mut coeffs {
        c.y[0][1] = 7; // any non-zero coefficient marks the MB as coded
    }
    (modes, coeffs)
}

fn config(simple: bool, key_frame: bool) -> FrameFilterConfig {
    FrameFilterConfig {
        simple,
        key_frame,
        loop_filter_level: 26,
        sharpness_level: 0,
        segmentation_enabled: false,
        segment_abs: false,
        segment_lf_level: [0; 4],
        delta_enabled: false,
        ref_delta_current: 0,
        bpred_mode_delta: 0,
        ref_delta_last: 0,
        ref_delta_golden: 0,
        ref_delta_altref: 0,
        zero_mv_mode_delta: 0,
        other_mv_mode_delta: 0,
        split_mv_mode_delta: 0,
    }
}

fn bench_loop_filter_frame(c: &mut Criterion) {
    let planes = synthesise_planes();
    let (modes, coeffs) = synthesise_mb_records();
    let n = MB_COLS * MB_ROWS;
    // Inter pass: every MB predicts NEWMV from LAST, the dense-P shape.
    let ref_frames = vec![Some(RefFrame::Last); n];
    let inter_modes = vec![Some(InterMode::New); n];

    let mut g = c.benchmark_group("loop_filter_frame");
    g.throughput(Throughput::Elements((MB_COLS * 16 * MB_ROWS * 16) as u64));
    g.sample_size(50);

    let cfg = config(false, true);
    g.bench_function("filter_frame_keyframe_320x240_normal", |b| {
        b.iter_batched(
            || planes.clone(),
            |mut p| {
                filter_frame(&mut p, black_box(&modes), black_box(&coeffs), &cfg);
                black_box(p.y[0])
            },
            BatchSize::LargeInput,
        );
    });

    let cfg_simple = config(true, true);
    g.bench_function("filter_frame_keyframe_320x240_simple", |b| {
        b.iter_batched(
            || planes.clone(),
            |mut p| {
                filter_frame(&mut p, black_box(&modes), black_box(&coeffs), &cfg_simple);
                black_box(p.y[0])
            },
            BatchSize::LargeInput,
        );
    });

    let cfg_inter = config(false, false);
    g.bench_function("filter_inter_frame_320x240_normal", |b| {
        b.iter_batched(
            || planes.clone(),
            |mut p| {
                filter_inter_frame(
                    &mut p,
                    black_box(&modes),
                    black_box(&coeffs),
                    black_box(&ref_frames),
                    black_box(&inter_modes),
                    &cfg_inter,
                );
                black_box(p.y[0])
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_loop_filter_frame);
criterion_main!(benches);
