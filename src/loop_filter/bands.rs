//! The §15 loop filter over a whole frame on several threads.
//!
//! §20.6 filters macroblocks in raster order and, inside a macroblock,
//! in a fixed order: left edge, inner vertical edges, top edge, inner
//! horizontal edges. Macroblock `(r, c)` reads up to four and rewrites up
//! to three pixels on the far side of its left and top edges, so it must
//! run after `(r, c − 1)` and `(r − 1, c)` — and also after `(r − 1, c + 1)`,
//! whose left-edge filter rewrites the rightmost columns of `(r − 1, c)`
//! all the way down to the rows `(r, c)`'s top edge touches. Conversely
//! nothing `(r, c)` writes is read again by a macroblock that came before
//! it. So macroblock rows can run concurrently as long as row `r` stays
//! two macroblocks behind row `r − 1`.
//!
//! What two adjacent rows share is the four pixel rows above the edge
//! between them. Each row job therefore filters in a buffer of its own —
//! four context rows on top of its sixteen (chroma: four on top of
//! eight) — and exchanges only those four rows with its neighbours
//! through a small shared buffer per row boundary:
//!
//! * a job loads its own bottom four rows unfiltered and, as it finishes
//!   each macroblock, publishes that macroblock's columns (the last
//!   three become final only when the next macroblock's left edge has
//!   run, which republishes them);
//! * the job below copies the sixteen columns it needs from there before
//!   filtering each macroblock (once the row above is two macroblocks
//!   further, so that they are final),
//!   filters across the boundary in its own buffer, and at the end writes
//!   those four rows — now final — back into the picture together with
//!   its own first twelve.
//!
//! The picture is cut into disjoint strips to match (a job owns the four
//! rows above its macroblock row and the first twelve of its own), so no
//! pixel is written by two jobs. Every macroblock sees exactly the pixel
//! values the raster-order pass would have shown it, so the output is
//! identical to the serial filter at every thread count
//! (`banded_filter_matches_serial`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::backoff::SharedBytes;

use super::{
    LoopFilterParams, filter_h_edge_normal, filter_h_edge_simple, filter_v_edge_normal,
    filter_v_edge_simple,
};
use crate::frame::KeyframePlanes;

/// How one macroblock is filtered: its §15.4 limits and whether its
/// inner (sub-block) edges are filtered. `None` in a spec list means the
/// macroblock's level is 0 and it is left alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MbFilterSpec {
    pub(crate) params: LoopFilterParams,
    pub(crate) filter_subblocks: bool,
}

/// Filter one macroblock inside (possibly partial) plane buffers: the
/// macroblock's luma origin is at `(y_x0, y_y0)` of `y` (stride `ys`),
/// its chroma origin at `(uv_x0, uv_y0)` of `u` / `v` (stride `cs`). The
/// §20.6 step order of the frame filters.
#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_mb(
    y: &mut [u8],
    ys: usize,
    u: &mut [u8],
    v: &mut [u8],
    cs: usize,
    (y_x0, y_y0): (usize, usize),
    (uv_x0, uv_y0): (usize, usize),
    spec: &MbFilterSpec,
    simple: bool,
    has_left: bool,
    has_top: bool,
) {
    let params = &spec.params;
    if simple {
        if has_left {
            filter_v_edge_simple(y, ys, y_x0, y_y0, params.mbedge_limit);
        }
        if spec.filter_subblocks {
            for q in 1..4 {
                filter_v_edge_simple(y, ys, y_x0 + 4 * q, y_y0, params.sub_bedge_limit);
            }
        }
        if has_top {
            filter_h_edge_simple(y, ys, y_y0, y_x0, params.mbedge_limit);
        }
        if spec.filter_subblocks {
            for q in 1..4 {
                filter_h_edge_simple(y, ys, y_y0 + 4 * q, y_x0, params.sub_bedge_limit);
            }
        }
        return;
    }
    if has_left {
        filter_v_edge_normal(y, ys, y_x0, y_y0, 16, params, true);
        filter_v_edge_normal(u, cs, uv_x0, uv_y0, 8, params, true);
        filter_v_edge_normal(v, cs, uv_x0, uv_y0, 8, params, true);
    }
    if spec.filter_subblocks {
        for q in 1..4 {
            filter_v_edge_normal(y, ys, y_x0 + 4 * q, y_y0, 16, params, false);
        }
        filter_v_edge_normal(u, cs, uv_x0 + 4, uv_y0, 8, params, false);
        filter_v_edge_normal(v, cs, uv_x0 + 4, uv_y0, 8, params, false);
    }
    if has_top {
        filter_h_edge_normal(y, ys, y_y0, y_x0, 16, params, true);
        filter_h_edge_normal(u, cs, uv_y0, uv_x0, 8, params, true);
        filter_h_edge_normal(v, cs, uv_y0, uv_x0, 8, params, true);
    }
    if spec.filter_subblocks {
        for q in 1..4 {
            filter_h_edge_normal(y, ys, y_y0 + 4 * q, y_x0, 16, params, false);
        }
        filter_h_edge_normal(u, cs, uv_y0 + 4, uv_x0, 8, params, false);
        filter_h_edge_normal(v, cs, uv_y0 + 4, uv_x0, 8, params, false);
    }
}

/// Filter a whole frame in raster order on the calling thread.
pub(crate) fn filter_serial(
    planes: &mut KeyframePlanes,
    specs: &[Option<MbFilterSpec>],
    simple: bool,
) {
    let (mb_cols, mb_rows) = (planes.mb_cols, planes.mb_rows);
    let (ys, cs) = (planes.y_stride, planes.uv_stride);
    for mb_row in 0..mb_rows {
        for mb_col in 0..mb_cols {
            if let Some(spec) = &specs[mb_row * mb_cols + mb_col] {
                filter_mb(
                    &mut planes.y,
                    ys,
                    &mut planes.u,
                    &mut planes.v,
                    cs,
                    (mb_col * 16, mb_row * 16),
                    (mb_col * 8, mb_row * 8),
                    spec,
                    simple,
                    mb_col > 0,
                    mb_row > 0,
                );
            }
        }
    }
}

/// The four pixel rows above a row boundary (luma, and chroma's four),
/// as the upper row's job leaves them. Lock-free: the two rows touch it
/// once per macroblock each (see [`SharedBytes`]).
struct Boundary {
    y: SharedBytes,
    u: SharedBytes,
    v: SharedBytes,
}

struct RowShared {
    boundary: Boundary,
    /// Macroblocks of the upper row that are filtered. The boundary
    /// columns of macroblock `c` are final for the upper row once
    /// `done >= min(c + 2, mb_cols)`: the next macroblock's left edge
    /// rewrites the last three columns of `c`.
    done: AtomicUsize,
}

/// One macroblock row of work with the picture strips it owns: the
/// boundary rows above it (absent for the top row), the first twelve
/// luma / four chroma rows of its own, and — for the bottom row — its
/// last four.
struct RowJob<'p> {
    mb_row: usize,
    above: Option<[&'p mut [u8]; 3]>,
    body: [&'p mut [u8]; 3],
    tail: Option<[&'p mut [u8]; 3]>,
}

struct AbortOnUnwind<'a>(&'a AtomicBool);

impl Drop for AbortOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One macroblock row's strips of a plane: the boundary rows above it,
/// its body, and (bottom row only) its tail.
type PlaneStrips<'p> = (Option<&'p mut [u8]>, &'p mut [u8], Option<&'p mut [u8]>);

/// Cut one plane into the strips the row jobs own: per macroblock row
/// `(boundary rows above, body, tail)`. `mb_h` is the macroblock height
/// in this plane (16 luma, 8 chroma); the boundary is always four rows.
fn cut_plane(plane: &mut [u8], stride: usize, mb_rows: usize, mb_h: usize) -> Vec<PlaneStrips<'_>> {
    let body_rows = mb_h - 4;
    let mut out = Vec::with_capacity(mb_rows);
    let mut rest = plane;
    for mb_row in 0..mb_rows {
        let above = if mb_row == 0 {
            None
        } else {
            let (head, tail) = rest.split_at_mut(4 * stride);
            rest = tail;
            Some(head)
        };
        let (body, tail) = rest.split_at_mut(body_rows * stride);
        rest = tail;
        let last = if mb_row + 1 == mb_rows {
            let (head, tail) = rest.split_at_mut(4 * stride);
            rest = tail;
            Some(head)
        } else {
            None
        };
        out.push((above, body, last));
    }
    out
}

/// Filter a whole frame on up to `threads` threads; identical output to
/// [`filter_serial`]. `specs` holds one entry per macroblock in raster
/// order.
pub(crate) fn filter_banded(
    planes: &mut KeyframePlanes,
    specs: &[Option<MbFilterSpec>],
    simple: bool,
    threads: usize,
) {
    let (mb_cols, mb_rows) = (planes.mb_cols, planes.mb_rows);
    let workers = threads.clamp(1, mb_rows.max(1));
    if workers == 1 || mb_cols == 0 {
        filter_serial(planes, specs, simple);
        return;
    }
    let (ys, cs) = (planes.y_stride, planes.uv_stride);

    // Every row's unfiltered bottom four rows, before the picture is cut
    // up.
    let shared: Vec<RowShared> = (0..mb_rows)
        .map(|r| {
            let y0 = (r * 16 + 12) * ys;
            let c0 = (r * 8 + 4) * cs;
            RowShared {
                boundary: Boundary {
                    y: SharedBytes::from_bytes(&planes.y[y0..y0 + 4 * ys]),
                    u: SharedBytes::from_bytes(&planes.u[c0..c0 + 4 * cs]),
                    v: SharedBytes::from_bytes(&planes.v[c0..c0 + 4 * cs]),
                },
                done: AtomicUsize::new(0),
            }
        })
        .collect();

    let jobs: Vec<RowJob> = cut_plane(&mut planes.y, ys, mb_rows, 16)
        .into_iter()
        .zip(cut_plane(&mut planes.u, cs, mb_rows, 8))
        .zip(cut_plane(&mut planes.v, cs, mb_rows, 8))
        .enumerate()
        .map(|(mb_row, ((y, u), v))| RowJob {
            mb_row,
            above: match (y.0, u.0, v.0) {
                (Some(a), Some(b), Some(c)) => Some([a, b, c]),
                _ => None,
            },
            body: [y.1, u.1, v.1],
            tail: match (y.2, u.2, v.2) {
                (Some(a), Some(b), Some(c)) => Some([a, b, c]),
                _ => None,
            },
        })
        .collect();
    let jobs = Mutex::new(jobs.into_iter());
    let abort = AtomicBool::new(false);

    let worker = || {
        let _guard = AbortOnUnwind(&abort);
        // Work buffers: four context rows + one macroblock row.
        let mut buf_y = vec![0u8; 20 * ys];
        let mut buf_u = vec![0u8; 12 * cs];
        let mut buf_v = vec![0u8; 12 * cs];
        loop {
            if abort.load(Ordering::Acquire) {
                break;
            }
            let Some(job) = lock(&jobs).next() else {
                break;
            };
            filter_row(
                job,
                &shared,
                specs,
                simple,
                mb_cols,
                (ys, cs),
                [&mut buf_y, &mut buf_u, &mut buf_v],
                &abort,
            );
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            scope.spawn(worker);
        }
        worker();
    });
}

/// Filter one macroblock row (see the module docs).
#[allow(clippy::too_many_arguments)]
fn filter_row(
    job: RowJob,
    shared: &[RowShared],
    specs: &[Option<MbFilterSpec>],
    simple: bool,
    mb_cols: usize,
    (ys, cs): (usize, usize),
    [buf_y, buf_u, buf_v]: [&mut Vec<u8>; 3],
    abort: &AtomicBool,
) {
    let mb_row = job.mb_row;
    let this_row = &shared[mb_row];
    let above_row = mb_row.checked_sub(1).map(|r| &shared[r]);
    // Load this row: body from the picture, bottom four rows from the
    // shared boundary (still unfiltered — only this job writes it).
    buf_y[4 * ys..16 * ys].copy_from_slice(job.body[0]);
    buf_u[4 * cs..8 * cs].copy_from_slice(job.body[1]);
    buf_v[4 * cs..8 * cs].copy_from_slice(job.body[2]);
    let own = &this_row.boundary;
    own.y.load(0, &mut buf_y[16 * ys..20 * ys]);
    own.u.load(0, &mut buf_u[8 * cs..12 * cs]);
    own.v.load(0, &mut buf_v[8 * cs..12 * cs]);

    for mb_col in 0..mb_cols {
        if let Some(above) = above_row {
            let need = (mb_col + 2).min(mb_cols);
            let mut backoff = crate::backoff::Backoff::new();
            while above.done.load(Ordering::Acquire) < need {
                if abort.load(Ordering::Acquire) {
                    return;
                }
                backoff.pause();
            }
            let boundary = &above.boundary;
            let (yx, cx) = (mb_col * 16, mb_col * 8);
            for r in 0..4 {
                boundary
                    .y
                    .load(r * ys + yx, &mut buf_y[r * ys + yx..r * ys + yx + 16]);
                boundary
                    .u
                    .load(r * cs + cx, &mut buf_u[r * cs + cx..r * cs + cx + 8]);
                boundary
                    .v
                    .load(r * cs + cx, &mut buf_v[r * cs + cx..r * cs + cx + 8]);
            }
        }
        if let Some(spec) = &specs[mb_row * mb_cols + mb_col] {
            filter_mb(
                buf_y,
                ys,
                buf_u,
                buf_v,
                cs,
                (mb_col * 16, 4),
                (mb_col * 8, 4),
                spec,
                simple,
                mb_col > 0,
                mb_row > 0,
            );
        }
        // Publish this macroblock's bottom rows, together with the last
        // eight columns of the previous one (whose last three this
        // macroblock's left edge may just have rewritten). The reader
        // waits for the macroblock after the one it copies, so it only
        // ever sees final columns.
        let (from_y, from_c) = (
            (mb_col * 16).saturating_sub(8),
            (mb_col * 8).saturating_sub(8),
        );
        let (to_y, to_c) = (mb_col * 16 + 16, mb_col * 8 + 8);
        for r in 0..4 {
            own.y.store(
                r * ys + from_y,
                &buf_y[(16 + r) * ys + from_y..(16 + r) * ys + to_y],
            );
            own.u.store(
                r * cs + from_c,
                &buf_u[(8 + r) * cs + from_c..(8 + r) * cs + to_c],
            );
            own.v.store(
                r * cs + from_c,
                &buf_v[(8 + r) * cs + from_c..(8 + r) * cs + to_c],
            );
        }
        this_row.done.store(mb_col + 1, Ordering::Release);
    }

    // Write back: the boundary rows above (final now), this row's body,
    // and — on the bottom row, which nothing filters across — its tail.
    if let Some([y, u, v]) = job.above {
        y.copy_from_slice(&buf_y[..4 * ys]);
        u.copy_from_slice(&buf_u[..4 * cs]);
        v.copy_from_slice(&buf_v[..4 * cs]);
    }
    job.body[0].copy_from_slice(&buf_y[4 * ys..16 * ys]);
    job.body[1].copy_from_slice(&buf_u[4 * cs..8 * cs]);
    job.body[2].copy_from_slice(&buf_v[4 * cs..8 * cs]);
    if let Some([y, u, v]) = job.tail {
        y.copy_from_slice(&buf_y[16 * ys..20 * ys]);
        u.copy_from_slice(&buf_u[8 * cs..12 * cs]);
        v.copy_from_slice(&buf_v[8 * cs..12 * cs]);
    }
}
