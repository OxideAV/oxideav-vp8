//! Round 288 — the §9.7 / §9.8 reference-frame slot rotation: the
//! per-frame `RefFrameSlot` clone churn `Vp8DecoderState::decode_frame`
//! runs after every decoded frame.
//!
//! The round-285 / round-286 ranked decoder hotspot table named this
//! path the standing **runner-up** profile-opt target: the
//! `memmove` / `memset` family was the #3 self-time cluster on the inter
//! decode profile (≈ 15–16 %, ~1600 of ~10.1k samples under nightly +
//! `simd`), attributed to "per-frame reference-slot plane copying
//! (`decode_frame` body + `RefFrameSlot::clone`)". Yet no bench isolated
//! it: every whole-frame decode bench (`keyframe_decode`,
//! `inter_decode_short_clip`, `token_decode`) folds the rotation in
//! behind header parse + token decode + reconstruction + loop filter,
//! and there was no micro-target measuring the clone-driven copies on
//! their own. This harness closes that gap with two layers:
//!
//! 1. **`rotate_*` micro** — the §20 page-147 rotation walk
//!    (`copy_arf → copy_gf → refresh_gf → refresh_arf → refresh_last`)
//!    in isolation. It mirrors the exact `RefFrameSlot::clone` sequence
//!    `decode_frame` stages into temporaries (`current_slot` +
//!    `pre_{last,golden,altref}` + `new_{altref,golden,last}`), driven
//!    over public `RefFrameSlot` fields at a realistic frame size, for
//!    the flag combinations that partition the decoder's per-frame cost:
//!    * `refresh_last_only` — the round-149 hardwired P-frame ladder
//!      (`refresh_last = 1`, everything else 0). The overwhelmingly
//!      common case: one current-slot capture + one move into LAST.
//!    * `refresh_all` — `refresh_{golden,alternate,last}` all set: three
//!      current-slot clones into three slots (the heaviest refresh).
//!    * `copy_gf_arf` — `copy_buffer_to_golden = 1` (LAST → GOLDEN) +
//!      `copy_buffer_to_alternate = 2` (GOLDEN → ALTREF) + refresh_last:
//!      the §16 cross-slot-copy shape (clones existing slots, not the
//!      current reconstruction).
//!
//! 2. **`decode_*` whole-stream** — decode through `Vp8DecoderState`
//!    over streams the crate's own §16 encoder produces with two refresh
//!    cadences, so the *actual* in-decoder rotation runs at its true
//!    per-frame share:
//!    * `last_only` — the default ladder (refresh_last every P-frame):
//!      the rotation baseline.
//!    * `golden_altref` — a periodic `refresh_golden_frame` /
//!      `refresh_alternate_frame` cadence (every 2nd / 3rd P-frame),
//!      driving the heavy three-clone branches the profile attributes
//!      the `memmove` cluster to.
//!
//! 3. **`shipped_*` micro (round 308)** — the SAME three flag
//!    combinations, but driven through `bench_rotate_reference_slots`,
//!    the shipped §9 *minimal-copy* rotation `decode_frame` actually
//!    runs (move each owned source into the last destination that
//!    selects it; clone only a genuinely multi-fed source). The layer-1
//!    `rotate_*` harness above is a clone-everything stand-in — it was a
//!    naive upper bound, not the cost the decoder pays. Pairing the two
//!    side by side makes the gap measurable: on the common
//!    refresh-last path the shipped rotation performs **zero** plane
//!    clones (every source moves into exactly one destination), so its
//!    cost collapses toward bookkeeping while the clone-everything
//!    stand-in keeps paying for six populated `Vec` copies. The residual
//!    real copy cost is concentrated in the §16 cross-slot-copy shape
//!    (`copy_gf_arf`, where one pre-rotation slot feeds two
//!    destinations) and the key-frame init (one source → three slots),
//!    which only a copy-on-write (`Rc`/`Arc`-backed) slot representation
//!    can remove — the move-based rotation already removes everything
//!    else.
//!
//! The round-308 measurement correction: the `rotate_*` rows are the
//! clone-everything ceiling; the `shipped_*` rows are the floor the
//! decoder already operates at. The standing copy-on-write profile-opt
//! target should be sized against the `shipped_*` numbers, not the
//! `rotate_*` ceiling.
//!
//! No decoder/encoder logic change — this is a bench-only (measurement)
//! round; the new `shipped_*` rows call the unmodified shipped rotation
//! through a `#[doc(hidden)]` measurement shim. The layer-1 micro path
//! reconstructs the clone-everything reference from the public
//! `RefFrameSlot` surface (its fields are `pub`); the whole-stream path
//! exercises the shipped `decode_frame` rotation unmodified.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{
    bench_rotate_reference_slots, I420Frame, KeyframeParams, RefFrameSlot, RefreshControls,
    Vp8DecoderState, Vp8InterStreamEncoder,
};

// A mid-size frame: 320×240 = 20×15 MBs — the same resolution the
// `keyframe_encode` / `keyframe_decode` benches drive, so the plane byte
// counts (and therefore the per-slot `memmove` volume) line up with the
// numbers in those benches.
const MB_COLS: usize = 20;
const MB_ROWS: usize = 15;

/// Build one `RefFrameSlot` with deterministic plane bytes at the
/// 320×240 geometry. Fields are public so the bench can synthesise a
/// slot without reaching into the decoder.
fn make_slot(seed: u8) -> RefFrameSlot {
    let y_stride = MB_COLS * 16;
    let uv_stride = MB_COLS * 8;
    let y_len = y_stride * MB_ROWS * 16;
    let uv_len = uv_stride * MB_ROWS * 8;
    let mut y = vec![0u8; y_len];
    let mut u = vec![0u8; uv_len];
    let mut v = vec![0u8; uv_len];
    for (i, p) in y.iter_mut().enumerate() {
        *p = (i as u8).wrapping_add(seed);
    }
    for (i, p) in u.iter_mut().enumerate() {
        *p = (i as u8).wrapping_mul(3).wrapping_add(seed);
    }
    for (i, p) in v.iter_mut().enumerate() {
        *p = (i as u8).wrapping_mul(5).wrapping_add(seed);
    }
    RefFrameSlot {
        y,
        u,
        v,
        y_stride,
        uv_stride,
        mb_cols: MB_COLS,
        mb_rows: MB_ROWS,
    }
}

/// The three pre-rotation reference slots — the §9 buffer state the
/// decoder carries into `decode_frame`.
struct SlotSet {
    last: Option<RefFrameSlot>,
    golden: Option<RefFrameSlot>,
    altref: Option<RefFrameSlot>,
}

/// The §20 page-147 rotation walk, replicating the temporaries
/// `Vp8DecoderState::decode_frame` stages exactly. `current` is the
/// just-decoded frame; `slots` is the pre-rotation buffer; `refresh`
/// carries the `copy_*` / `refresh_*` selectors as the coded-header
/// fields name them. Returns the new `SlotSet` so the clones are
/// observable (not dead-code-eliminated).
fn rotate(current: &RefFrameSlot, slots: &SlotSet, refresh: &RefreshControls) -> SlotSet {
    let pre_last = slots.last.clone();
    let pre_golden = slots.golden.clone();
    let pre_altref = slots.altref.clone();

    // copy_buffer_to_alternate: 1 = copy LAST, 2 = copy GOLDEN.
    let mut new_altref = pre_altref.clone();
    match refresh.copy_buffer_to_alternate {
        1 => new_altref = pre_last.clone(),
        2 => new_altref = pre_golden.clone(),
        _ => {}
    }
    // copy_buffer_to_golden: 1 = copy LAST, 2 = copy ALTREF.
    let mut new_golden = pre_golden.clone();
    match refresh.copy_buffer_to_golden {
        1 => new_golden = pre_last.clone(),
        2 => new_golden = pre_altref.clone(),
        _ => {}
    }
    if refresh.refresh_golden_frame {
        new_golden = Some(current.clone());
    }
    if refresh.refresh_alternate_frame {
        new_altref = Some(current.clone());
    }
    let new_last = if refresh.refresh_last {
        Some(current.clone())
    } else {
        pre_last
    };
    SlotSet {
        last: new_last,
        golden: new_golden,
        altref: new_altref,
    }
}

fn bench_rotate_micro(c: &mut Criterion) {
    let current = make_slot(1);
    let slots = SlotSet {
        last: Some(make_slot(2)),
        golden: Some(make_slot(3)),
        altref: Some(make_slot(4)),
    };

    // One frame's worth of plane bytes — the volume each `clone()` moves.
    let bytes_per_slot = current.y.len() + current.u.len() + current.v.len();

    // refresh_last only: the round-149 hardwired P-frame ladder.
    let refresh_last_only = RefreshControls::default();
    // refresh_golden + refresh_alternate + refresh_last: three
    // current-slot clones into three slots (heaviest refresh shape).
    let refresh_all = RefreshControls {
        refresh_golden_frame: true,
        refresh_alternate_frame: true,
        copy_buffer_to_golden: 0,
        copy_buffer_to_alternate: 0,
        refresh_last: true,
    };
    // copy_buffer_to_golden = 1 (LAST → GOLDEN), copy_buffer_to_alternate
    // = 2 (GOLDEN → ALTREF), refresh_last: the §16 cross-slot-copy shape.
    let copy_gf_arf = RefreshControls {
        refresh_golden_frame: false,
        refresh_alternate_frame: false,
        copy_buffer_to_golden: 1,
        copy_buffer_to_alternate: 2,
        refresh_last: true,
    };

    let mut g = c.benchmark_group("ref_slot_rotation");
    g.throughput(Throughput::Bytes(bytes_per_slot as u64));

    for (name, refresh) in [
        ("rotate_refresh_last_only", &refresh_last_only),
        ("rotate_refresh_all", &refresh_all),
        ("rotate_copy_gf_arf", &copy_gf_arf),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let out = rotate(black_box(&current), black_box(&slots), black_box(refresh));
                black_box(&out).last.as_ref().map(|s| s.y[0])
            });
        });
    }

    g.finish();
}

/// The shipped §9 minimal-copy rotation under the same three flag
/// combinations as `bench_rotate_micro`, driven through the
/// `bench_rotate_reference_slots` shim so the bench measures what
/// `decode_frame` actually pays (move-based; clone only a multi-fed
/// source) rather than the clone-everything ceiling. Each iteration
/// re-clones the entry slots into owned inputs (the shim consumes by
/// value, exactly as `decode_frame` passes `self.{last,golden,altref}.take()`),
/// then runs the rotation; the input clones are charged identically across
/// all three rows so the row-to-row delta isolates the rotation's own
/// copy work.
fn bench_rotate_shipped(c: &mut Criterion) {
    let current = make_slot(1);
    let pre_last = make_slot(2);
    let pre_golden = make_slot(3);
    let pre_altref = make_slot(4);

    let bytes_per_slot = current.y.len() + current.u.len() + current.v.len();

    // (copy_arf, copy_gf, refresh_gf, refresh_arf, refresh_last)
    type RefreshFlags = (u8, u8, bool, bool, bool);
    let cases: [(&str, RefreshFlags); 3] = [
        // refresh_last only: current moves into LAST, both pre slots move
        // through untouched — zero clones in the shipped path.
        ("shipped_refresh_last_only", (0, 0, false, false, true)),
        // refresh all three: current feeds LAST, GOLDEN and ALTREF, so it
        // is cloned twice and moved once.
        ("shipped_refresh_all", (0, 0, true, true, true)),
        // copy_buffer_to_golden = 1 (LAST → GOLDEN), copy_buffer_to_alternate
        // = 2 (GOLDEN → ALTREF), refresh_last: pre-LAST feeds both new-LAST
        // and new-GOLDEN (one clone), pre-GOLDEN feeds new-ALTREF (move).
        ("shipped_copy_gf_arf", (2, 1, false, false, true)),
    ];

    let mut g = c.benchmark_group("ref_slot_rotation");
    g.throughput(Throughput::Bytes(bytes_per_slot as u64));

    for (name, (copy_arf, copy_gf, refresh_gf, refresh_arf, refresh_last)) in cases {
        g.bench_function(name, |b| {
            b.iter(|| {
                let out = bench_rotate_reference_slots(
                    black_box(current.clone()),
                    Some(black_box(pre_last.clone())),
                    Some(black_box(pre_golden.clone())),
                    Some(black_box(pre_altref.clone())),
                    black_box(copy_arf),
                    black_box(copy_gf),
                    black_box(refresh_gf),
                    black_box(refresh_arf),
                    black_box(refresh_last),
                );
                black_box(&out).0.as_ref().map(|s| s.y[0])
            });
        });
    }

    g.finish();
}

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const P_FRAMES: usize = 8;

/// Deterministic drifting clip — one keyframe + `P_FRAMES` P-frames, each
/// a one-pixel translation of the previous so the stream carries real
/// motion vectors (mirrors the `inter_decode_short_clip` generator at
/// 320×240).
fn synthesise() -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let cw = (WIDTH as usize).div_ceil(2);
    let ch = (HEIGHT as usize).div_ceil(2);
    (0..=P_FRAMES)
        .map(|t| {
            let mut y = vec![0u8; w * h];
            for row in 0..h {
                for col in 0..w {
                    y[row * w + col] = (((col + t) * 256 / w + (row + t) * 256 / h) / 2) as u8;
                }
            }
            let mut u = vec![128u8; cw * ch];
            let mut v = vec![128u8; cw * ch];
            for row in 0..ch {
                for col in 0..cw {
                    u[row * cw + col] = (120 + (col * 16 / cw) + t) as u8;
                    v[row * cw + col] = (130 + (row * 16 / ch) + t) as u8;
                }
            }
            (y, u, v)
        })
        .collect()
}

/// Encode the clip once with a caller-driven refresh cadence. `golden`
/// engages a periodic `refresh_golden_frame` / `refresh_alternate_frame`
/// pattern; otherwise every P-frame uses the default refresh-last ladder.
fn encode_stream(golden: bool) -> Vec<Vec<u8>> {
    let clip = synthesise();
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };
    // Large keyframe interval so only frame 0 is a keyframe; the rest are
    // P-frames we drive explicitly.
    let mut enc = Vp8InterStreamEncoder::new(params, 10_000).expect("Vp8InterStreamEncoder::new");
    let mut packets = Vec::with_capacity(clip.len());
    // Frame 0: keyframe (refreshes all slots).
    {
        let (y, u, v) = &clip[0];
        let frame = I420Frame::packed(WIDTH, HEIGHT, y, u, v);
        packets.push(enc.encode_frame(&frame).expect("keyframe encode").bytes);
    }
    // P-frames 1..=P_FRAMES.
    for (idx, (y, u, v)) in clip.iter().enumerate().skip(1) {
        let frame = I420Frame::packed(WIDTH, HEIGHT, y, u, v);
        let refresh = if golden {
            // Periodic golden/altref refresh so the heavy three-clone
            // rotation branches fire on a realistic cadence: refresh
            // golden every 2nd P-frame, altref every 3rd. Never sets a
            // copy_buffer selector and the matching refresh bit together
            // (RefreshControls::validate forbids it).
            RefreshControls {
                refresh_golden_frame: idx % 2 == 0,
                refresh_alternate_frame: idx % 3 == 0,
                copy_buffer_to_golden: 0,
                copy_buffer_to_alternate: 0,
                refresh_last: true,
            }
        } else {
            RefreshControls::default()
        };
        let pkt = enc
            .encode_p_frame_with_refresh(&frame, &refresh)
            .expect("p-frame encode")
            .bytes;
        packets.push(pkt);
    }
    packets
}

fn bench_decode_with_refresh(c: &mut Criterion) {
    let last_only = encode_stream(false);
    let golden_altref = encode_stream(true);

    let mut g = c.benchmark_group("ref_slot_rotation");
    g.throughput(Throughput::Elements(
        (WIDTH * HEIGHT) as u64 * (P_FRAMES as u64 + 1),
    ));
    g.sample_size(20);

    for (name, packets) in [
        ("decode_1k8p_last_only", &last_only),
        ("decode_1k8p_golden_altref", &golden_altref),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut dec = Vp8DecoderState::new();
                let mut acc = 0usize;
                for pkt in packets {
                    let frame = dec.decode_frame(black_box(pkt)).expect("decode_frame");
                    acc += frame.y.len();
                }
                black_box(acc)
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_rotate_micro,
    bench_rotate_shipped,
    bench_decode_with_refresh
);
criterion_main!(benches);
