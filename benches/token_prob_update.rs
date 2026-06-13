//! Round 291 — the §13.4 `token_prob_update()` flag-read loop in
//! isolation.
//!
//! `coded_header::parse_token_prob_update` reads 1056 update flags
//! (4 planes × 8 bands × 3 prev-token classes × 11 positions) once per
//! decoded frame, each at the position-specific `COEFF_UPDATE_PROBS`
//! probability (§13.4 — most entries are 255, "almost always no
//! update"). The round-288 ranked decoder table named it the standing
//! #5 self-time target (≈ 5 %, the "specialised flag-read loop"), but it
//! had no isolated A/B instrument: the whole-frame decode benches fold
//! it inside the rest of the §9 header parse (dimensions, quant indices,
//! loop-filter deltas, refresh flags), so a loop-shape change had no
//! direct measurement.
//!
//! This bench drives `bench_parse_token_prob_update` directly over a
//! pre-encoded header partition produced by the crate's own §13.4
//! writer (`write_no_token_prob_updates` / `write_token_prob_updates`),
//! so the measured loop is a proven-valid bitstream walk (the setup
//! decodes once and asserts the recovered `TokenProbUpdates` equals the
//! intended payload). Two payload shapes partition the cost:
//!
//! * `parse_no_updates` — the all-`None` frame (every flag false): the
//!   overwhelmingly common per-frame payload, so this is the loop's hot
//!   shape. No `L(8)` literal reads — pure 1056-flag descent.
//! * `parse_sparse_updates` — a frame with a handful of `Some(prob)`
//!   replacements scattered across the table, exercising the
//!   `read_literal(8)` replacement-value branch on top of the flag walk.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::coded_header::bench_parse_token_prob_update;
use oxideav_vp8::encoder::{
    write_no_token_prob_updates, write_token_prob_updates, BoolEncoder, COEFF_UPDATE_PROBS_FLAT,
};
use oxideav_vp8::{BoolDecoder, TokenProbUpdates};

/// Encode an all-`None` token_prob_update() block (the common frame).
fn encode_no_updates() -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    write_no_token_prob_updates(&mut enc, &COEFF_UPDATE_PROBS_FLAT);
    // A trailing flag so the final renormalisation has input to pull,
    // mirroring the real header where mb_no_skip_coeff follows.
    enc.write_bool(128, false);
    enc.finish()
}

/// Encode a sparse-update token_prob_update() block: a deterministic
/// scattering of `Some(prob)` replacements exercising the L(8) branch.
fn sparse_updates() -> TokenProbUpdates {
    let mut updates: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
    // Touch ~24 positions spread across the four planes / eight bands so
    // the literal-read branch fires on a realistic minority of flags.
    let mut counter: u8 = 7;
    for (i, plane) in updates.iter_mut().enumerate() {
        for (j, band) in plane.iter_mut().enumerate() {
            if (i + j) % 3 == 0 {
                // Replace the first position of this band's first ctx.
                band[0][0] = Some(counter);
                counter = counter.wrapping_add(29);
            }
        }
    }
    updates
}

fn encode_sparse_updates(updates: &TokenProbUpdates) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    write_token_prob_updates(&mut enc, updates, &COEFF_UPDATE_PROBS_FLAT);
    enc.write_bool(128, false);
    enc.finish()
}

fn bench_token_prob_update(c: &mut Criterion) {
    let no_updates_bytes = encode_no_updates();
    let sparse = sparse_updates();
    let sparse_bytes = encode_sparse_updates(&sparse);

    // Validate both partitions decode to the intended payloads before
    // the measured loop runs — the bench can only ever time a valid
    // bitstream walk.
    {
        let mut dec = BoolDecoder::init(&no_updates_bytes).expect("init no-update partition");
        let parsed = bench_parse_token_prob_update(&mut dec).expect("parse no-update payload");
        let empty: TokenProbUpdates = [[[[None; 11]; 3]; 8]; 4];
        assert_eq!(parsed, empty, "no-update payload must decode to all-None");

        let mut dec = BoolDecoder::init(&sparse_bytes).expect("init sparse partition");
        let parsed = bench_parse_token_prob_update(&mut dec).expect("parse sparse payload");
        assert_eq!(parsed, sparse, "sparse payload must round-trip");
    }

    let mut group = c.benchmark_group("token_prob_update");
    // 1056 flags parsed per call — the per-flag throughput.
    group.throughput(Throughput::Elements(1056));

    group.bench_function("parse_no_updates", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::init(black_box(&no_updates_bytes)).unwrap();
            black_box(bench_parse_token_prob_update(&mut dec).unwrap())
        })
    });

    group.bench_function("parse_sparse_updates", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::init(black_box(&sparse_bytes)).unwrap();
            black_box(bench_parse_token_prob_update(&mut dec).unwrap())
        })
    });

    group.finish();
}

criterion_group!(benches, bench_token_prob_update);
criterion_main!(benches);
