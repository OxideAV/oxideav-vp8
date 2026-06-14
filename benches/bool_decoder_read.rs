//! Round 295 — the §7.3 boolean entropy decoder primitive in isolation.
//!
//! Every coded bit in a VP8 frame — header flags, macroblock modes,
//! motion-vector components, and the dominant share, DCT-coefficient
//! tokens — passes through [`BoolDecoder::read_bool`] and its inner
//! [`renormalize`] step. The whole-frame `keyframe_decode` /
//! `inter_decode_short_clip` benches and the `token_decode` bench all
//! exercise this primitive, but always wrapped inside the §13 token
//! descent, §12/§16 prediction, §14 transforms, and the §15 loop
//! filter. None of them attributes wall-time to the §7.3 `read_bool` /
//! `read_literal` loop *alone*.
//!
//! `bool_decoder.rs` documents a deliberate optimisation: the §7.3
//! bit-at-a-time renormalisation loop is collapsed into a single
//! `range.leading_zeros()`-derived shift, with at most one input-byte
//! pull per `read_bool`. That batching has no isolated A/B target —
//! any future change to the renormalisation shape (e.g. a different
//! byte-refill cadence, or a two-bit-at-a-time read) would only show
//! up diluted inside the whole-frame numbers. This bench gives the
//! §7.3 layer its own harness across the three regimes that bracket
//! its real cost:
//!
//! * `read_bool_skewed_64k` — 65 536 booleans at a strongly skewed
//!   probability (one outcome ~97 % likely). The interval split rarely
//!   needs more than zero or one doubling, so the renormalisation
//!   fast-path (`shift == 0` early return) dominates. This is the
//!   well-modelled regime: a confident coefficient-context or skip flag.
//! * `read_bool_balanced_64k` — 65 536 booleans at probability 128
//!   (a fair coin). The split lands near the middle of the interval,
//!   so almost every read triggers a one- or two-bit renormalisation
//!   and the byte-refill path fires often — the §7.3 renorm worst case.
//! * `read_literal_8b_8k` — 8 192 `read_literal(8)` calls, i.e. 65 536
//!   flat-probability-128 flag reads assembled MSB-first into bytes.
//!   This is the §9 header / §17 MV-magnitude idiom (`L(n)` macro),
//!   measuring the `read_literal` accumulator loop on top of `read_bool`.
//! * `read_literal_mixed_width` — 8 192 `read_literal` calls of widths
//!   spread across `1..=16` (the real §9 header / §13 partition-size mix),
//!   so the register-local accumulator loop is exercised across the full
//!   width spread rather than the single width-8 shape.
//! * `read_signed_literal_7b_8k` — 8 192 `read_signed_literal(7)` calls
//!   (sign bit + six magnitude bits), the §17 MV-component idiom that the
//!   width-8 literal bench never touched.
//!
//! As of round 306 the `read_literal` accumulator is a register-local
//! specialisation of the generic `num_bits × read_bool(128)` loop (the
//! fixed-probability-128 split collapses the §7.2 interval multiply to a
//! single shift). The mixed-width and signed cases give that loop the
//! width-varied A/B coverage the flat width-8 bench lacked; bit-exactness
//! against the generic loop is enforced by the `bool_decoder` unit tests
//! (`read_literal_fast_matches_generic_loop` /
//! `read_literal_fast_matches_generic_on_end_of_stream`).
//!
//! Every partition is produced by the crate's own §7.3
//! [`BoolEncoder`] (the exact inverse of the decoder under test) and
//! decoded once at setup, asserting the recovered bit/byte stream
//! equals the intended one — so the measured loop is always a
//! proven-valid bitstream walk, never an error path.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{BoolDecoder, BoolEncoder};

/// Number of booleans in each `read_bool` partition.
const NUM_BOOLS: usize = 65_536;
/// Number of `read_literal(8)` calls (each consuming 8 booleans).
const NUM_LITERALS: usize = 8_192;
/// Width of each literal in the `read_literal` bench.
const LITERAL_BITS: u32 = 8;

/// Number of mixed-width `read_literal` calls in the multi-width bench.
const NUM_MIXED_LITERALS: usize = 8_192;
/// Number of `read_signed_literal` calls in the signed-literal bench.
const NUM_SIGNED_LITERALS: usize = 8_192;
/// Width (sign + magnitude) of each signed literal — the §17 MV-magnitude
/// `read_signed_literal` idiom uses several fixed widths; 7 brackets the
/// common case.
const SIGNED_LITERAL_BITS: u32 = 7;

/// xorshift32 — deterministic, dependency-free value stream so the
/// encoded partition is fixed across runs and machines.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Build the deterministic boolean sequence for the `read_bool` benches.
///
/// `prob` is the §7.3 probability-of-`false` byte used for *both* the
/// encode and the decode (the decoder must use the same probability the
/// encoder split against). The bits are skewed to match: when `prob` is
/// large, `false` is generated with matching frequency, so the encoded
/// stream is short and the decode exercises the modelled fast path; at
/// `prob == 128` the bits are an even split.
fn build_bools(prob: u8) -> Vec<bool> {
    let mut rng = 0x9e37_79b9u32;
    (0..NUM_BOOLS)
        .map(|_| {
            // Threshold the xorshift draw against `prob` so the realised
            // frequency of `false` tracks `prob / 256` — the regime the
            // probability models.
            let draw = (xorshift(&mut rng) & 0xff) as u8;
            draw >= prob
        })
        .collect()
}

/// Encode a boolean sequence into one §7.3 partition at fixed `prob`.
fn encode_bools(bits: &[bool], prob: u8) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    for &b in bits {
        enc.write_bool(prob, b);
    }
    enc.finish()
}

/// Decode the partition once and assert it reproduces `bits` — the
/// setup-time validity proof for the `read_bool` benches.
fn verify_bools(bytes: &[u8], bits: &[bool], prob: u8) {
    let mut dec = BoolDecoder::init(bytes).expect("init bool partition");
    for (i, &want) in bits.iter().enumerate() {
        let got = dec.read_bool(prob).expect("read_bool");
        assert_eq!(got, want, "bool mismatch at index {i}");
    }
}

/// Build the deterministic literal sequence for the `read_literal` bench.
fn build_literals() -> Vec<u32> {
    let mut rng = 0x1234_5678u32;
    let mask = (1u32 << LITERAL_BITS) - 1;
    (0..NUM_LITERALS)
        .map(|_| xorshift(&mut rng) & mask)
        .collect()
}

/// Encode literals MSB-first at flat probability 128 (the `L(n)` macro).
fn encode_literals(vals: &[u32]) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    for &v in vals {
        enc.write_literal(v, LITERAL_BITS);
    }
    enc.finish()
}

/// Decode the literal partition once and assert it reproduces `vals`.
fn verify_literals(bytes: &[u8], vals: &[u32]) {
    let mut dec = BoolDecoder::init(bytes).expect("init literal partition");
    for (i, &want) in vals.iter().enumerate() {
        let got = dec.read_literal(LITERAL_BITS).expect("read_literal");
        assert_eq!(got, want, "literal mismatch at index {i}");
    }
}

/// Build a deterministic mixed-width literal sequence: each entry pairs a
/// width in `1..=16` with a value masked to that width. Exercises the
/// `read_literal` register-local loop across the full width spread the §9
/// header / §13 partition-size reads use, not just the flat width-8 case.
fn build_mixed_literals() -> Vec<(u32, u32)> {
    let mut rng = 0x0bad_c0deu32;
    (0..NUM_MIXED_LITERALS)
        .map(|_| {
            let w = (xorshift(&mut rng) % 16) + 1; // 1..=16
            let mask = (1u32 << w) - 1;
            let v = xorshift(&mut rng) & mask;
            (w, v)
        })
        .collect()
}

/// Encode mixed-width literals MSB-first at flat probability 128.
fn encode_mixed_literals(items: &[(u32, u32)]) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    for &(w, v) in items {
        enc.write_literal(v, w);
    }
    enc.finish()
}

/// Decode the mixed-width partition once and assert it reproduces `items`.
fn verify_mixed_literals(bytes: &[u8], items: &[(u32, u32)]) {
    let mut dec = BoolDecoder::init(bytes).expect("init mixed literal partition");
    for (i, &(w, want)) in items.iter().enumerate() {
        let got = dec.read_literal(w).expect("read_literal");
        assert_eq!(got, want, "mixed literal mismatch at index {i}");
    }
}

/// Build a deterministic signed-literal sequence (the §17 `read_signed_literal`
/// idiom: a sign bit then `SIGNED_LITERAL_BITS - 1` magnitude bits).
fn build_signed_literals() -> Vec<i32> {
    let mut rng = 0x00c0_ffeeu32;
    let mag_bits = SIGNED_LITERAL_BITS - 1;
    let mag_mask = (1u32 << mag_bits) - 1;
    (0..NUM_SIGNED_LITERALS)
        .map(|_| {
            let r = xorshift(&mut rng);
            let mag = (r & mag_mask) as i32;
            if (r >> 16) & 1 == 1 {
                -1 - mag
            } else {
                mag
            }
        })
        .collect()
}

/// Encode signed literals under the decoder's accumulate-from-`-1`
/// convention: emit the sign bit, then the magnitude bits whose
/// `(v << 1) | bit` accumulation from the sign-seeded value reproduces the
/// target — i.e. the low `SIGNED_LITERAL_BITS - 1` bits of the two's
/// complement value, MSB-first.
fn encode_signed_literals(vals: &[i32]) -> Vec<u8> {
    let mut enc = BoolEncoder::new();
    let mag_bits = SIGNED_LITERAL_BITS - 1;
    for &val in vals {
        let sign = val < 0;
        enc.write_bool(128, sign);
        let bits = (val as u32) & ((1u32 << mag_bits) - 1);
        enc.write_literal(bits, mag_bits);
    }
    enc.finish()
}

/// Decode the signed-literal partition once and assert it reproduces `vals`.
fn verify_signed_literals(bytes: &[u8], vals: &[i32]) {
    let mut dec = BoolDecoder::init(bytes).expect("init signed literal partition");
    for (i, &want) in vals.iter().enumerate() {
        let got = dec
            .read_signed_literal(SIGNED_LITERAL_BITS)
            .expect("read_signed_literal");
        assert_eq!(got, want, "signed literal mismatch at index {i}");
    }
}

fn bench_read_bool(c: &mut Criterion) {
    // prob 248 ≈ 97 % skew toward `false` (modelled fast path);
    // prob 128 is the fair-coin renorm worst case.
    for (name, prob) in [
        ("read_bool_skewed_64k", 248u8),
        ("read_bool_balanced_64k", 128u8),
    ] {
        let bits = build_bools(prob);
        let bytes = encode_bools(&bits, prob);
        verify_bools(&bytes, &bits, prob);

        let mut g = c.benchmark_group("bool_decoder_read");
        g.throughput(Throughput::Elements(NUM_BOOLS as u64));
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut dec = BoolDecoder::init(black_box(&bytes)).expect("init");
                let mut acc = 0u64;
                for _ in 0..NUM_BOOLS {
                    acc += dec.read_bool(prob).expect("read_bool") as u64;
                }
                black_box(acc)
            });
        });
        g.finish();
    }
}

fn bench_read_literal(c: &mut Criterion) {
    let vals = build_literals();
    let bytes = encode_literals(&vals);
    verify_literals(&bytes, &vals);

    let mut g = c.benchmark_group("bool_decoder_read");
    // One element per coded bit, so the number is comparable to the
    // `read_bool` benches' per-bool throughput.
    g.throughput(Throughput::Elements(
        NUM_LITERALS as u64 * LITERAL_BITS as u64,
    ));
    g.bench_function("read_literal_8b_8k", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::init(black_box(&bytes)).expect("init");
            let mut acc = 0u64;
            for _ in 0..NUM_LITERALS {
                acc += dec.read_literal(LITERAL_BITS).expect("read_literal") as u64;
            }
            black_box(acc)
        });
    });
    g.finish();
}

fn bench_read_mixed_literal(c: &mut Criterion) {
    let items = build_mixed_literals();
    let bytes = encode_mixed_literals(&items);
    verify_mixed_literals(&bytes, &items);

    let total_bits: u64 = items.iter().map(|&(w, _)| w as u64).sum();

    let mut g = c.benchmark_group("bool_decoder_read");
    g.throughput(Throughput::Elements(total_bits));
    g.bench_function("read_literal_mixed_width", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::init(black_box(&bytes)).expect("init");
            let mut acc = 0u64;
            for &(w, _) in &items {
                acc += dec.read_literal(w).expect("read_literal") as u64;
            }
            black_box(acc)
        });
    });
    g.finish();
}

fn bench_read_signed_literal(c: &mut Criterion) {
    let vals = build_signed_literals();
    let bytes = encode_signed_literals(&vals);
    verify_signed_literals(&bytes, &vals);

    let mut g = c.benchmark_group("bool_decoder_read");
    g.throughput(Throughput::Elements(
        NUM_SIGNED_LITERALS as u64 * SIGNED_LITERAL_BITS as u64,
    ));
    g.bench_function("read_signed_literal_7b_8k", |b| {
        b.iter(|| {
            let mut dec = BoolDecoder::init(black_box(&bytes)).expect("init");
            let mut acc = 0i64;
            for _ in 0..NUM_SIGNED_LITERALS {
                acc += dec
                    .read_signed_literal(SIGNED_LITERAL_BITS)
                    .expect("read_signed_literal") as i64;
            }
            black_box(acc)
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_read_bool,
    bench_read_literal,
    bench_read_mixed_literal,
    bench_read_signed_literal
);
criterion_main!(benches);
