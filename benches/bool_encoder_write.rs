//! Round 409 — the §7.3 boolean entropy *encoder* primitive in isolation.
//!
//! Every coded bit an encoded VP8 frame carries — header flags,
//! macroblock modes, motion vectors, and the dominant share, DCT
//! coefficient tokens — is produced by [`BoolEncoder::write_bool`] and
//! its carry-propagating renormalisation loop. The decoder-side twin
//! (`bool_decoder_read`, round 295) has had an isolated harness for a
//! hundred rounds; the write side never did — the whole-frame
//! `keyframe_encode` / `inter_encode_short_clip` benches fold it behind
//! prediction, transforms and RD scoring, so no change to the §7.3
//! write loop shape (renormalisation cadence, carry scan, literal
//! specialisation) has ever had a direct A/B target.
//!
//! The regimes mirror the decoder bench so the two sides compare
//! directly:
//!
//! * `write_bool_skewed_64k` — 65 536 booleans at a strongly skewed
//!   probability (value ~97 % `false` at prob 248). The interval keeps
//!   most of its width on nearly every write, so the renormalisation
//!   `while range < 128` loop rarely iterates: the well-modelled
//!   coefficient-context / skip-flag regime.
//! * `write_bool_balanced_64k` — 65 536 fair-coin booleans at prob 128.
//!   The split halves the interval on every write, so the
//!   renormalisation loop runs ≈ once per bool and the byte-emit path
//!   fires every eighth write: the §7.3 write worst case.
//! * `write_literal_8b_8k` — 8 192 `write_literal(v, 8)` calls (65 536
//!   flat-probability-128 flag writes assembled MSB-first), the §9
//!   header / §13 partition-size `L(n)` idiom.
//! * `write_signed_literal_7b_8k` — 8 192 `write_signed_literal(v, 7)`
//!   calls (six magnitude bits + sign), the §17 MV-component idiom.
//!
//! Each measured iteration constructs a fresh encoder, writes the whole
//! deterministic value stream and finishes the partition, so the
//! byte-vector growth policy is measured alongside the arithmetic (the
//! two are inseparable in real frame emission). Value streams come from
//! a fixed xorshift32 so every run measures identical work, and each
//! stream is decoded once at setup with the crate's own §7.3
//! [`BoolDecoder`], asserting exact round-trip — the measured loop is
//! always a proven-valid partition build.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::{BoolDecoder, BoolEncoder};

/// Number of booleans in each `write_bool` stream.
const NUM_BOOLS: usize = 65_536;
/// Number of `write_literal(8)` calls (each producing 8 booleans).
const NUM_LITERALS: usize = 8_192;
/// Width of each literal in the `write_literal` bench.
const LITERAL_BITS: u32 = 8;
/// Number of `write_signed_literal` calls in the signed bench.
const NUM_SIGNED_LITERALS: usize = 8_192;
/// Sign + magnitude width of each signed literal (the §17 idiom).
const SIGNED_LITERAL_BITS: u32 = 7;

/// The skewed regime's probability-of-false.
const SKEWED_PROB: u8 = 248;
/// The balanced regime's probability-of-false (a fair coin).
const BALANCED_PROB: u8 = 128;

/// xorshift32 — deterministic, dependency-free value stream so each
/// measured iteration writes the identical bit sequence.
struct XorShift32(u32);

impl XorShift32 {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// A boolean stream matching a probability-of-false `prob`: each value
/// is `true` with probability `(256 - prob) / 256`.
fn bool_stream(prob: u8, seed: u32, len: usize) -> Vec<bool> {
    let mut rng = XorShift32(seed);
    (0..len)
        .map(|_| (rng.next() & 0xFF) as u16 >= prob as u16)
        .collect()
}

/// A literal-value stream, each masked to `bits` wide.
fn literal_stream(bits: u32, seed: u32, len: usize) -> Vec<u32> {
    let mut rng = XorShift32(seed);
    (0..len).map(|_| rng.next() & ((1 << bits) - 1)).collect()
}

/// A signed-literal stream for `write_signed_literal(v, bits)`:
/// magnitudes strictly below `1 << bits`, both signs.
fn signed_stream(bits: u32, seed: u32, len: usize) -> Vec<i32> {
    let mut rng = XorShift32(seed);
    (0..len)
        .map(|_| {
            let magnitude = (rng.next() & ((1 << bits) - 1)) as i32;
            if rng.next() & 1 == 1 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

/// Round-trip a bool stream through the crate's own decoder, proving the
/// benched writes build a valid partition.
fn verify_bools(prob: u8, values: &[bool]) {
    let mut enc = BoolEncoder::new();
    for &v in values {
        enc.write_bool(prob, v);
    }
    let bytes = enc.finish();
    let mut dec = BoolDecoder::init(&bytes).expect("partition init");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(dec.read_bool(prob).expect("read"), v, "bool {i} mismatch");
    }
}

fn bench_write_bool(c: &mut Criterion) {
    let mut g = c.benchmark_group("bool_encoder_write");

    for (name, prob, seed) in [
        ("write_bool_skewed_64k", SKEWED_PROB, 0x1234_5678),
        ("write_bool_balanced_64k", BALANCED_PROB, 0x9E37_79B9),
    ] {
        let values = bool_stream(prob, seed, NUM_BOOLS);
        verify_bools(prob, &values);
        g.throughput(Throughput::Elements(NUM_BOOLS as u64));
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut enc = BoolEncoder::new();
                for &v in &values {
                    enc.write_bool(black_box(prob), v);
                }
                black_box(enc.finish().len())
            })
        });
    }

    g.finish();
}

fn bench_write_literals(c: &mut Criterion) {
    let mut g = c.benchmark_group("bool_encoder_write");

    // Unsigned L(8) literals.
    {
        let values = literal_stream(LITERAL_BITS, 0xDEAD_BEEF, NUM_LITERALS);
        // Round-trip proof.
        let mut enc = BoolEncoder::new();
        for &v in &values {
            enc.write_literal(v, LITERAL_BITS);
        }
        let bytes = enc.finish();
        let mut dec = BoolDecoder::init(&bytes).expect("partition init");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(
                dec.read_literal(LITERAL_BITS).expect("read"),
                v,
                "literal {i} mismatch"
            );
        }
        g.throughput(Throughput::Elements((NUM_LITERALS as u64) * 8));
        g.bench_function("write_literal_8b_8k", |b| {
            b.iter(|| {
                let mut enc = BoolEncoder::new();
                for &v in &values {
                    enc.write_literal(black_box(v), LITERAL_BITS);
                }
                black_box(enc.finish().len())
            })
        });
    }

    // Signed 6-magnitude-bit + sign literals.
    {
        let values = signed_stream(SIGNED_LITERAL_BITS - 1, 0xCAFE_F00D, NUM_SIGNED_LITERALS);
        let mut enc = BoolEncoder::new();
        for &v in &values {
            enc.write_signed_literal(v, SIGNED_LITERAL_BITS - 1);
        }
        let bytes = enc.finish();
        // `write_signed_literal` uses the §9.3 delta wire format — L(n)
        // magnitude then an L(1) sign flag — so the round-trip proof
        // reads it back the same way.
        let mut dec = BoolDecoder::init(&bytes).expect("partition init");
        for (i, &v) in values.iter().enumerate() {
            let magnitude = dec.read_literal(SIGNED_LITERAL_BITS - 1).expect("read") as i32;
            let negative = dec.read_bool(128).expect("read sign");
            let recovered = if negative { -magnitude } else { magnitude };
            assert_eq!(recovered, v, "signed literal {i} mismatch");
        }
        g.throughput(Throughput::Elements(
            (NUM_SIGNED_LITERALS as u64) * SIGNED_LITERAL_BITS as u64,
        ));
        g.bench_function("write_signed_literal_7b_8k", |b| {
            b.iter(|| {
                let mut enc = BoolEncoder::new();
                for &v in &values {
                    enc.write_signed_literal(black_box(v), SIGNED_LITERAL_BITS - 1);
                }
                black_box(enc.finish().len())
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_write_bool, bench_write_literals);
criterion_main!(benches);
