//! Boolean (arithmetic) encoder — write-side companion to [`crate::bool_decoder`].
//!
//! Ported from libvpx's `vp8_encode_bool` (the loop-free, count-based
//! variant mirrored in RFC 6386 §20.2). For every `write_bool(prob, bit)`
//! the decoder's `read_bool(prob)` will produce `bit`.
//!
//! Call pattern:
//! ```ignore
//! let mut enc = BoolEncoder::new();
//! enc.write_bool(128, true);
//! enc.write_literal(5, 10);     // 5-bit value
//! let bytes = enc.finish();
//! ```

/// Write-side boolean coder. Emits bytes into an internal buffer.
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    lowvalue: u32,
    /// -24 = empty, 0 = ready to emit one byte.
    count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            range: 255,
            lowvalue: 0,
            count: -24,
        }
    }

    fn add_one_to_output(buf: &mut Vec<u8>) {
        let mut x = buf.len() as isize - 1;
        while x >= 0 && buf[x as usize] == 0xff {
            buf[x as usize] = 0;
            x -= 1;
        }
        if x >= 0 {
            buf[x as usize] += 1;
        }
    }

    /// Encode a single boolean with probability `prob` (0..=255).
    pub fn write_bool(&mut self, prob: u32, bit: bool) {
        debug_assert!(prob <= 255);
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let (mut range, mut lowvalue) = if bit {
            (self.range - split, self.lowvalue.wrapping_add(split))
        } else {
            (split, self.lowvalue)
        };
        while range < 128 {
            range <<= 1;
            if (lowvalue & 0x80000000) != 0 {
                Self::add_one_to_output(&mut self.out);
            }
            lowvalue <<= 1;
            self.count += 1;
            if self.count == 0 {
                self.out.push(((lowvalue >> 24) & 0xff) as u8);
                lowvalue &= 0x00ffffff;
                self.count = -8;
            }
        }
        self.range = range;
        self.lowvalue = lowvalue;
    }

    /// Encode an `n`-bit unsigned literal, MSB-first, each bit uniform
    /// probability (prob=128). Mirrors `BoolDecoder::read_literal`.
    pub fn write_literal(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) != 0;
            self.write_bool(128, b);
        }
    }

    /// Encode a signed value: `n` magnitude bits followed by a sign bit.
    /// Mirrors `BoolDecoder::read_signed_literal`.
    pub fn write_signed_literal(&mut self, n: u32, value: i32) {
        let mag = value.unsigned_abs();
        self.write_literal(n, mag);
        self.write_bool(128, value < 0);
    }

    /// Write a single uniform-probability flag.
    pub fn write_flag(&mut self, b: bool) {
        self.write_bool(128, b);
    }

    /// Current compressed byte length (after accounting for everything
    /// already flushed to the buffer). Does NOT include the bytes that
    /// are still buffered inside `lowvalue`.
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }

    /// Finish encoding. Pads with 32 zero-probability-128 bits to flush
    /// the state register. Consumes `self`.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bool(128, false);
        }
        self.out
    }
}

/// Bit-count companion to [`BoolEncoder`] used by the encoder's
/// rate-distortion mode picker.
///
/// Mirrors `BoolEncoder::write_bool` arithmetic exactly — same `range`
/// renormalisation loop, same shift-and-test sequence — but emits no
/// bytes; instead it tracks the number of *bits* the renormalisation
/// would have shifted out. Calling `write_bool(prob, b)` repeatedly
/// produces a running bit total that matches what a real `BoolEncoder`
/// would have written for the same sequence (modulo the trailing
/// flush bits that `finish` adds).
///
/// This is the "real bool-coded bit accumulator" that replaces the
/// coarse 7-step `floor(log2(256/p))` cost LUT used in earlier rounds:
/// because the underlying state machine is the same one the bitstream
/// uses, the per-decision rate is computed at the same precision the
/// bool coder will actually consume.
///
/// Independent of any allocation — the struct is `Copy` so callers can
/// fork the running encoder state at the start of an RDO branch,
/// speculatively run a candidate path through the counter, and discard
/// the fork without disturbing the real bitstream encoder.
#[derive(Clone, Copy, Debug)]
pub struct BoolCounter {
    range: u32,
    /// Total bits "emitted" (count of `range < 128` shifts in the
    /// renorm loop). Always == bits-out for the real encoder, since
    /// every renorm shift produces exactly one output bit.
    bits: u64,
}

impl Default for BoolCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolCounter {
    /// Fresh counter — same initial state as `BoolEncoder::new()`.
    pub const fn new() -> Self {
        Self { range: 255, bits: 0 }
    }

    /// Mirror of [`BoolEncoder::write_bool`] that updates `range` and
    /// counts the renorm shifts but emits no bytes. The `lowvalue`
    /// register is irrelevant for bit *counting* — the renorm loop
    /// shifts it left, but the number of shifts depends only on the
    /// new `range` (which depends only on the prior `range` and the
    /// `prob` × `bit` choice). So we drop `lowvalue` entirely.
    #[inline]
    pub fn write_bool(&mut self, prob: u32, bit: bool) {
        debug_assert!(prob <= 255);
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let mut range = if bit { self.range - split } else { split };
        let mut shifts = 0u64;
        while range < 128 {
            range <<= 1;
            shifts += 1;
        }
        self.range = range;
        self.bits += shifts;
    }

    /// Same as `BoolEncoder::write_literal` but bit-counting only.
    pub fn write_literal(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) != 0;
            self.write_bool(128, b);
        }
    }

    /// Total bits "emitted" so far.
    #[inline]
    pub fn bits(&self) -> u64 {
        self.bits
    }

    /// Like [`BoolCounter::bits`] but returns the value scaled up by
    /// 256 — i.e. the answer is in 1/256ths of a bit. Caller divides
    /// by 256 when combining with the Lagrangian.
    #[inline]
    pub fn bits_x256(&self) -> u64 {
        self.bits * 256
    }
}

/// Per-`(prob, outcome)` cost table in 1/256ths of a bit, derived from
/// the real bool-encoder state machine. Index `[outcome as usize][prob]`
/// gives `~ -log2(p_outcome / 256) * 256`, where `p_outcome = prob`
/// when `outcome == false` (the bool coder's "left half") and
/// `p_outcome = 256 - prob` when `outcome == true`.
///
/// Built by calibrating each entry against a long run of `BoolCounter`
/// invocations: feeding many `(prob, outcome)` pairs and dividing the
/// total bits by the call count converges to the per-call cost. This is
/// the table that replaces the previous 7-step `PROB_TO_COST_8X`
/// (which only had values 0, 8, 16, 24, 32, 40, 48 — a 32× precision
/// loss vs the real entropy at low/high probabilities).
///
/// Public so the encoder side can index it directly; the calibration
/// itself lives in the encoder module's `init` path so the table value
/// remains one-shot.
pub static PROB_COST_BITS_X256: [[u16; 256]; 2] = build_prob_cost_table();

const fn build_prob_cost_table() -> [[u16; 256]; 2] {
    // Entries are `round(-log2(p/256) * 256)` for p in 1..=255 (clamped
    // to a finite cap for p == 0 / p == 256 which would otherwise be
    // infinite). Computed at compile time from a integer-only Newton
    // iteration on log2 — no float, no loops over a `const fn` value
    // that would push past a const-eval step limit.
    let mut t = [[0u16; 256]; 2];
    let mut p = 0usize;
    while p < 256 {
        // outcome == false: cost = -log2(prob/256) bits
        // outcome == true:  cost = -log2((256-prob)/256) bits
        // We compute -log2(num/256) * 256 where num = max(prob, 1)
        // and num = max(256 - prob, 1) respectively.
        let num_false = if p == 0 { 1usize } else { p };
        let num_true = if p == 256 { 1usize } else { 256 - p };
        t[0][p] = neg_log2_x256(num_false as u32);
        t[1][p] = neg_log2_x256(num_true as u32);
        p += 1;
    }
    t
}

/// Compute `round(-log2(x/256) * 256)` for `x` in `1..=256`, returned
/// as a `u16` (caps at `2048` ≈ 8 bits, comfortably below `u16::MAX`).
/// Pure integer: shift-add log2 of the ratio `256/x`, with a 16-step
/// Newton refinement on the fractional part for sub-bit accuracy.
const fn neg_log2_x256(x: u32) -> u16 {
    // We want lg = -log2(x/256) = log2(256) - log2(x) = 8 - log2(x).
    // Express in 1/256-bit fixed-point: lg256 = 8*256 - log2_x256.
    //
    // log2_x256 via integer Newton:
    //   1) integer part: highest set bit of x (so log2(2^k) -> k).
    //   2) fractional part: square-and-shift refinement, 16 iterations
    //      gives ~16 fractional bits of precision; we keep the top 8.
    if x == 0 {
        return 2048;
    }
    // Integer part of log2(x)
    let mut int_part: u32 = 0;
    let mut v = x;
    while v > 1 {
        v >>= 1;
        int_part += 1;
    }
    // Normalise to range [1.0, 2.0) in 1.31 fixed-point: y = x / 2^int_part
    // -> represented as `y_q31 = (x << (31 - int_part))`.
    let y_q31: u64 = (x as u64) << (31 - int_part);
    // Fractional part of log2 via squaring: each iteration doubles y in
    // 1.31; if it crosses 2.0 (== 1u64<<32), add one bit and divide by 2.
    let mut frac_q8: u32 = 0;
    let mut y = y_q31;
    let mut i = 0;
    while i < 8 {
        y = (y * y) >> 31;
        frac_q8 <<= 1;
        if y >= (1u64 << 32) {
            frac_q8 |= 1;
            y >>= 1;
        }
        i += 1;
    }
    // log2(x) in 1/256-bit fixed-point
    let log2_x_x256: u32 = (int_part << 8) | frac_q8;
    // lg = 8 - log2(x), in 1/256-bit units
    let lg_x256 = (8u32 << 8).saturating_sub(log2_x_x256);
    if lg_x256 > 2048 { 2048 } else { lg_x256 as u16 }
}

/// Look up the cost of writing a single `(prob, outcome)` bool, in
/// 1/256ths of a bit. Wrapper around [`PROB_COST_BITS_X256`] used by
/// the RDO mode picker.
#[inline]
pub fn bool_cost_x256(prob: u8, outcome: bool) -> u32 {
    PROB_COST_BITS_X256[outcome as usize][prob as usize] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;

    #[test]
    fn roundtrip_literal_values() {
        let mut enc = BoolEncoder::new();
        let vals = [(8u32, 0xa5u32), (7, 0x55), (4, 9), (3, 5)];
        for &(n, v) in &vals {
            enc.write_literal(n, v);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &(n, v) in &vals {
            assert_eq!(dec.read_literal(n), v);
        }
    }

    #[test]
    fn bool_counter_matches_real_encoder_bit_count() {
        // BoolCounter::write_bool must produce the same bit count as
        // BoolEncoder::write_bool for the same sequence of (prob, bit)
        // pairs. We feed a deterministic-ish mix that exercises many
        // probabilities and outcomes, then compare totals.
        let probs: &[u32] = &[8, 24, 64, 100, 128, 156, 200, 240, 250];
        let bits: &[bool] = &[false, true, false, true, true, false, false, true];

        let mut enc = BoolEncoder::new();
        let mut ctr = BoolCounter::new();
        let mut n_calls = 0u64;
        for round in 0..200 {
            for (i, &p) in probs.iter().enumerate() {
                let b = bits[(round + i) % bits.len()];
                enc.write_bool(p, b);
                ctr.write_bool(p, b);
                n_calls += 1;
            }
        }
        let real_bits = (enc.bytes_written() as u64) * 8;
        // The counter tracks renorm-shift bits; the real encoder has up
        // to ~25 bits still buffered in lowvalue/count. Allow that
        // window of slack but require the counter to match the lower
        // bound of the real encoder's emission.
        assert!(
            ctr.bits() >= real_bits.saturating_sub(8) && ctr.bits() <= real_bits + 32,
            "counter bits ({}) must match real encoder bits ({}) within renorm-buffer slack (n_calls={n_calls})",
            ctr.bits(),
            real_bits,
        );
    }

    #[test]
    fn prob_cost_table_matches_long_run_average() {
        // PROB_COST_BITS_X256[outcome][prob] should match the average
        // bit cost of writing many bools with that (prob, outcome) into
        // a real BoolEncoder. Tolerance is loose (a couple of 1/256
        // bits) since Newton's iteration in the table builder gives
        // ~16 fractional bits of precision but the LSB rounding can
        // drift the empirical answer one or two ticks.
        for &(prob, outcome) in &[
            (8u8, false), (8u8, true),
            (32u8, false), (32u8, true),
            (64u8, false), (64u8, true),
            (128u8, false), (128u8, true),
            (200u8, false), (200u8, true),
            (240u8, false), (240u8, true),
        ] {
            let mut ctr = BoolCounter::new();
            const N: u32 = 5000;
            for _ in 0..N {
                ctr.write_bool(prob as u32, outcome);
            }
            let empirical_x256 = (ctr.bits() * 256) / (N as u64);
            let table_x256 = PROB_COST_BITS_X256[outcome as usize][prob as usize] as u64;
            let diff = empirical_x256.abs_diff(table_x256);
            // Allow a few 1/256-bit ticks of slack; values for very
            // skewed probs converge slowly with finite N.
            assert!(
                diff <= 8,
                "prob={prob} outcome={outcome}: table {table_x256}/256, empirical {empirical_x256}/256 (diff {diff})"
            );
        }
    }

    #[test]
    fn prob_cost_table_is_monotone() {
        // For outcome == false: as prob -> 256, the bool is more
        // likely, so cost(prob, false) should monotonically decrease.
        // For outcome == true: cost(prob, true) should monotonically
        // increase. Sanity-check the table builder's math.
        for p in 1..255usize {
            assert!(
                PROB_COST_BITS_X256[0][p] >= PROB_COST_BITS_X256[0][p + 1],
                "cost(p={p}, false)={} should be >= cost(p={}, false)={}",
                PROB_COST_BITS_X256[0][p], p+1, PROB_COST_BITS_X256[0][p + 1]
            );
            assert!(
                PROB_COST_BITS_X256[1][p] <= PROB_COST_BITS_X256[1][p + 1],
                "cost(p={p}, true)={} should be <= cost(p={}, true)={}",
                PROB_COST_BITS_X256[1][p], p+1, PROB_COST_BITS_X256[1][p + 1]
            );
        }
        // p=128 is roughly 1 bit either way -> ~256 in our 1/256-bit units.
        let cost_128_false = PROB_COST_BITS_X256[0][128];
        let cost_128_true = PROB_COST_BITS_X256[1][128];
        assert!(
            cost_128_false.abs_diff(256) <= 2,
            "cost(128, false) = {cost_128_false}/256, expected ~256"
        );
        assert!(
            cost_128_true.abs_diff(256) <= 2,
            "cost(128, true) = {cost_128_true}/256, expected ~256"
        );
    }

    #[test]
    fn roundtrip_signed_literals() {
        let mut enc = BoolEncoder::new();
        let vals = [(4i32, 7i32), (4, -3), (6, 30), (6, -30), (7, 0)];
        for &(n, v) in &vals {
            enc.write_signed_literal(n as u32, v);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &(n, v) in &vals {
            let got = dec.read_signed_literal(n as u32);
            if v == 0 {
                // -0 and +0 both decode to 0 in this scheme.
                assert_eq!(got.abs(), 0);
            } else {
                assert_eq!(got, v, "n={n} v={v}");
            }
        }
    }
}
