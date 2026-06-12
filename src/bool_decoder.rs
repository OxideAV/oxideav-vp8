//! VP8 boolean (range) entropy decoder.
//!
//! This module implements the decoder side of the VP8 boolean coder
//! described in RFC 6386 §7. Essentially the entire VP8 elementary
//! stream is encoded as a sequence of boolean values processed by this
//! coder; every higher-level decode step (frame header, macroblock
//! mode, motion vectors, DCT tokens) is built on top of it.
//!
//! The coder is a binary arithmetic decoder. State is just two
//! 8-bit-domain registers (`range` and `value`) plus a single-bit
//! shift counter and an input cursor. Decoding one bool requires only
//! a comparison and an optional renormalisation that shifts new input
//! bits in 8 at a time.
//!
//! See RFC 6386 §7.2 (algorithm description) and §7.3 (reference
//! C pseudocode) for the authoritative specification.
//!
//! # Status
//!
//! Foundational primitive only. No frame header / macroblock decode is
//! wired up against this yet — see the per-method docs and the
//! crate-level scaffold notes.

use core::fmt;

/// Out-of-band conditions surfaced by [`BoolDecoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolDecoderError {
    /// Initialisation requested but fewer than two bytes of input are
    /// available. The decoder reads two bytes into `value` during
    /// `init` (RFC 6386 §7.3 `init_bool_decoder`) and cannot start
    /// short of that.
    InputTooShort,
    /// Renormalisation tried to pull a fresh byte but the input cursor
    /// has already passed the end of the partition. The spec relies on
    /// the encoder having appended enough flushed bytes to satisfy any
    /// reads the decoder issues; we surface this as an explicit error
    /// rather than silently returning zero bits, so callers can
    /// distinguish a truncated stream from a legitimate trailing
    /// boolean.
    EndOfStream,
}

impl fmt::Display for BoolDecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolDecoderError::InputTooShort => {
                write!(f, "vp8 bool decoder: input shorter than 2 bytes")
            }
            BoolDecoderError::EndOfStream => {
                write!(f, "vp8 bool decoder: read past end of partition")
            }
        }
    }
}

impl std::error::Error for BoolDecoderError {}

/// VP8 boolean entropy decoder state per RFC 6386 §7.3
/// (`bool_decoder`).
///
/// The decoder owns the input slice for its partition and walks it
/// from front to back. Eight bits at a time are shifted into the
/// `value` register during renormalisation; consumers never touch the
/// underlying bytes directly.
///
/// # Invariants
///
/// After [`BoolDecoder::init`] returns `Ok(_)` and between any pair of
/// successful [`BoolDecoder::read_bool`] / [`BoolDecoder::read_literal`]
/// calls:
///
/// * `range` is in `128..=255` (RFC 6386 §7.2);
/// * `value` is `< (range << 8)` interpreted with the unread tail of
///   the partition appended on the right.
#[derive(Debug, Clone)]
pub struct BoolDecoder<'a> {
    /// Remaining partition input; the next byte read by
    /// renormalisation is `input[0]`.
    input: &'a [u8],
    /// `range` field of RFC 6386 §7.3 `bool_decoder`. Maintained in
    /// `128..=255` outside renormalisation. We hold it in `u32` to
    /// match the reference pseudocode's widening behaviour during
    /// renormalisation; only the low 8 bits ever carry information.
    range: u32,
    /// `value` field of RFC 6386 §7.3 `bool_decoder`. Two-byte window
    /// over the not-yet-consumed bool stream.
    value: u32,
    /// `bit_count` field of RFC 6386 §7.3 `bool_decoder`: the number
    /// of bits already shifted out of `value` since the last full byte
    /// was loaded. Range `0..=7`.
    bit_count: u32,
}

impl<'a> BoolDecoder<'a> {
    /// Initialise the decoder from the start of a partition.
    ///
    /// Per RFC 6386 §7.3 `init_bool_decoder` the first two bytes of
    /// the partition are loaded into `value`, `range` starts at 255,
    /// and the bit-shift counter is zero.
    pub fn init(input: &'a [u8]) -> Result<Self, BoolDecoderError> {
        if input.len() < 2 {
            return Err(BoolDecoderError::InputTooShort);
        }
        let first = input[0] as u32;
        let second = input[1] as u32;
        let value = (first << 8) | second;
        Ok(BoolDecoder {
            input: &input[2..],
            range: 255,
            value,
            bit_count: 0,
        })
    }

    /// Initialise the decoder from a possibly-undersized partition.
    ///
    /// Mirrors the §20 reference's `init_bool_decoder`, which tolerates
    /// `sz < 2` by zero-initialising the value register and presenting an
    /// empty input. Tiny VP8 frames where the DCT partition rounds down
    /// to a 0- or 1-byte slice can land here: the surrounding macroblock
    /// layer reads only `mb_skip = 1` flags from this partition, which
    /// is satisfiable from the `value = 0` initial state.
    ///
    /// Callers reading the §19.2 control partition or the §9.1 frame
    /// header should keep using [`BoolDecoder::init`], which rejects
    /// truncated input as an error (every valid VP8 frame's control
    /// partition is at least 2 bytes).
    pub fn init_partition(input: &'a [u8]) -> Self {
        if input.len() >= 2 {
            let first = input[0] as u32;
            let second = input[1] as u32;
            let value = (first << 8) | second;
            BoolDecoder {
                input: &input[2..],
                range: 255,
                value,
                bit_count: 0,
            }
        } else {
            // §20 reference: tiny partition → value = 0, no buffered input.
            BoolDecoder {
                input: &[],
                range: 255,
                value: 0,
                bit_count: 0,
            }
        }
    }

    /// Read one boolean with probability `prob / 256` of being 0.
    ///
    /// Returns `Ok(false)` for a coded zero and `Ok(true)` for a coded
    /// one. `prob` must lie in `1..=255` per the encoder's contract
    /// (RFC 6386 §7.2); a probability of zero or 256 has no meaning
    /// because `split` would degenerate.
    #[inline]
    pub fn read_bool(&mut self, prob: u8) -> Result<bool, BoolDecoderError> {
        // split is approximately (range * prob) / 256 and is strictly
        // greater than 0 and strictly less than range.
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        // The comparison is between `value` and split << 8 because
        // `value` lives in the upper byte of the 16-bit interval.
        let big_split = split << 8;

        let retval = if self.value >= big_split {
            // Coded a one: subtract the left endpoint of the interval
            // and reduce range.
            self.value -= big_split;
            self.range -= split;
            true
        } else {
            // Coded a zero: reduce range, leave value alone.
            self.range = split;
            false
        };

        // Renormalise: double range/value until range is back in the
        // 128..=255 band, pulling a fresh input byte whenever we run
        // out of buffered bits.
        self.renormalize()?;
        Ok(retval)
    }

    /// Read an unsigned `num_bits`-wide literal — `num_bits` flag bits
    /// at fixed probability 128, MSB first.
    ///
    /// Per the RFC 6386 §7.3 `read_literal` convenience function, the
    /// returned value is the integer assembled left-to-right from the
    /// individual flag reads. `num_bits` must be in `0..=32`; values
    /// outside that range are rejected via the `bit_count` field's
    /// natural width.
    pub fn read_literal(&mut self, num_bits: u32) -> Result<u32, BoolDecoderError> {
        debug_assert!(num_bits <= 32, "vp8 read_literal: num_bits must be <= 32");
        let mut v: u32 = 0;
        for _ in 0..num_bits {
            let bit = self.read_bool(128)? as u32;
            v = (v << 1) | bit;
        }
        Ok(v)
    }

    /// Read a signed `num_bits`-wide literal. A leading sign bit
    /// (probability 128) is followed by `num_bits - 1` magnitude bits
    /// when `num_bits > 0`; `num_bits == 0` returns 0. Per the
    /// reference pseudocode the sign is realised by initialising the
    /// accumulator to -1 when the sign bit is set.
    pub fn read_signed_literal(&mut self, num_bits: u32) -> Result<i32, BoolDecoderError> {
        debug_assert!(
            num_bits <= 31,
            "vp8 read_signed_literal: num_bits must be <= 31"
        );
        if num_bits == 0 {
            return Ok(0);
        }
        let sign = self.read_bool(128)?;
        let mut v: i32 = if sign { -1 } else { 0 };
        for _ in 1..num_bits {
            let bit = self.read_bool(128)? as i32;
            v = (v << 1) | bit;
        }
        Ok(v)
    }

    /// Renormalise `range` and `value` until `range >= 128`, pulling
    /// input bytes as needed.
    ///
    /// This is the inner loop of `read_bool` (RFC 6386 §7.3) lifted
    /// out so it can be reused in `init`-time tests.
    ///
    /// The §7.3 listing doubles `range` / `value` one bit at a time
    /// until `range` is back in `128..=255`. The number of doublings is
    /// a pure function of `range` (which is in `1..=255` after a
    /// `read_bool` interval split), so the loop is collapsed here into
    /// a single shift computed from `range.leading_zeros()`. At most
    /// one fresh input byte can be needed per renormalisation — the
    /// shift caps at 7 and `bit_count` is at most 7, so the §7.3
    /// "shift counter reaches 8" byte pull fires at most once — and it
    /// lands in `value` at bit offset `bit_count + shift - 8`, exactly
    /// where the bit-at-a-time listing accumulates it (the byte is
    /// OR-ed in when the counter wraps, then shifted up by the
    /// doublings still outstanding). The `EndOfStream` trigger
    /// condition is unchanged: a byte is required iff
    /// `bit_count + shift >= 8`. The batched form is bit-exact against
    /// the §7.3 listing on every decode
    /// (`batched_renormalize_matches_bit_at_a_time_listing`).
    #[inline]
    fn renormalize(&mut self) -> Result<(), BoolDecoderError> {
        // `range` is in 1..=255 here; bringing it into 128..=255 takes
        // exactly its leading-zero count above bit 7 (0..=7 doublings).
        let shift = self.range.leading_zeros() - 24;
        if shift == 0 {
            return Ok(());
        }
        self.value <<= shift;
        self.range <<= shift;
        let bit_count = self.bit_count + shift;
        if bit_count >= 8 {
            let (next, rest) = match self.input.split_first() {
                Some(pair) => pair,
                None => return Err(BoolDecoderError::EndOfStream),
            };
            self.bit_count = bit_count - 8;
            self.value |= (*next as u32) << self.bit_count;
            self.input = rest;
        } else {
            self.bit_count = bit_count;
        }
        Ok(())
    }

    /// Current `range` register. Exposed for testing / auditing only.
    pub fn range(&self) -> u32 {
        self.range
    }

    /// Current `value` register. Exposed for testing / auditing only.
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Bytes of the partition not yet shifted into `value`.
    pub fn remaining_input(&self) -> &[u8] {
        self.input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 16-bit boolean encoder used only inside this test module to
    // produce ground-truth byte sequences. It mirrors the encoder
    // described in RFC 6386 §7.2: the same "split = 1 + ((range - 1)
    // * prob) >> 8" arithmetic; the same renormalise-while-range<128
    // loop. We need an encoder to exercise the decoder against; the
    // alternative would be hand-rolling expected bytes for every test
    // case which is much more error-prone.
    //
    // This encoder lives ONLY in tests and is NOT exported. The
    // production codepath here is decode-only.
    struct TestEncoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl TestEncoder {
        fn new() -> Self {
            TestEncoder {
                out: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn write_bool(&mut self, prob: u8, val: bool) {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            if val {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                // Propagate carry into already-written bytes if the
                // top bit of `bottom` is set. We replicate the
                // reference encoder's `add_one_to_output` helper here.
                if (self.bottom >> 31) & 1 == 1 {
                    let mut i = self.out.len();
                    while i > 0 {
                        i -= 1;
                        if self.out[i] == 255 {
                            self.out[i] = 0;
                        } else {
                            self.out[i] = self.out[i].wrapping_add(1);
                            break;
                        }
                    }
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    let byte = (self.bottom >> 24) as u8;
                    self.out.push(byte);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn write_literal(&mut self, value: u32, num_bits: u32) {
            for i in (0..num_bits).rev() {
                let bit = ((value >> i) & 1) == 1;
                self.write_bool(128, bit);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            // Flush per RFC 6386 §7.3 `flush_bool_encoder`. Note the
            // bit_count field is treated as a signed int in the
            // reference C.
            let c = self.bit_count;
            let mut v = self.bottom;
            if v & (1u32 << (32 - c)) != 0 {
                let mut i = self.out.len();
                while i > 0 {
                    i -= 1;
                    if self.out[i] == 255 {
                        self.out[i] = 0;
                    } else {
                        self.out[i] = self.out[i].wrapping_add(1);
                        break;
                    }
                }
            }
            v <<= c & 7;
            let mut c_shift = c >> 3;
            while c_shift > 0 {
                v <<= 8;
                c_shift -= 1;
            }
            for _ in 0..4 {
                self.out.push((v >> 24) as u8);
                v <<= 8;
            }
            self.out
        }
    }

    #[test]
    fn init_rejects_short_input() {
        assert_eq!(
            BoolDecoder::init(&[]).unwrap_err(),
            BoolDecoderError::InputTooShort
        );
        assert_eq!(
            BoolDecoder::init(&[0xAA]).unwrap_err(),
            BoolDecoderError::InputTooShort
        );
    }

    #[test]
    fn init_loads_first_two_bytes_into_value() {
        let d = BoolDecoder::init(&[0xAB, 0xCD, 0x00]).unwrap();
        // value = (0xAB << 8) | 0xCD = 0xABCD
        assert_eq!(d.value(), 0xABCD);
        assert_eq!(d.range(), 255);
        assert_eq!(d.remaining_input(), &[0x00]);
    }

    #[test]
    fn roundtrip_alternating_bits_at_p128() {
        // Probability 128 means each bit is a coin flip and the
        // encoder's renormalisation triggers on every write. Good
        // exercise of the byte-pulling path.
        let bits = [true, false, true, true, false, false, true, false];
        let mut enc = TestEncoder::new();
        for &b in &bits {
            enc.write_bool(128, b);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        for &expected in &bits {
            let got = dec.read_bool(128).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn roundtrip_biased_probabilities() {
        // Sweep both extremes of the legal probability range to be
        // sure the renormalise-many-times-then-renormalise-not-at-all
        // paths both work.
        let probs: &[u8] = &[1, 16, 64, 128, 200, 240, 255];
        let mut enc = TestEncoder::new();
        // Write a value=true and value=false at every probability.
        for &p in probs {
            enc.write_bool(p, false);
            enc.write_bool(p, true);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        for &p in probs {
            assert!(!dec.read_bool(p).unwrap(), "prob={p}");
            assert!(dec.read_bool(p).unwrap(), "prob={p}");
        }
    }

    #[test]
    fn roundtrip_literal_msb_first() {
        // A 12-bit literal should round-trip unchanged and the bits
        // should come out MSB-first per RFC 6386 §7.3.
        let mut enc = TestEncoder::new();
        enc.write_literal(0xABC, 12);
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        assert_eq!(dec.read_literal(12).unwrap(), 0xABC);
    }

    #[test]
    fn signed_literal_zero_width_returns_zero() {
        // RFC 6386 §7.3 read_signed_literal returns 0 immediately
        // when num_bits == 0 — no bits are read from the stream.
        let mut enc = TestEncoder::new();
        enc.write_bool(128, true); // arbitrary trailing data
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        assert_eq!(dec.read_signed_literal(0).unwrap(), 0);
        // The trailing bool is still available.
        assert!(dec.read_bool(128).unwrap());
    }

    #[test]
    fn signed_literal_roundtrip() {
        // Encoder side: the spec defines the signed form as "sign bit
        // then magnitude" but the *decoded* int is the result of
        // accumulating with v = -1 when the sign is set. So encoding
        // -3 (5-bit) under that convention is: sign=1 followed by the
        // four bits that, when accumulated into v=-1 (0xFFFF_FFFF) as
        // (v << 1) | bit, produce -3. Working backwards from -3 == 0b...11101:
        //   step 0: v=-1     (sign=1)
        //   step 1: v=(v<<1)|b1 = ... so we need bits 1,1,0,1
        // Encode that bit pattern as the magnitude.
        let mut enc = TestEncoder::new();
        enc.write_bool(128, true); // sign
        enc.write_bool(128, true);
        enc.write_bool(128, true);
        enc.write_bool(128, false);
        enc.write_bool(128, true);
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        assert_eq!(dec.read_signed_literal(5).unwrap(), -3);
    }

    #[test]
    fn end_of_stream_surfaced() {
        // Only 2 bytes available. After init, value=0 and range=255.
        // Reading at prob=1: split = 1 + ((254 * 1) >> 8) = 1, so
        // value(0) < (split<<8 == 256), retval=0, range=1.
        // Renormalisation then needs to shift range from 1 up to 128
        // — seven doublings, each pulling a new bit from value (also
        // zero) and incrementing bit_count. After 8 increments the
        // loop tries to load a fresh input byte; with no input left,
        // we must surface EndOfStream rather than silently reading
        // garbage.
        let mut dec = BoolDecoder::init(&[0x00, 0x00]).unwrap();
        // First read renormalises only as much as the buffered
        // bit_count allows; it should succeed. The second read forces
        // a fresh byte pull.
        let _ = dec.read_bool(1).unwrap();
        let err = dec.read_bool(1).unwrap_err();
        assert_eq!(err, BoolDecoderError::EndOfStream);
    }

    /// Reference §7.3 `read_bool`: the same interval split as the
    /// production method, but with the renormalisation written as the
    /// literal bit-at-a-time listing (double `range`/`value`, count
    /// shifted bits, pull a fresh byte when the counter reaches 8).
    /// The production `renormalize` collapses that loop into a single
    /// `leading_zeros`-derived shift; this is the ground truth it must
    /// match state-for-state.
    fn read_bool_reference(d: &mut BoolDecoder<'_>, prob: u8) -> Result<bool, BoolDecoderError> {
        let split = 1 + (((d.range - 1) * prob as u32) >> 8);
        let big_split = split << 8;
        let retval = if d.value >= big_split {
            d.value -= big_split;
            d.range -= split;
            true
        } else {
            d.range = split;
            false
        };
        while d.range < 128 {
            d.value <<= 1;
            d.range <<= 1;
            d.bit_count += 1;
            if d.bit_count == 8 {
                d.bit_count = 0;
                let (next, rest) = match d.input.split_first() {
                    Some(pair) => pair,
                    None => return Err(BoolDecoderError::EndOfStream),
                };
                d.value |= *next as u32;
                d.input = rest;
            }
        }
        Ok(retval)
    }

    #[test]
    fn batched_renormalize_matches_bit_at_a_time_listing() {
        // Drive the production decoder and the bit-at-a-time reference
        // in lockstep over a probability sweep that exercises every
        // renormalisation depth (shift 0 through 7, byte pulls landing
        // at every bit_count offset) and assert the full decoder state
        // matches after every read.
        let mut enc = TestEncoder::new();
        let mut probs = Vec::new();
        let mut bits = Vec::new();
        let mut seed = 0x2545f491u32;
        for _ in 0..4096 {
            // xorshift32 — deterministic prob/bit stream.
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let p = (seed % 254 + 1) as u8;
            let b = (seed >> 16) & 1 == 1;
            probs.push(p);
            bits.push(b);
            enc.write_bool(p, b);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        let mut dec_ref = BoolDecoder::init(&buf).unwrap();
        for (i, (&p, &expected)) in probs.iter().zip(bits.iter()).enumerate() {
            let got = dec.read_bool(p).unwrap();
            let got_ref = read_bool_reference(&mut dec_ref, p).unwrap();
            assert_eq!(got, expected, "decoded bit {i} (prob {p})");
            assert_eq!(got, got_ref, "reference disagreement at bit {i}");
            assert_eq!(dec.range, dec_ref.range, "range diverged at bit {i}");
            assert_eq!(dec.value, dec_ref.value, "value diverged at bit {i}");
            assert_eq!(
                dec.bit_count, dec_ref.bit_count,
                "bit_count diverged at bit {i}"
            );
            assert_eq!(dec.input, dec_ref.input, "input cursor diverged at bit {i}");
        }
    }

    #[test]
    fn range_invariant_holds_across_reads() {
        // After every successful read_bool, range must be in 128..=255
        // per RFC 6386 §7.2.
        let mut enc = TestEncoder::new();
        for i in 0..64 {
            enc.write_bool(((i * 13 + 7) % 254 + 1) as u8, (i & 1) == 1);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::init(&buf).unwrap();
        for i in 0..64 {
            let _ = dec.read_bool(((i * 13 + 7) % 254 + 1) as u8).unwrap();
            assert!(
                (128..=255).contains(&dec.range()),
                "range out of band after read {i}: {}",
                dec.range()
            );
        }
    }
}
