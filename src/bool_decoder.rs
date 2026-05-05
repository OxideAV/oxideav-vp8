//! Boolean (arithmetic) decoder — RFC 6386 §7.
//!
//! VP8 uses a binary arithmetic decoder whose state is `(value, range, bit_count)`
//! plus a byte-position cursor into the compressed source. Each call to
//! `read_bool(prob)` consumes a fractional number of input bits by splitting
//! `range` into a `split = 1 + ((range - 1) * prob) / 256` partition.
//!
//! The reference implementation from RFC 6386 (section 20.2 — `dixie/bool_decoder.h`)
//! is the authoritative behaviour. The implementation below tracks that
//! reference closely.

use crate::error::{Result, Vp8Error as Error};

/// Boolean (arithmetic) decoder. Mutable cursor over a borrowed `&[u8]`.
#[derive(Debug)]
pub struct BoolDecoder<'a> {
    buf: &'a [u8],
    /// Next byte index to fetch when refilling `value`.
    pos: usize,
    /// Width of the active sub-interval, scaled up so that
    /// 128 ≤ range ≤ 255 after every renormalisation.
    range: u32,
    /// Encoded value register. Two priming bytes are loaded into the low
    /// 16 bits at construction; `read_bool` compares against
    /// `split << 8` and shifts left as it consumes bits.
    value: u32,
    /// Number of bits currently consumed within the high byte of `value`.
    bit_count: i32,
}

impl<'a> BoolDecoder<'a> {
    /// Construct a decoder positioned at the start of `buf`. Two priming
    /// bytes are loaded into `value`.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < 2 {
            return Err(Error::invalid(
                "VP8 bool decoder: need at least 2 bytes to prime",
            ));
        }
        let value = ((buf[0] as u32) << 8) | (buf[1] as u32);
        Ok(Self {
            buf,
            pos: 2,
            range: 255,
            value,
            bit_count: 0,
        })
    }

    /// Decode a single boolean using the given probability (0..=255). A
    /// probability of `prob` corresponds to a P(0) = prob / 256.
    #[inline]
    pub fn read_bool(&mut self, prob: u32) -> bool {
        debug_assert!(prob <= 255);
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let big_split = split << 8;
        let bit;
        if self.value >= big_split {
            self.range -= split;
            self.value = self.value.wrapping_sub(big_split);
            bit = true;
        } else {
            self.range = split;
            bit = false;
        }
        // Renormalise. The naive RFC 6386 §20.2 reference walks one bit
        // at a time (`while range < 128 { range<<=1; … }`) but on a real
        // CPU the shift count is `(range as u8).leading_zeros()`: range
        // sits in (0,255] after the subtraction so the u8 view captures
        // exactly the right magnitude. We batch the shift, then perform
        // at most one byte refill.
        //
        // Refill logic (matches the per-bit reference iterated `shift`
        // times): after `shift` shifts, the original `bit_count` bits
        // and the `shift` newly-shifted bits combine. If their sum hits
        // 8, the queued byte (low 8 bits of `value`) has been fully
        // consumed and the next source byte is loaded at the current
        // residual `bit_count` (i.e. `bit_count - 8` after the
        // subtraction). `shift <= 7` because `range` was in (0,255], and
        // pre-shift `bit_count <= 7`, so the sum is at most 14 — we
        // never need two refills in one call.
        let shift = (self.range as u8).leading_zeros() as i32;
        self.range <<= shift;
        self.value <<= shift;
        self.bit_count += shift;
        if self.bit_count >= 8 {
            self.bit_count -= 8;
            if self.pos < self.buf.len() {
                // Past EOF reads as zero (RFC 6386 §7.3).
                self.value |= (self.buf[self.pos] as u32) << self.bit_count;
                self.pos += 1;
            }
        }
        bit
    }

    /// Decode an unsigned integer of `n` bits (MSB first), each bit at
    /// 50/50 probability.
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | (self.read_bool(128) as u32);
        }
        v
    }

    /// Decode a signed integer: `n` magnitude bits followed by a sign bit.
    pub fn read_signed_literal(&mut self, n: u32) -> i32 {
        let mag = self.read_literal(n) as i32;
        if self.read_bool(128) {
            -mag
        } else {
            mag
        }
    }

    /// Decode a single uniform-probability bit (used inside `read_literal`
    /// expansions that prefer to expose the underlying primitive).
    pub fn read_flag(&mut self) -> bool {
        self.read_bool(128)
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Snapshot of (pos, range, value, bit_count) — used by trace
    /// harnesses to localise where the decoder loses sync against a
    /// reference trace.
    pub fn debug_state(&self) -> (usize, u32, u32, i32) {
        (self.pos, self.range, self.value, self.bit_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference encoder ported from libvpx's `vp8_encode_bool` (the
    /// loop-free, count-based variant). Used to generate test vectors for
    /// the decoder.
    struct BoolEncoder {
        out: Vec<u8>,
        range: u32,
        lowvalue: u32,
        /// Negative count of bits buffered: -24 = empty, 0 = ready to emit
        /// one byte.
        count: i32,
    }

    impl BoolEncoder {
        fn new() -> Self {
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

        fn write_bool(&mut self, prob: u32, bit: bool) {
            let split = 1 + (((self.range - 1) * prob) >> 8);
            let (mut range, mut lowvalue) = if bit {
                (self.range - split, self.lowvalue.wrapping_add(split))
            } else {
                (split, self.lowvalue)
            };
            // Renormalise one bit at a time. Mirrors the RFC 6386 §20.2
            // pseudo-code with `count = bit_count - 24` so we just check
            // `count == 0` to know when to emit.
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

        fn flush(mut self) -> Vec<u8> {
            // Pad with 32 zero bits.
            for _ in 0..32 {
                self.write_bool(128, false);
            }
            self.out
        }
    }

    #[test]
    fn roundtrip_all_true() {
        let mut enc = BoolEncoder::new();
        let n = 64;
        for _ in 0..n {
            enc.write_bool(128, true);
        }
        let buf = enc.flush();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for i in 0..n {
            let got = dec.read_bool(128);
            assert!(got, "bit {i} should be true");
        }
    }

    #[test]
    fn roundtrip_all_false() {
        let mut enc = BoolEncoder::new();
        let n = 64;
        for _ in 0..n {
            enc.write_bool(128, false);
        }
        let buf = enc.flush();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for i in 0..n {
            let got = dec.read_bool(128);
            assert!(!got, "bit {i} should be false");
        }
    }

    #[test]
    fn roundtrip_uniform_bits() {
        let mut enc = BoolEncoder::new();
        let bits: Vec<bool> = (0..256).map(|i| i & 1 == 0).collect();
        for &b in &bits {
            enc.write_bool(128, b);
        }
        let buf = enc.flush();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for (i, &expected) in bits.iter().enumerate() {
            let got = dec.read_bool(128);
            assert_eq!(got, expected, "at bit {i}");
        }
    }

    #[test]
    fn roundtrip_skewed() {
        let mut enc = BoolEncoder::new();
        let bits: Vec<(u32, bool)> = (0..512)
            .map(|i| {
                let p = ((i * 7 + 13) % 255) as u32 + 1;
                let b = i % 5 == 0;
                (p, b)
            })
            .collect();
        for &(p, b) in &bits {
            enc.write_bool(p, b);
        }
        let buf = enc.flush();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &(p, b) in &bits {
            assert_eq!(dec.read_bool(p), b);
        }
    }

    #[test]
    fn roundtrip_literals() {
        let mut enc = BoolEncoder::new();
        let vals = [0xa5u32, 0x3c, 0xff, 0x00, 0x77, 0x12];
        for &v in &vals {
            for i in (0..8).rev() {
                enc.write_bool(128, ((v >> i) & 1) != 0);
            }
        }
        let buf = enc.flush();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &v in &vals {
            assert_eq!(dec.read_literal(8), v);
        }
    }
}
