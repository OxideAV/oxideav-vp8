//! Waiting on another worker's progress.
//!
//! The wavefront passes of this crate (macroblock rows of an encode, a
//! decode or the loop filter, each a couple of macroblocks behind the
//! row above) wait for progress that normally arrives within
//! microseconds, far below what parking a thread costs. They poll an
//! atomic counter and call [`Backoff::pause`] between polls: it spins for
//! a while and only then starts handing the processor to other threads,
//! so a short wait stays cheap on an idle machine and a long one (the
//! other worker was descheduled) does not monopolise a core.

/// Polls spent spinning before the waiter starts yielding. A poll is a
/// few nanoseconds, so this bounds the spinning to tens of microseconds
/// — longer than a macroblock row takes in any of the passes.
const SPIN_POLLS: u32 = 4096;

/// Spin-then-yield waiting state for one wait.
#[derive(Debug, Default)]
pub(crate) struct Backoff {
    polls: u32,
}

impl Backoff {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Pause before polling again.
    #[inline]
    pub(crate) fn pause(&mut self) {
        if self.polls < SPIN_POLLS {
            self.polls += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

/// A byte buffer one worker fills and others read without a lock.
///
/// The rows of a wavefront hand each other small strips of data
/// (boundary pixels, contexts) once per macroblock; a mutex around them
/// would be taken by both sides at that rate and contended constantly.
/// Here the bytes live in atomic words written and read with relaxed
/// ordering; what makes a read see a finished write is the progress
/// counter the two sides already share (stored with `Release` after the
/// write, loaded with `Acquire` before the read). Offsets and lengths
/// are in bytes and must be multiples of eight.
#[derive(Debug)]
pub(crate) struct SharedBytes {
    words: Vec<AtomicU64>,
}

impl SharedBytes {
    /// `len` zero bytes.
    pub(crate) fn new(len: usize) -> Self {
        debug_assert!(len.is_multiple_of(8));
        SharedBytes {
            words: (0..len / 8).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// A buffer holding a copy of `bytes`.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let shared = Self::new(bytes.len());
        shared.store(0, bytes);
        shared
    }

    /// Write `bytes` at byte offset `at`.
    #[inline]
    pub(crate) fn store(&self, at: usize, bytes: &[u8]) {
        debug_assert!(at.is_multiple_of(8) && bytes.len().is_multiple_of(8));
        let (chunks, _) = bytes.as_chunks::<8>();
        for (word, chunk) in self.words[at / 8..].iter().zip(chunks) {
            word.store(u64::from_le_bytes(*chunk), Ordering::Relaxed);
        }
    }

    /// Read `out.len()` bytes from byte offset `at`.
    #[inline]
    pub(crate) fn load(&self, at: usize, out: &mut [u8]) {
        debug_assert!(at.is_multiple_of(8) && out.len().is_multiple_of(8));
        let (chunks, _) = out.as_chunks_mut::<8>();
        for (word, chunk) in self.words[at / 8..].iter().zip(chunks) {
            *chunk = word.load(Ordering::Relaxed).to_le_bytes();
        }
    }
}

/// One §13.3 non-zero context ([`MbEntropyCtx`]) per macroblock column,
/// shared the same way as [`SharedBytes`].
///
/// [`MbEntropyCtx`]: crate::dct_tokens::MbEntropyCtx
#[derive(Debug)]
pub(crate) struct SharedEntropy {
    flags: Vec<AtomicU16>,
}

impl SharedEntropy {
    pub(crate) fn new(mb_cols: usize) -> Self {
        SharedEntropy {
            flags: (0..mb_cols).map(|_| AtomicU16::new(0)).collect(),
        }
    }

    #[inline]
    pub(crate) fn store(&self, mb_col: usize, ctx: &crate::dct_tokens::MbEntropyCtx) {
        let mut bits = 0u16;
        for (slot, &nonzero) in ctx.nonzero.iter().enumerate() {
            bits |= (nonzero as u16) << slot;
        }
        self.flags[mb_col].store(bits, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn load(&self, mb_col: usize) -> crate::dct_tokens::MbEntropyCtx {
        let bits = self.flags[mb_col].load(Ordering::Relaxed);
        let mut ctx = crate::dct_tokens::MbEntropyCtx::default();
        for (slot, nonzero) in ctx.nonzero.iter_mut().enumerate() {
            *nonzero = bits >> slot & 1 != 0;
        }
        ctx
    }
}

/// The bottom pixel line of a macroblock row's reconstruction (luma and
/// both chroma planes), published macroblock by macroblock for the intra
/// prediction of the row below.
#[derive(Debug)]
pub(crate) struct SharedPixelLine {
    y: SharedBytes,
    u: SharedBytes,
    v: SharedBytes,
}

impl SharedPixelLine {
    pub(crate) fn new(mb_cols: usize) -> Self {
        SharedPixelLine {
            y: SharedBytes::new(mb_cols * 16),
            u: SharedBytes::new(mb_cols * 8),
            v: SharedBytes::new(mb_cols * 8),
        }
    }

    /// Publish the bottom line of the macroblock at `mb_col`.
    #[inline]
    pub(crate) fn store(&self, mb_col: usize, recon: &crate::reconstruct::ReconstructedMb) {
        self.y.store(mb_col * 16, &recon.y[240..256]);
        self.u.store(mb_col * 8, &recon.u[56..64]);
        self.v.store(mb_col * 8, &recon.v[56..64]);
    }

    /// Fill the above-row fields of `neighbors` for the macroblock at
    /// `mb_col` of the row below: the above strips, the §12.3 above-right
    /// extension and the corner samples. The row above must have
    /// published macroblocks `..= mb_col + 1` (or its whole row).
    pub(crate) fn load_above(
        &self,
        mb_col: usize,
        mb_cols: usize,
        neighbors: &mut crate::reconstruct::MbNeighbors,
    ) {
        // The line around this macroblock: the eight samples before it
        // (for the corner), its own, and the eight after (for the
        // above-right extension).
        let (yx, cx) = (mb_col * 16, mb_col * 8);
        let mut y_line = [0u8; 32];
        let from = yx.saturating_sub(8);
        let to = (yx + 24).min(mb_cols * 16);
        self.y.load(from, &mut y_line[from + 8 - yx..to + 8 - yx]);
        let mut u_line = [0u8; 16];
        let mut v_line = [0u8; 16];
        let cfrom = cx.saturating_sub(8);
        self.u.load(cfrom, &mut u_line[cfrom + 8 - cx..]);
        self.v.load(cfrom, &mut v_line[cfrom + 8 - cx..]);

        let mut y_above = [0u8; 16];
        y_above.copy_from_slice(&y_line[8..24]);
        neighbors.y_above = Some(y_above);
        let mut u_above = [0u8; 8];
        u_above.copy_from_slice(&u_line[8..]);
        neighbors.u_above = Some(u_above);
        let mut v_above = [0u8; 8];
        v_above.copy_from_slice(&v_line[8..]);
        neighbors.v_above = Some(v_above);
        // §12.3: the rightmost macroblock repeats its own (-1, 15).
        let mut above_right = [y_line[23]; 4];
        if mb_col + 1 < mb_cols {
            above_right.copy_from_slice(&y_line[24..28]);
        }
        neighbors.y_above_right = Some(above_right);
        if mb_col == 0 {
            // §20.14: left of the frame, below the top row → 129.
            let left = crate::intra_predict::DEFAULT_LEFT_PIXEL;
            neighbors.y_topleft = Some(left);
            neighbors.u_topleft = Some(left);
            neighbors.v_topleft = Some(left);
        } else {
            neighbors.y_topleft = Some(y_line[7]);
            neighbors.u_topleft = Some(u_line[7]);
            neighbors.v_topleft = Some(v_line[7]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_bytes_round_trip() {
        let source: Vec<u8> = (0..64u8).map(|b| b.wrapping_mul(37)).collect();
        let shared = SharedBytes::from_bytes(&source);
        let mut out = [0u8; 64];
        shared.load(0, &mut out);
        assert_eq!(&out[..], &source[..]);
        shared.store(16, &[9u8; 8]);
        let mut part = [0u8; 24];
        shared.load(8, &mut part);
        assert_eq!(&part[..8], &source[8..16]);
        assert_eq!(&part[8..16], &[9u8; 8]);
        assert_eq!(&part[16..], &source[24..32]);
    }

    #[test]
    fn shared_entropy_round_trip() {
        let shared = SharedEntropy::new(3);
        for pattern in 0..512u16 {
            let mut ctx = crate::dct_tokens::MbEntropyCtx::default();
            for (slot, nonzero) in ctx.nonzero.iter_mut().enumerate() {
                *nonzero = pattern >> slot & 1 != 0;
            }
            shared.store(1, &ctx);
            assert_eq!(shared.load(1), ctx);
            assert_eq!(shared.load(0), crate::dct_tokens::MbEntropyCtx::default());
        }
    }
}
