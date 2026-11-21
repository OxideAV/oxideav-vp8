//! What the rows of a wavefront share: progress they wait on, and small
//! strips of data handed from one row to the next.
//!
//! The wavefront passes of this crate (macroblock rows of an encode, a
//! decode or the loop filter) each run a couple of macroblocks behind
//! the row above. A row that catches up with the one above **blocks**:
//! it registers the progress it wants to be woken at and sleeps on a
//! condition variable; the row above wakes it when it gets there. Nobody
//! polls.
//!
//! Waking a thread costs far more than a macroblock does, so a row that
//! has to wait asks for more than it needs right now — a batch of
//! macroblocks ([`Wave::wait`]). It then runs that batch without
//! touching the lock again, and the row above pays for a wake-up once
//! per batch instead of once per macroblock. In steady state every row
//! trails the one above by a batch and nobody blocks at all.

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Condvar, Mutex};

/// How far one row has got (in macroblocks), with the progress the
/// nearest waiter wants to be woken at.
#[derive(Debug)]
pub(crate) struct Progress {
    done: AtomicUsize,
    /// `usize::MAX` while nobody waits on this row.
    wake_at: AtomicUsize,
}

impl Progress {
    pub(crate) fn new() -> Self {
        Progress {
            done: AtomicUsize::new(0),
            wake_at: AtomicUsize::new(usize::MAX),
        }
    }

    /// The published progress.
    #[inline]
    pub(crate) fn get(&self) -> usize {
        self.done.load(Ordering::Acquire)
    }
}

/// The shared state of one wavefront pass: the lock and condition
/// variable its rows sleep on, and the flag that calls the pass off.
#[derive(Debug)]
pub(crate) struct Wave {
    aborted: AtomicBool,
    gate: Mutex<()>,
    moved: Condvar,
    /// Macroblocks a waiting row asks for beyond the ones it needs.
    batch: usize,
    /// Macroblocks per row — the most a row can ask for.
    row_len: usize,
}

impl Wave {
    /// The state of a pass over rows of `row_len` macroblocks.
    pub(crate) fn new(row_len: usize) -> Self {
        Wave {
            aborted: AtomicBool::new(false),
            gate: Mutex::new(()),
            moved: Condvar::new(),
            batch: (row_len / 16).clamp(2, 16),
            row_len,
        }
    }

    fn gate(&self) -> std::sync::MutexGuard<'_, ()> {
        self.gate.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Whether the pass was called off.
    #[inline]
    pub(crate) fn aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    /// Call the pass off and wake every waiting row.
    pub(crate) fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        let _gate = self.gate();
        self.moved.notify_all();
    }

    /// Publish that `row` has finished `done` macroblocks, waking the
    /// rows that asked for that much.
    #[inline]
    pub(crate) fn advance(&self, row: &Progress, done: usize) {
        // Store, then look for a waiter; a waiter registers, then looks
        // at the progress (`wait`). Sequentially consistent, so at least
        // one side sees the other.
        row.done.store(done, Ordering::SeqCst);
        if done >= row.wake_at.load(Ordering::SeqCst) {
            self.wake(row);
        }
    }

    #[cold]
    fn wake(&self, row: &Progress) {
        let _gate = self.gate();
        row.wake_at.store(usize::MAX, Ordering::SeqCst);
        self.moved.notify_all();
    }

    /// Return once `row` has finished `need` macroblocks; `false` if the
    /// pass was called off instead.
    ///
    /// When the row is not there yet the caller sleeps until it is a
    /// batch further (or complete), so that it does not have to come
    /// back for every macroblock.
    #[inline]
    pub(crate) fn wait(&self, row: &Progress, need: usize) -> bool {
        if row.get() >= need {
            return !self.aborted();
        }
        self.sleep_until(row, (need + self.batch).min(self.row_len).max(need))
    }

    #[cold]
    fn sleep_until(&self, row: &Progress, target: usize) -> bool {
        let mut gate = self.gate();
        loop {
            if self.aborted() {
                return false;
            }
            row.wake_at.fetch_min(target, Ordering::SeqCst);
            if row.done.load(Ordering::SeqCst) >= target {
                return true;
            }
            gate = self.moved.wait(gate).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// Calls the pass off if the thread unwinds, so nobody sleeps on this
/// thread's row forever.
pub(crate) struct AbortOnUnwind<'a>(pub(crate) &'a Wave);

impl Drop for AbortOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.abort();
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

    /// A follower never observes less progress than it asked for, sleeps
    /// rather than polls, and is released by an abort.
    #[test]
    fn follower_blocks_until_the_leader_is_there() {
        const ROW: usize = 200;
        let wave = Wave::new(ROW);
        let leader = Progress::new();
        let follower = Progress::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for done in 1..=ROW {
                    assert!(wave.wait(&leader, (done + 2).min(ROW)));
                    assert!(leader.get() >= (done + 2).min(ROW));
                    wave.advance(&follower, done);
                }
            });
            for done in 1..=ROW {
                if done % 7 == 0 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                wave.advance(&leader, done);
            }
        });
        assert_eq!(follower.get(), ROW);

        // A row that never moves: the waiter is released by the abort.
        let wave = Wave::new(ROW);
        let stuck = Progress::new();
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| wave.wait(&stuck, 1));
            std::thread::sleep(std::time::Duration::from_millis(5));
            wave.abort();
            assert!(!waiter.join().expect("waiter"));
        });
        assert!(!wave.wait(&stuck, 0), "an aborted pass reports it");
    }

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
