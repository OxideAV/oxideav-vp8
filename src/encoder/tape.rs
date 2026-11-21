//! The §13.3 token decisions of a macroblock row, recorded for a later
//! arithmetic-coding pass.
//!
//! The mode search of a frame prices tokens against one probability
//! table, but the table that ships can be fitted to the frame's own
//! statistics (§13.4) only once every decision is known. So the search
//! records *which* coefficient-tree probability each decision is coded
//! against (its position in the `[4][8][3][11]` table) instead of the
//! probability value; the tape is then tallied into branch counts, the
//! updates are fitted, and the tape is coded against the fitted table —
//! one search, no second pass.

use super::{BoolEncoder, BoolSink, BranchCounts, CoeffProbs};

/// §13.3 token decisions of one macroblock row, in coding order.
///
/// Each entry holds the decision's bit in bit 15 and, in bits 0..=11,
/// where its probability is found in a [`ReplayProbs`] table: a
/// coefficient-tree decision carries the flat index of its probability
/// in the `[4][8][3][11]` table (below [`LITERAL_BASE`]); a decision at
/// a fixed probability (extra bits, sign) carries `LITERAL_BASE` plus
/// that probability.
#[derive(Debug, Default, Clone)]
pub(super) struct TokenTape(Vec<u16>);

const TAPE_BIT: u16 = 0x8000;
const TAPE_INDEX_MASK: u16 = 0x0FFF;
/// First index of the fixed probabilities in a [`ReplayProbs`] table.
const LITERAL_BASE: u16 = 0x0800;

/// The probability of every tape entry in one table: the frame's
/// coefficient probabilities, then (from [`LITERAL_BASE`]) the 256 fixed
/// probabilities at their own value — so coding an entry is one lookup,
/// whichever kind it is.
struct ReplayProbs([u8; LITERAL_BASE as usize + 256]);

impl ReplayProbs {
    fn new(probs: &CoeffProbs) -> Self {
        let mut table = [0u8; LITERAL_BASE as usize + 256];
        let flat = probs.as_flattened().as_flattened().as_flattened();
        table[..flat.len()].copy_from_slice(flat);
        for (p, slot) in table[LITERAL_BASE as usize..].iter_mut().enumerate() {
            *slot = p as u8;
        }
        ReplayProbs(table)
    }
}

impl BoolSink for TokenTape {
    #[inline]
    fn put(&mut self, prob: u8, value: bool) {
        self.0
            .push(LITERAL_BASE | prob as u16 | if value { TAPE_BIT } else { 0 });
    }

    #[inline]
    fn put_tree(&mut self, slot: u16, _prob: u8, value: bool) {
        debug_assert!(slot < LITERAL_BASE);
        self.0.push(slot | if value { TAPE_BIT } else { 0 });
    }
}

impl TokenTape {
    /// An empty tape with room for `decisions` entries.
    pub(super) fn with_capacity(decisions: usize) -> Self {
        TokenTape(Vec::with_capacity(decisions))
    }

    /// Number of recorded decisions.
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    /// Forget the recorded decisions, keeping the storage.
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    /// Code the tape into `enc` with the coefficient probabilities
    /// `probs`.
    pub(super) fn replay(&self, enc: &mut BoolEncoder, probs: &CoeffProbs) {
        let table = ReplayProbs::new(probs);
        // Every index a tape holds is below the table's length; the mask
        // keeps the lookup in bounds without a check per decision.
        let table: &[u8] = &table.0;
        for &d in &self.0 {
            let index = (d & TAPE_INDEX_MASK) as usize;
            let prob = table.get(index).copied().unwrap_or(128);
            enc.write_bool(prob, d & TAPE_BIT != 0);
        }
    }

    /// Add the tape's coefficient-tree decisions to `counts`.
    pub(super) fn tally(&self, counts: &mut BranchCounts) {
        let flat = counts
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut();
        for &d in &self.0 {
            if d & LITERAL_BASE == 0 {
                let slot = &mut flat[(d & TAPE_INDEX_MASK) as usize];
                let bit = (d >> 15) as u32;
                slot.0 += 1 - bit;
                slot.1 += bit;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tape codes the bytes the direct calls code, and tallies exactly
    /// its coefficient-tree decisions.
    #[test]
    fn tape_replays_and_tallies_what_was_recorded() {
        let probs = crate::dct_tokens::DEFAULT_COEFF_PROBS;
        let flat = probs.as_flattened().as_flattened().as_flattened();
        let mut tape = TokenTape::with_capacity(0);
        let mut direct = BoolEncoder::new();
        let mut want = super::super::empty_branch_counts();
        let mut lcg = 0x1357_9BDFu32;
        for n in 0..20_000usize {
            lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let value = (lcg >> 9) & 1 != 0;
            if n % 3 == 0 {
                let prob = ((lcg >> 16) as u8).max(1);
                tape.put(prob, value);
                direct.write_bool(prob, value);
            } else {
                let slot = (lcg >> 12) as usize % flat.len();
                tape.put_tree(slot as u16, flat[slot], value);
                direct.write_bool(flat[slot], value);
                let cell = &mut want
                    .as_flattened_mut()
                    .as_flattened_mut()
                    .as_flattened_mut()[slot];
                if value {
                    cell.1 += 1;
                } else {
                    cell.0 += 1;
                }
            }
        }
        let mut replayed = BoolEncoder::new();
        tape.replay(&mut replayed, &probs);
        assert_eq!(replayed.finish(), direct.finish());
        let mut got = super::super::empty_branch_counts();
        tape.tally(&mut got);
        assert_eq!(got, want);
        tape.clear();
        let mut empty = super::super::empty_branch_counts();
        tape.tally(&mut empty);
        assert_eq!(empty, super::super::empty_branch_counts());
    }
}
