#![no_main]

//! Fuzz: the §11 key-frame macroblock *mode-info* tree walk driven
//! *directly* off a bool-decoder partition —
//! `macroblock::parse_key_frame_macroblock_modes`.
//!
//! This is the per-macroblock side of the key-frame decode that no
//! existing target drives in isolation. The reconstruction-core target
//! `panic_free_keyframe_reconstruct` calls `frame::decode_keyframe` with
//! a grid of *already-decoded* [`MacroblockModes`] synthesised straight
//! from the fuzz bytes — it never walks the §11 mode trees. The
//! bitstream-gated targets (`panic_free_decode_keyframe`,
//! `decode_stream_token_descent`) reach `parse_key_frame_macroblock_modes`
//! only behind the §9.1 frame-tag and §19.2 coded-header validation
//! gates, so the only mode-tree probabilities they ever feed it are the
//! self-consistent ones a structurally-valid header carries. This target
//! drives the mode walk with an attacker-shaped bool partition AND an
//! attacker-shaped [`Vp8CodedHeader`], so libFuzzer can explore the
//! tree-descent surface itself.
//!
//! What the §11 walk does per macroblock, all exercised here:
//!
//! * §10 segment-id decode through the 4-leaf `MB_SEGMENT_TREE` against
//!   the three `mb_segment_tree_probs` resolved off the
//!   `update_segmentation` block (each `segment_prob` entry either an
//!   attacker byte or the §9.3-item-5 fallback to 255), gated on the
//!   `segmentation_enabled && update_mb_segmentation_map` predicate,
//! * §11.1 `mb_skip_coeff` read against an attacker `prob_skip_false`,
//!   gated on the `mb_no_skip_coeff` frame flag,
//! * §11.2 key-frame Y-mode tree walk (`read_kf_y_mode`),
//! * §11.3 / §11.5 `B_PRED` sixteen-sub-block-mode walk
//!   (`read_kf_b_mode`) with the cross-macroblock `above`/`left`
//!   sub-block-mode predictor bookkeeping (the frame-width `above`
//!   buffer and the per-row `left` column), reached whenever the Y-mode
//!   resolves to `B_PRED`,
//! * §11.4 chroma-mode tree walk (`read_kf_uv_mode`).
//!
//! The bool partition is initialised with [`BoolDecoder::init_partition`]
//! — the §20 reference's short-input fallback — so a truncated /
//! zero-length partition is a first-class input: the walk must terminate
//! cleanly (an `Err` once the partition is exhausted) rather than panic.
//!
//! Oracle: panic-freedom (no overflow, no out-of-bounds index into the
//! `above_subblock` / `left_subblock` predictor buffers, no debug-assert
//! in the tree descent) plus, on the `Ok` path, a structural check that
//! exactly `mb_rows * mb_cols` entries came back and that each decoded
//! `segment_id` (when present) is inside the documented `0..=3` envelope.
//! Every decoded field is folded into an FNV-1a accumulator so the full
//! returned vector is read.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::bool_decoder::BoolDecoder;
use oxideav_vp8::coded_header::{
    MbLfAdjustments, QuantIndices, UpdateSegmentation, Vp8CodedHeader,
};
use oxideav_vp8::macroblock::{parse_key_frame_macroblock_modes, IntraYMode};

/// Cap on the macroblock grid. 64 MBs (e.g. 8×8 or 1×64) keeps the
/// returned `Vec<MacroblockModes>` tiny while still forcing multi-row /
/// multi-column `above`/`left` predictor edges in the §11.3 B_PRED walk.
/// The walk allocates only metadata (no plane rasters), so this is well
/// inside the runner's per-iteration memory cap.
const MAX_MBS: usize = 64;

/// Forward byte cursor over the fuzz input. Readers return a defined
/// value (0) once the bytes run out so header construction stays total.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Remaining bytes from the current position — fed to the bool
    /// decoder as the §11 mode partition.
    fn rest(&self) -> &'a [u8] {
        self.data.get(self.pos..).unwrap_or(&[])
    }
}

/// An `Option<u8>`: the low bit of a flag byte decides presence, the
/// next byte is the value. Used for the §9.3 `segment_prob` entries so
/// both the attacker-byte arm and the §9.3-item-5 "fall back to 255"
/// arm are reachable.
fn opt_byte(c: &mut Cursor) -> Option<u8> {
    if c.byte() & 1 == 1 {
        Some(c.byte())
    } else {
        None
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let mut c = Cursor::new(data);

    // Grid shape: both axes forced into 1..=MAX_MBS and the product
    // capped so the returned vector stays small.
    let mb_cols = (c.byte() as usize % MAX_MBS) + 1;
    let mut mb_rows = (c.byte() as usize % MAX_MBS) + 1;
    while mb_cols * mb_rows > MAX_MBS {
        mb_rows -= 1;
    }
    if mb_rows == 0 {
        return;
    }

    // Frame-level toggles steering the §10 / §11.1 gates.
    let flags = c.byte();
    let segmentation_enabled = flags & 0b0000_0001 != 0;
    let update_mb_segmentation_map = flags & 0b0000_0010 != 0;
    let segment_feature_mode_absolute = flags & 0b0000_0100 != 0;
    let mb_no_skip_coeff = flags & 0b0000_1000 != 0;

    // §9.3 segment-tree branch probabilities — each entry an attacker
    // byte or the item-5 fallback.
    let segment_prob = [opt_byte(&mut c), opt_byte(&mut c), opt_byte(&mut c)];
    // §11.1 skip probability — present iff the frame enables the flag.
    let prob_skip_false = if mb_no_skip_coeff {
        Some(c.byte())
    } else {
        None
    };

    let update_segmentation = if segmentation_enabled {
        Some(UpdateSegmentation {
            update_mb_segmentation_map,
            update_segment_feature_data: false,
            segment_feature_mode_absolute,
            quantizer_update: [None; 4],
            loop_filter_update: [None; 4],
            segment_prob,
        })
    } else {
        None
    };

    // A key-frame coded header: only the fields the §11 mode walk reads
    // are attacker-shaped; the rest take inert key-frame values.
    let header = Vp8CodedHeader {
        color_space: Some(false),
        clamping_type: Some(false),
        segmentation_enabled,
        update_segmentation,
        filter_type: false,
        loop_filter_level: 0,
        sharpness_level: 0,
        mb_lf_adjustments: MbLfAdjustments {
            loop_filter_adj_enable: false,
            mode_ref_lf_delta_update: false,
            ref_frame_delta_update: [None; 4],
            mb_mode_delta_update: [None; 4],
        },
        log2_nbr_of_dct_partitions: 0,
        nbr_of_dct_partitions: 1,
        quant_indices: QuantIndices {
            y_ac_qi: 0,
            y_dc_delta: None,
            y2_dc_delta: None,
            y2_ac_delta: None,
            uv_dc_delta: None,
            uv_ac_delta: None,
        },
        refresh_entropy_probs: false,
        refresh_golden_frame: None,
        refresh_alternate_frame: None,
        copy_buffer_to_golden: None,
        copy_buffer_to_alternate: None,
        sign_bias_golden: None,
        sign_bias_alternate: None,
        refresh_last: None,
        token_prob_updates: [[[[None; 11]; 3]; 8]; 4],
        mb_no_skip_coeff,
        prob_skip_false,
        prob_intra: None,
        prob_last: None,
        prob_gf: None,
        intra_y_mode_prob_update: None,
        intra_uv_mode_prob_update: None,
        mv_prob_update: None,
    };

    // The §11 mode partition: the remaining attacker bytes. Use the
    // short-input-tolerant initialiser so a truncated / empty partition
    // is a valid input rather than an early bail.
    let mut dec = BoolDecoder::init_partition(c.rest());

    let modes = match parse_key_frame_macroblock_modes(&mut dec, &header, mb_rows, mb_cols) {
        Ok(m) => m,
        Err(_) => return,
    };

    // Structural invariant: one entry per macroblock, raster order.
    assert_eq!(modes.len(), mb_rows * mb_cols);

    // Fold every decoded field, and assert the §10 segment-id stays in
    // the documented 0..=3 envelope (a 4-leaf tree can only emit those).
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for m in &modes {
        if let Some(sid) = m.segment_id {
            assert!(sid <= 3, "segment_id {sid} outside 0..=3");
            acc ^= sid as u64;
        }
        acc ^= (m.mb_skip_coeff as u64) << 8;
        acc ^= (m.y_mode as u64) << 16;
        acc ^= (m.uv_mode as u64) << 24;
        if m.y_mode == IntraYMode::B {
            if let Some(sub) = m.subblock_modes {
                for (i, b) in sub.iter().enumerate() {
                    acc ^= (*b as u64).wrapping_shl(i as u32 & 31);
                }
            }
        }
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    std::hint::black_box(acc);
});
