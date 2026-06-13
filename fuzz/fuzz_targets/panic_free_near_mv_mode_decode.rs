#![no_main]

//! Fuzz: panic-freedom of the public §16 inter-MB *mode-info decode*
//! surface (RFC 6386 §16.2 / §16.3 / §16.4 / §17 / §18 / §20.11).
//!
//! Surface coverage — every entry point reached directly from attacker
//! bytes, with adversarial neighbour records, sign-bias flags, MV
//! probability contexts, reference planes, and a bool-coder partition:
//!
//! * `decode_inter_mb(dec, reference, mb_col, mb_row, above, left,
//!   aboveleft, current_ref, sign_bias, mv_contexts, full_pixel,
//!   filters, mb_skip_coeff, y2/y/u/v_dequant)` — the §16.2/§16.3/§18
//!   whole-MB integration entry point. Runs the §16.3 census
//!   (`find_near_mvs`) + §16.2 inter-mode tree (`read_inter_mode`) +
//!   §18.1 clamp (`clamp_mv`) + §17 NEWMV differential (`read_mv` via
//!   `resolve_inter_mb_mv`), then drives `reconstruct_inter_mb`; a
//!   `SPLITMV` resolution surfaces `InterMbError::SplitNotSupported`.
//! * `decode_split_mv_mb(dec, reference, …)` — the §16.4 SPLITMV
//!   analogue: census + tree, asserts SPLITMV, then the partition walk
//!   ([`decode_split_mv`] → `read_mv_partition` / `submv_ref` /
//!   neighbour-MV lookups) + `reconstruct_split_mv_mb`. Mis-dispatch
//!   (whole-MB mode reaching the split entry) surfaces a clean error.
//! * `resolve_inter_mb_mv` — driven independently so the four whole-MB
//!   modes (ZEROMV / NEARESTMV / NEARMV / NEWMV) + the SPLITMV base are
//!   all reachable from the bool partition without the reconstruction
//!   cost dominating the corpus.
//! * `find_near_mvs(above, left, aboveleft, current_ref, sign_bias)` —
//!   the §16.3 census. Fed adversarial neighbour `MbInfo`s (every
//!   `ref_frame` / `is_split` / `split_mvs` combination, MVs at the
//!   §17.1 ±1023 envelope edges + zero) so the dedup cursor, the
//!   SPLITMV-count recompute, the near/nearest swap, and the best/zero
//!   merge are all exercised. The §16.3 `mv_bias` sign correction is
//!   driven across every `(sign_bias, current_ref, neighbour_ref)`
//!   combination.
//! * `mv_ref_probs(cnt)` — the census → probability lookup. The census
//!   counts are bounded `0..=5` by construction; the harness also
//!   feeds `find_near_mvs`-produced counts straight through, asserting
//!   the documented `cnt[i] as usize` index never exceeds the
//!   `MV_COUNTS_TO_PROBS` row count.
//! * `decode_split_mv(dec, above, left, best_mv, mv_contexts)` — the
//!   §16.4 partition walk in isolation, so the partition-group anchor
//!   search (`while table[k] != j`), the `submv_ref` context selection
//!   ([`submv_ref_context`] across all five rows), and the NEW4X4
//!   differential add are reached on every partition shape from a
//!   short partition.
//! * `read_inter_mode` / `read_mv_partition` / `submv_ref` — the three
//!   bool-tree walks, each driven directly against a fresh
//!   `BoolDecoder` so a truncated / adversarial partition exercises the
//!   `BoolDecoderError` early-return on every node.
//! * `above_block_mv` / `left_block_mv` — the §20.11 neighbour-MV
//!   lookups for every sub-block index 0..=15, with SPLITMV and
//!   whole-MB and intra neighbour records (the `b + 12` / `b + 3`
//!   SPLITMV branches + the intra-zero fallback).
//! * `clamp_mv` / `MvClampRect::for_mb` — the §18.1 one-MB-border
//!   clamp at every plane position.
//!
//! Surface gap the existing fuzz targets leave cold:
//!
//! The round-263 `panic_free_inter_mb_reconstruct` target drives the
//! §16 *reconstruction* orchestrators (`reconstruct_inter_mb` /
//! `reconstruct_split_mv_mb`) with attacker-shaped vectors fed in
//! directly — it never runs the bool-coder-driven §16.2/§16.3/§16.4
//! *mode-info decode* that produces those vectors. The decode-side
//! `decode_vp8` / `Vp8DecoderState::decode_frame` targets reach §16
//! only through a fully-formed previous keyframe + §9.7 reference
//! refresh state machine, so the census neighbour records are always
//! self-consistent and the MV contexts are the §17.2-resolved frame
//! tables — never the adversarial `(MbInfo, SignBias, MvContexts)`
//! tuples this target constructs. None of the existing targets drive
//! `decode_inter_mb` / `decode_split_mv_mb` / `find_near_mvs` /
//! `decode_split_mv` / `resolve_inter_mb_mv` with attacker bytes for
//! the bool partition AND attacker-shaped neighbour/context state.
//!
//! Input layout (consumed from the front of the libFuzzer `data`):
//!
//! | Bytes | Meaning |
//! |------:|---------|
//! | `[0]`  | flags: bit0 `full_pixel`; bit1 `mb_skip_coeff`; bit2 `sign_bias.golden`; bit3 `sign_bias.alternate`; bit4 `version` selector; bits 5..=6 `current_ref` (`Last`/`Golden`/`AltRef`); bit7 zero-residue toggle |
//! | `[1]`  | `mb_cols` selector — `(data[1] % 3) + 1` ∈ {1,2,3} |
//! | `[2]`  | `mb_rows` selector — `(data[2] % 3) + 1` ∈ {1,2,3} |
//! | `[3]`  | `mb_col` selector (saturated against `mb_cols`) |
//! | `[4]`  | `mb_row` selector (saturated against `mb_rows`) |
//! | `[5]`  | above-neighbour descriptor (ref_frame / is_split / mv-edge selectors, see `mb_info_from`) |
//! | `[6]`  | left-neighbour descriptor |
//! | `[7]`  | above-left-neighbour descriptor |
//! | `[8]`  | NEWMV `best`-base / split-best selector (raw, drives the §17 differential add path) |
//! | `[9..28]` | row MV context (19 bytes), tiled over `default_mv_contexts()[0]` |
//! | `[28..47]`| col MV context (19 bytes), tiled over `default_mv_contexts()[1]` |
//! | `[47..]`  | bool-coder partition for the mode/MV tree walks AND reference-plane + §14 residue payload (tiled) |
//!
//! The harness caps input at the libFuzzer 4 KiB default (re-checked at
//! entry) and requires ≥ 47 bytes for the fixed header. Per-iteration
//! heap traffic: three I420 reference planes (≤ 9 MBs ⇒ ≤ 3456 B each)
//! plus the partition slice borrow; every coefficient / context / MV
//! array is stack-allocated.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::{
    bool_decoder::BoolDecoder,
    inter::{
        above_block_mv, clamp_mv, decode_inter_mb, decode_split_mv, decode_split_mv_mb,
        filter_set_for_version, find_near_mvs, left_block_mv, mv_ref_probs, read_inter_mode,
        read_mv_partition, resolve_inter_mb_mv, submv_ref, submv_ref_context, FilterSet, InterMode,
        MbInfo, MvClampRect, RefFrame, ReferencePlanes, ResolvedInterMode, SignBias,
    },
    motion_vector::{default_mv_contexts, Mv, MvContexts},
};

const MIN_INPUT_BYTES: usize = 47;
const MAX_INPUT_BYTES: usize = 4 * 1024;

/// §17.1 motion-vector component clamp envelope.
const MV_CLAMP: i16 = 1023;

/// Number of `MvContext` probabilities per component (§17.1 `MVPcount`).
const MV_PROB_COUNT: usize = 19;

/// Map a descriptor byte to one of the three reference frames.
fn ref_frame_from(sel: u8) -> RefFrame {
    match sel % 3 {
        0 => RefFrame::Last,
        1 => RefFrame::Golden,
        _ => RefFrame::AltRef,
    }
}

/// Build an adversarial neighbour [`MbInfo`] from one descriptor byte.
///
/// Layout of the descriptor:
/// * bits 0..=1 — `ref_frame`: 0 = intra (`None`), else `Last`/`Golden`/`AltRef`.
/// * bit2 — `is_split`.
/// * bit3 — populate `split_mvs` (only meaningful when `is_split`).
/// * bits 4..=5 — `mv` edge selector (0,0 / +max,+max / -max,-max / +max,-max).
/// * bits 6..=7 — split_mvs edge tiling selector.
fn mb_info_from(desc: u8) -> MbInfo {
    let ref_sel = desc & 0b11;
    let ref_frame = if ref_sel == 0 {
        None
    } else {
        Some(ref_frame_from(ref_sel - 1))
    };
    let is_split = (desc & 0b0000_0100) != 0;
    let want_split_mvs = (desc & 0b0000_1000) != 0;

    let mv = match (desc >> 4) & 0b11 {
        0 => Mv { row: 0, col: 0 },
        1 => Mv {
            row: MV_CLAMP,
            col: MV_CLAMP,
        },
        2 => Mv {
            row: -MV_CLAMP,
            col: -MV_CLAMP,
        },
        _ => Mv {
            row: MV_CLAMP,
            col: -MV_CLAMP,
        },
    };

    // Tile sixteen distinct edge-of-envelope vectors so the §20.11
    // neighbour-MV lookups read a non-uniform split-MV array.
    let split_mvs = if is_split && want_split_mvs {
        let base = ((desc >> 6) & 0b11) as i16;
        let mut arr = [Mv::default(); 16];
        for (i, slot) in arr.iter_mut().enumerate() {
            let v = MV_CLAMP - (((i as i16) * 67 + base * 31) % (2 * MV_CLAMP + 1));
            *slot = Mv { row: v, col: -v };
        }
        Some(arr)
    } else {
        None
    };

    MbInfo {
        ref_frame,
        mv,
        is_split,
        split_mvs,
    }
}

/// Build a per-component `MvContext` by tiling 19 payload bytes over the
/// §17 defaults. An empty slice leaves the defaults untouched.
fn tile_mv_context(base: [u8; MV_PROB_COUNT], src: &[u8]) -> [u8; MV_PROB_COUNT] {
    if src.is_empty() {
        return base;
    }
    let mut out = base;
    for (i, slot) in out.iter_mut().enumerate() {
        // Probabilities are u8; any byte is a valid §7.3 probability for
        // the bool coder (it clamps the split internally), so no masking.
        *slot = src[i % src.len()];
    }
    out
}

/// Tile `payload` into a destination of `len` bytes. Empty → all zeros.
fn tile_into(payload: &[u8], len: usize) -> Vec<u8> {
    if payload.is_empty() {
        return vec![0u8; len];
    }
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(payload.len());
        out.extend_from_slice(&payload[..take]);
    }
    out
}

/// Fill a 16-coefficient §14.2 residue block from a tiled payload.
fn fill_coeff_block(payload: &[u8], start: usize) -> [i16; 16] {
    let mut out = [0i16; 16];
    if payload.is_empty() {
        return out;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let p = (start + i * 2) % payload.len();
        let q = (start + i * 2 + 1) % payload.len();
        let raw = i16::from_le_bytes([payload[p], payload[q]]);
        // §14.4 inverse-DCT intermediate-product envelope (±2047).
        *slot = raw.clamp(-2047, 2047);
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < MIN_INPUT_BYTES || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let flags = data[0];
    let full_pixel = (flags & 0b0000_0001) != 0;
    let mb_skip_coeff = (flags & 0b0000_0010) != 0;
    let sign_bias = SignBias::new((flags & 0b0000_0100) != 0, (flags & 0b0000_1000) != 0);
    let version = (flags >> 4) & 1;
    let current_ref = ref_frame_from((flags >> 5) & 0b11);
    let zero_residue = (flags & 0b1000_0000) != 0;

    // §18.2 reference-plane dimensions — whole-MB, ≥ one MB.
    let mb_cols = ((data[1] as usize) % 3) + 1;
    let mb_rows = ((data[2] as usize) % 3) + 1;
    let lw = mb_cols * 16;
    let lh = mb_rows * 16;
    let cw = mb_cols * 8;
    let ch = mb_rows * 8;

    let mb_col = (data[3] as usize) % mb_cols;
    let mb_row = (data[4] as usize) % mb_rows;

    // §16.3 adversarial neighbour records.
    let above = mb_info_from(data[5]);
    let left = mb_info_from(data[6]);
    let aboveleft = mb_info_from(data[7]);
    let best_sel = data[8];

    // §17.2 MV contexts — attacker-tiled over the §17 defaults.
    let defaults = default_mv_contexts();
    let row_ctx = tile_mv_context(defaults[0], &data[9..28]);
    let col_ctx = tile_mv_context(defaults[1], &data[28..47]);
    let mv_contexts: MvContexts = [row_ctx, col_ctx];

    let payload = &data[47..];

    // ───── §16.3 census + §16.2 probability lookup, in isolation ─────
    let near = find_near_mvs(&above, &left, &aboveleft, current_ref, sign_bias);
    // The §16.3 census counts must stay inside the MV_COUNTS_TO_PROBS
    // row count (6); `mv_ref_probs` indexes with `cnt[i] as usize`.
    for &c in &near.cnt {
        assert!(
            (0..=5).contains(&c),
            "§16.3 census count {c} outside the documented 0..=5 envelope"
        );
    }
    let _probs = mv_ref_probs(&near.cnt);

    // ───── §18.1 clamp at this plane position ─────
    let bounds = MvClampRect::for_mb(mb_col, mb_row, mb_cols, mb_rows);
    for &m in &near.mvs {
        let _ = clamp_mv(m, &bounds);
    }

    // ───── §20.11 neighbour-MV lookups for every sub-block ─────
    let this_split = [Mv::default(); 16];
    for b in 0..16 {
        let l = left_block_mv(&this_split, &left, b);
        let a = above_block_mv(&this_split, &above, b);
        let _ = submv_ref_context(l, a);
    }

    // ───── §16.2/§16.3 tree walks against a fresh bool partition ─────
    // A short partition exercises the `BoolDecoderError` early-return at
    // every tree node on truncation.
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        if let Ok(mode) = read_inter_mode(&mut dec, &_probs) {
            // ZEROMV / NEARESTMV / NEARMV / NEWMV / SPLITMV all reachable.
            let _ = mode;
        }
    }
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        let _ = read_mv_partition(&mut dec);
    }
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        // Drive `submv_ref` across the five §16.4 context rows by
        // sourcing (left, above) from the census output.
        let l = near.mvs[0];
        let a = near.mvs[1];
        let _ = submv_ref(&mut dec, l, a);
    }

    // ───── §16.2/§16.3/§17 resolve, in isolation ─────
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        match resolve_inter_mb_mv(
            &mut dec,
            &above,
            &left,
            &aboveleft,
            current_ref,
            sign_bias,
            &bounds,
            &mv_contexts,
        ) {
            Ok(ResolvedInterMode::Whole { mode, mv }) => {
                assert!(
                    !matches!(mode, InterMode::Split),
                    "resolve_inter_mb_mv returned Whole with a Split mode"
                );
                let _ = mv;
            }
            Ok(ResolvedInterMode::Split { best }) => {
                // §16.4 partition walk in isolation against the same base.
                if let Ok(mut sdec) = BoolDecoder::init(payload) {
                    let _ = decode_split_mv(&mut sdec, &above, &left, best, &mv_contexts);
                }
            }
            Err(_) => {}
        }
    }

    // ───── §16.4 partition walk in isolation with a §17-derived base ─
    {
        let best = clamp_mv(
            Mv {
                row: (best_sel as i16) - 128,
                col: 127 - (best_sel as i16),
            },
            &bounds,
        );
        if let Ok(mut dec) = BoolDecoder::init(payload) {
            let _ = decode_split_mv(&mut dec, &above, &left, best, &mv_contexts);
        }
    }

    // ───── §18.3 filter set ─────
    let filter_set = filter_set_for_version(version);
    let filters = match filter_set {
        FilterSet::Sixtap | FilterSet::Bilinear => filter_set.taps(),
    };

    // ───── §18.2 reference planes ─────
    let y_plane = tile_into(payload, lw * lh);
    let u_plane = tile_into(payload, cw * ch);
    let v_plane = tile_into(payload, cw * ch);
    let reference = ReferencePlanes {
        y: &y_plane,
        u: &u_plane,
        v: &v_plane,
        y_stride: lw,
        uv_stride: cw,
        mb_cols,
        mb_rows,
    };

    // ───── §14.2/§14.4 dequantised residue blocks ─────
    let mut y2 = [0i16; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut u_coeffs = [[0i16; 16]; 4];
    let mut v_coeffs = [[0i16; 16]; 4];
    if !zero_residue {
        y2 = fill_coeff_block(payload, 0);
        for (i, blk) in y_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 + i * 32);
        }
        for (i, blk) in u_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 * 17 + i * 32);
        }
        for (i, blk) in v_coeffs.iter_mut().enumerate() {
            *blk = fill_coeff_block(payload, 32 * 21 + i * 32);
        }
    }

    // ───── §16.2/§16.3/§18 end-to-end whole-MB decode + reconstruct ─
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        let _ = decode_inter_mb(
            &mut dec,
            &reference,
            mb_col,
            mb_row,
            &above,
            &left,
            &aboveleft,
            current_ref,
            sign_bias,
            &mv_contexts,
            full_pixel,
            filters,
            mb_skip_coeff,
            &y2,
            &y_coeffs,
            &u_coeffs,
            &v_coeffs,
        );
    }

    // ───── §16.4 SPLITMV end-to-end decode + reconstruct ─────
    // (No Y2 block: the §14.2 SPLITMV short-circuit.)
    if let Ok(mut dec) = BoolDecoder::init(payload) {
        let _ = decode_split_mv_mb(
            &mut dec,
            &reference,
            mb_col,
            mb_row,
            &above,
            &left,
            &aboveleft,
            current_ref,
            sign_bias,
            &mv_contexts,
            full_pixel,
            filters,
            mb_skip_coeff,
            &y_coeffs,
            &u_coeffs,
            &v_coeffs,
        );
    }
});
