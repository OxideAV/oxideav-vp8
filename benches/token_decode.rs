//! Round 285 — the fused §13.2 token-descent decode path in isolation
//! and on an inter-heavy multi-frame stream.
//!
//! Round 283 rewrote the per-coefficient §13.2 `COEFF_TREE` walk as a
//! fused branch-coded descent (`decode_block_core`) with a write-order
//! raster output and a batched §7.3 bool-decoder renormalisation. The
//! whole-frame decode benches measured the win (−7…−10 %), but no bench
//! attributes wall-time to the token layer *alone*: `keyframe_decode` /
//! `inter_decode_short_clip` wrap it inside §12/§16 prediction, §14
//! transforms, and the §15 loop filter, so a future descent-shape change
//! (e.g. a two-bit-at-a-time read, or a band-state cache) has no
//! isolated A/B target. This bench gives the §13 layer its own harness:
//!
//! * `decode_mb_coeffs_dense_64mb` — a 64-MB token partition whose every
//!   block carries a long token run (small magnitudes through Cat
//!   extra-bits territory), i.e. the dense-residue worst case where the
//!   descent + extra-bits + sign reads dominate.
//! * `decode_mb_coeffs_sparse_64mb` — the same walk over blocks that are
//!   mostly zero-runs + early EOBs, the §13.2 "skip dct_eob after DCT_0"
//!   inner-loop shape that dominates well-predicted inter residue.
//! * `inter_decode_12f_176x144_token_heavy` — a whole-stream decode of a
//!   1-keyframe + 11-P-frame 176×144 clip with textured drifting content,
//!   so the fused descent runs in its real inter-frame context (the
//!   round-283 target workload) at a P-frame share three times the
//!   existing `inter_decode_short_clip` bench's.
//!
//! The token partitions are produced by the crate's own §13 encoder
//! (`TokenEncoder` / `encode_coeff_block`), and the setup decodes each
//! partition once and asserts the recovered coefficients byte-equal the
//! intended blocks — so the measured loop is a proven-valid bitstream
//! walk, never an error path.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_vp8::dct_tokens::{
    decode_mb_coeffs, BlockType, MbEntropyCtx, DEFAULT_COEFF_PROBS, ZIGZAG,
};
use oxideav_vp8::encoder::TokenEncoder;
use oxideav_vp8::{BoolDecoder, I420Frame, KeyframeParams, Vp8DecoderState, Vp8InterStreamEncoder};

/// Number of macroblocks in the synthetic token partition — one row of
/// 64 MBs (a 1024-pixel-wide frame's worth of luma token work).
const NUM_MBS: usize = 64;

/// One macroblock's coefficient payload in **scan order** (the §13.2
/// token-emission order): the Y2 block, sixteen Y blocks, four U and
/// four V blocks. `has_y2` is always true here (the 16×16-luma-mode /
/// non-SPLITMV inter shape — the hot path), so Y blocks start at scan
/// position 1 and scan slot 0 stays zero.
#[derive(Clone)]
struct MbScanCoeffs {
    y2: [i16; 16],
    y: [[i16; 16]; 16],
    u: [[i16; 16]; 4],
    v: [[i16; 16]; 4],
}

/// xorshift32 — deterministic, dependency-free value stream.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Dense block: a long run of non-zero coefficients whose magnitudes
/// sweep the §13.2 token classes (1 / 2 / 3-4 / Cat1 / Cat2, with an
/// occasional Cat3-4 spike), followed by an explicit EOB. `first` is 1
/// for Y-after-Y2 blocks, 0 otherwise.
fn dense_block(rng: &mut u32, first: usize) -> [i16; 16] {
    let mut out = [0i16; 16];
    // 11 non-zero coefficients starting at `first` — deep into band 6+.
    for slot in out.iter_mut().skip(first).take(11) {
        let r = xorshift(rng);
        let mag = match r % 10 {
            0..=2 => 1 + (r >> 8) % 2,  // 1..=2 (literal tokens)
            3..=5 => 3 + (r >> 8) % 2,  // 3..=4 (DCT_3 / DCT_4)
            6..=7 => 5 + (r >> 8) % 14, // 5..=18 (Cat1 / Cat2)
            8 => 19 + (r >> 8) % 16,    // 19..=34 (Cat3)
            _ => 35 + (r >> 8) % 32,    // 35..=66 (Cat4)
        } as i16;
        *slot = if r & 0x1000 != 0 { -mag } else { mag };
    }
    out
}

/// Sparse block: two small values separated by a §13.2 zero-run (the
/// DCT_0 inner-loop shape), then EOB. One block in four is entirely
/// empty (immediate EOB).
fn sparse_block(rng: &mut u32, first: usize) -> [i16; 16] {
    let mut out = [0i16; 16];
    let r = xorshift(rng);
    if r % 4 == 0 {
        return out; // immediate EOB
    }
    let v1 = 1 + (r % 3) as i16;
    let v2 = 1 + ((r >> 8) % 2) as i16;
    out[first] = if r & 0x10000 != 0 { -v1 } else { v1 };
    // Zero-run of 3..=6 positions, then the second value.
    let gap = 3 + (r >> 16) % 4;
    let pos2 = first + 1 + gap as usize;
    if pos2 < 16 {
        out[pos2] = if r & 0x20000 != 0 { -v2 } else { v2 };
    }
    out
}

/// Build `NUM_MBS` macroblocks of scan-order coefficients.
fn build_mbs(dense: bool) -> Vec<MbScanCoeffs> {
    let mut rng = if dense { 0x2853_77c5 } else { 0x11d2_85f1 };
    (0..NUM_MBS)
        .map(|_| {
            let gen = |rng: &mut u32, first: usize| {
                if dense {
                    dense_block(rng, first)
                } else {
                    sparse_block(rng, first)
                }
            };
            MbScanCoeffs {
                y2: gen(&mut rng, 0),
                y: core::array::from_fn(|_| gen(&mut rng, 1)),
                u: core::array::from_fn(|_| gen(&mut rng, 0)),
                v: core::array::from_fn(|_| gen(&mut rng, 0)),
            }
        })
        .collect()
}

/// The §20.16 above/left predictor-slot index for residual block
/// `0..=24`: Y sub-block `b` lands in slot `b & 3` (above; its column)
/// or `b >> 2` (left; its row), U blocks 16..=19 in slots 4..=5, V
/// blocks 20..=23 in slots 6..=7, and the Y2 block 24 in slot 8 —
/// the nine-entry `MbEntropyCtx` layout.
fn ctx_slots(block_index: usize) -> (usize, usize) {
    match block_index {
        b @ 0..=15 => (b & 3, b >> 2),
        b @ 16..=19 => (4 + ((b - 16) & 1), 4 + ((b - 16) >> 1)),
        b @ 20..=23 => (6 + ((b - 20) & 1), 6 + ((b - 20) >> 1)),
        24 => (8, 8),
        _ => unreachable!(),
    }
}

/// Encode the MB sequence into one §13 token partition with the crate's
/// own token encoder, walking blocks in the §13/§14.2 residual order
/// (Y2 → 16 Y → 4 U → 4 V) and maintaining the same above/left non-zero
/// predictors `decode_mb_coeffs` keeps on the other side. The grid is a
/// single MB row: `above` is per-MB-column, `left` rolls within the row.
fn encode_partition(mbs: &[MbScanCoeffs]) -> Vec<u8> {
    let mut enc = TokenEncoder::new(DEFAULT_COEFF_PROBS);
    let mut above = vec![MbEntropyCtx::default(); mbs.len()];
    let mut left = MbEntropyCtx::default();

    for (mb_i, mb) in mbs.iter().enumerate() {
        let mut emit = |block_index: usize,
                        block_type: BlockType,
                        coeffs: &[i16; 16],
                        above: &mut MbEntropyCtx,
                        left: &mut MbEntropyCtx| {
            let (a_slot, l_slot) = ctx_slots(block_index);
            let nz = enc
                .encode_block(
                    block_type,
                    above.nonzero[a_slot],
                    left.nonzero[l_slot],
                    coeffs,
                )
                .expect("encode_block");
            above.nonzero[a_slot] = nz != 0;
            left.nonzero[l_slot] = nz != 0;
        };
        emit(24, BlockType::Y2, &mb.y2, &mut above[mb_i], &mut left);
        for (i, blk) in mb.y.iter().enumerate() {
            emit(i, BlockType::YAfterY2, blk, &mut above[mb_i], &mut left);
        }
        for (i, blk) in mb.u.iter().enumerate() {
            emit(16 + i, BlockType::UV, blk, &mut above[mb_i], &mut left);
        }
        for (i, blk) in mb.v.iter().enumerate() {
            emit(20 + i, BlockType::UV, blk, &mut above[mb_i], &mut left);
        }
    }
    enc.finish()
}

/// Decode the partition once and assert every recovered raster block
/// equals the intended scan-order block routed through the §13 zig-zag
/// (`raster[ZIGZAG[c]] = scan[c]`) — the setup-time validity proof.
fn verify_partition(bytes: &[u8], mbs: &[MbScanCoeffs]) {
    let mut dec = BoolDecoder::init_partition(bytes);
    let mut above = vec![MbEntropyCtx::default(); mbs.len()];
    let mut left = MbEntropyCtx::default();
    let to_raster = |scan: &[i16; 16]| {
        let mut raster = [0i16; 16];
        for c in 0..16 {
            raster[ZIGZAG[c]] = scan[c];
        }
        raster
    };
    for (mb_i, mb) in mbs.iter().enumerate() {
        let out = decode_mb_coeffs(
            &mut dec,
            /* has_y2 = */ true,
            /* mb_skip_coeff = */ false,
            &DEFAULT_COEFF_PROBS,
            &mut above[mb_i],
            &mut left,
        )
        .expect("decode_mb_coeffs");
        assert_eq!(out.y2, to_raster(&mb.y2), "Y2 mismatch at MB {mb_i}");
        for i in 0..16 {
            assert_eq!(out.y[i], to_raster(&mb.y[i]), "Y{i} mismatch at MB {mb_i}");
        }
        for i in 0..4 {
            assert_eq!(out.u[i], to_raster(&mb.u[i]), "U{i} mismatch at MB {mb_i}");
            assert_eq!(out.v[i], to_raster(&mb.v[i]), "V{i} mismatch at MB {mb_i}");
        }
    }
}

fn bench_decode_mb_coeffs(c: &mut Criterion) {
    for (name, dense) in [
        ("decode_mb_coeffs_dense_64mb", true),
        ("decode_mb_coeffs_sparse_64mb", false),
    ] {
        let mbs = build_mbs(dense);
        let bytes = encode_partition(&mbs);
        verify_partition(&bytes, &mbs);

        let mut g = c.benchmark_group("token_decode");
        g.throughput(Throughput::Elements(NUM_MBS as u64));
        g.bench_function(name, |b| {
            let mut above = vec![MbEntropyCtx::default(); NUM_MBS];
            b.iter(|| {
                let mut dec = BoolDecoder::init_partition(black_box(&bytes));
                above.fill(MbEntropyCtx::default());
                let mut left = MbEntropyCtx::default();
                let mut nz_acc = 0i64;
                for actx in above.iter_mut() {
                    let out = decode_mb_coeffs(
                        &mut dec,
                        true,
                        false,
                        &DEFAULT_COEFF_PROBS,
                        actx,
                        &mut left,
                    )
                    .expect("decode_mb_coeffs");
                    nz_acc += out.y2[0] as i64 + out.y[0][0] as i64;
                }
                black_box(nz_acc)
            });
        });
        g.finish();
    }
}

// ───────────────────── inter-heavy multi-frame stream ─────────────────────

const CLIP_W: u32 = 176;
const CLIP_H: u32 = 144;
const CLIP_FRAMES: usize = 12;

/// Deterministic textured drift clip: a moving mixed-frequency gradient
/// plus a travelling high-contrast square over a static hash texture, so
/// every P frame carries real motion vectors *and* a dense residue (the
/// texture decorrelates under drift, keeping the §13 token partitions
/// busy — the workload the round-283 fused descent targets).
fn synthesise_clip() -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = CLIP_W as usize;
    let h = CLIP_H as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    (0..CLIP_FRAMES)
        .map(|t| {
            let mut y = vec![0u8; w * h];
            for row in 0..h {
                for col in 0..w {
                    // Drifting gradient + static texture hash.
                    let grad = (((col + t) * 256 / w + (row + t) * 256 / h) / 2) as u32;
                    let tex = ((row as u32).wrapping_mul(31) ^ (col as u32).wrapping_mul(17)) & 31;
                    y[row * w + col] = (grad + tex).min(255) as u8;
                }
            }
            // Travelling 24×24 high-contrast square (3 px/frame).
            let sx = 16 + t * 3;
            let sy = 24 + t * 2;
            for row in sy..(sy + 24).min(h) {
                for col in sx..(sx + 24).min(w) {
                    y[row * w + col] = if (row + col) & 1 == 0 { 235 } else { 16 };
                }
            }
            let mut u = vec![128u8; cw * ch];
            let mut v = vec![128u8; cw * ch];
            for row in 0..ch {
                for col in 0..cw {
                    u[row * cw + col] = (118 + (col * 20 / cw) + t) as u8;
                    v[row * cw + col] = (132 + (row * 20 / ch) + t) as u8;
                }
            }
            (y, u, v)
        })
        .collect()
}

fn bench_inter_decode_token_heavy(c: &mut Criterion) {
    // Encode once (setup, unmeasured): keyframe_interval = CLIP_FRAMES
    // so the stream is exactly one K + eleven P frames.
    let clip = synthesise_clip();
    let params = KeyframeParams {
        y_ac_qi: 32,
        ..KeyframeParams::default()
    };
    let mut enc =
        Vp8InterStreamEncoder::new(params, CLIP_FRAMES as u64).expect("Vp8InterStreamEncoder");
    let packets: Vec<Vec<u8>> = clip
        .iter()
        .map(|(yi, ui, vi)| {
            let frame = I420Frame::packed(CLIP_W, CLIP_H, yi, ui, vi);
            enc.encode_frame(&frame).expect("inter encode_frame").bytes
        })
        .collect();

    let mut g = c.benchmark_group("token_decode");
    g.throughput(Throughput::Elements(
        (CLIP_W * CLIP_H) as u64 * CLIP_FRAMES as u64,
    ));
    g.sample_size(20);
    g.bench_function("inter_decode_12f_176x144_token_heavy", |b| {
        b.iter(|| {
            let mut dec = Vp8DecoderState::new();
            let mut acc = 0usize;
            for pkt in &packets {
                let frame = dec.decode_frame(black_box(pkt)).expect("decode_frame");
                acc += frame.y.len();
            }
            black_box(acc)
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_decode_mb_coeffs,
    bench_inter_decode_token_heavy
);
criterion_main!(benches);
