#![no_main]

//! Fuzz: the §14/§12 key-frame intra-reconstruction core driven
//! *directly* — `frame::decode_keyframe`.
//!
//! The full-stream targets (`panic_free_decode_keyframe`,
//! `decode_stream_token_descent`) reach this reconstruction walker only
//! behind the §9.1 frame-tag and §19.2 coded-header validation gates,
//! so a random or fixture-derived bitstream rarely drives it through the
//! full Cartesian product of intra modes, B_PRED sub-block-mode mixes,
//! skip flags, and hostile post-dequant coefficient magnitudes. This
//! target bypasses the parser: it materialises a `mb_cols × mb_rows`
//! grid of arbitrary [`MacroblockModes`] + [`MbCoeffs`] straight from the
//! fuzz bytes and feeds them to [`decode_keyframe`], so libFuzzer can
//! explore the reconstruction surface itself.
//!
//! What the reconstruction core does per macroblock, all exercised here:
//!
//! * §11/§12 neighbour gather across MB boundaries (top row / left
//!   column edge handling, the §12.2 above-left corner sample),
//! * every §12 16×16 luma mode (`DC_PRED` / `V_PRED` / `H_PRED` /
//!   `TM_PRED`) and the §11.3 `B_PRED` path with all ten §11.2 4×4
//!   sub-block predictors in arbitrary per-sub-block arrangements,
//! * every §11.4 8×8 chroma mode,
//! * the §14.3 inverse WHT (Y2) → §14.4 inverse DCT (Y) dispatch, the
//!   §14.5 add-and-clamp residue pass, and the `mb_skip_coeff`
//!   short-circuit,
//! * arbitrary (already-dequantized) `i16` coefficient values, including
//!   the magnitudes that stress the §14.4 IDCT's intermediate `i32`
//!   accumulation and the §14.5 `0..=255` clamp.
//!
//! Oracle: panic-freedom (no overflow, no out-of-bounds write into the
//! plane rasters, no debug-assert) plus full plane materialisation —
//! every output byte is folded into an FNV-1a accumulator so ASan reads
//! the entire claimed-produced raster (a short-write / stale-stride bug
//! surfaces here). On success the plane lengths are also checked against
//! the documented `mb_cols·16 × mb_rows·16` (luma) / `·8 × ·8` (chroma)
//! geometry.
//!
//! Input budget: the grid is capped at 16 macroblocks total (e.g. 4×4),
//! well under the fuzz-runner per-iteration memory cap, with mutation
//! headroom left in the byte stream for the mode/coeff fields.

use libfuzzer_sys::fuzz_target;
use oxideav_vp8::frame::{decode_keyframe, MbCoeffs};
use oxideav_vp8::macroblock::{IntraBmode, IntraUvMode, IntraYMode, MacroblockModes};

/// Maximum macroblocks per reconstructed frame. 16 MBs (any factoring,
/// e.g. 4×4 or 1×16) is a 64×64-pixel luma raster at most — ~6 KiB of
/// I420 plane bytes, trivially inside the runner's memory cap while still
/// forcing multi-row / multi-column neighbour-gather edges.
const MAX_MBS: usize = 16;

/// A tiny forward byte cursor over the fuzz input. Every reader returns a
/// well-defined value once the bytes run out (0 / the first enum variant)
/// so the harness keeps producing a structurally valid grid no matter how
/// short or hostile the input is.
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

    /// Little-endian `i16` from two bytes — used for the post-dequant
    /// coefficient magnitudes, so the full signed range (including the
    /// extremes that stress the §14.4 IDCT accumulation) is reachable.
    fn i16(&mut self) -> i16 {
        let lo = self.byte();
        let hi = self.byte();
        i16::from_le_bytes([lo, hi])
    }
}

fn y_mode(b: u8) -> IntraYMode {
    match b % 5 {
        0 => IntraYMode::Dc,
        1 => IntraYMode::V,
        2 => IntraYMode::H,
        3 => IntraYMode::Tm,
        _ => IntraYMode::B,
    }
}

fn uv_mode(b: u8) -> IntraUvMode {
    match b % 4 {
        0 => IntraUvMode::Dc,
        1 => IntraUvMode::V,
        2 => IntraUvMode::H,
        _ => IntraUvMode::Tm,
    }
}

fn b_mode(b: u8) -> IntraBmode {
    match b % 10 {
        0 => IntraBmode::Dc,
        1 => IntraBmode::Tm,
        2 => IntraBmode::Ve,
        3 => IntraBmode::He,
        4 => IntraBmode::Ld,
        5 => IntraBmode::Rd,
        6 => IntraBmode::Vr,
        7 => IntraBmode::Vl,
        8 => IntraBmode::Hd,
        _ => IntraBmode::Hu,
    }
}

/// Fill a 4×4 coefficient block with `i16`s drawn from the cursor. Most
/// bytes get spent on the §14.4 DCT-stressing Y blocks; chroma / Y2 share
/// the same well so the cursor exhaustion behaviour is uniform.
fn fill_block(c: &mut Cursor, block: &mut [i16; 16]) {
    for coeff in block.iter_mut() {
        *coeff = c.i16();
    }
}

fn build_coeffs(c: &mut Cursor, has_y2: bool) -> MbCoeffs {
    let mut coeffs = MbCoeffs {
        y2: [0; 16],
        y: [[0; 16]; 16],
        u: [[0; 16]; 4],
        v: [[0; 16]; 4],
    };
    if has_y2 {
        fill_block(c, &mut coeffs.y2);
    }
    for blk in coeffs.y.iter_mut() {
        fill_block(c, blk);
    }
    for blk in coeffs.u.iter_mut() {
        fill_block(c, blk);
    }
    for blk in coeffs.v.iter_mut() {
        fill_block(c, blk);
    }
    coeffs
}

fn build_modes(c: &mut Cursor) -> MacroblockModes {
    let ym = y_mode(c.byte());
    let skip = c.byte() & 1 == 1;
    let uvm = uv_mode(c.byte());
    let subblock_modes = if ym == IntraYMode::B {
        let mut sub = [IntraBmode::Dc; 16];
        for s in sub.iter_mut() {
            *s = b_mode(c.byte());
        }
        Some(sub)
    } else {
        None
    };
    MacroblockModes {
        // segment_id is irrelevant to the reconstruction walker (the
        // §10 quantizer override has already been folded into the
        // dequantized coefficients we synthesise), so leave it None.
        segment_id: None,
        mb_skip_coeff: skip,
        y_mode: ym,
        subblock_modes,
        uv_mode: uvm,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let mut c = Cursor::new(data);

    // Pick a grid shape from the first two bytes. Both axes are forced
    // into 1..=MAX_MBS and the product capped, so the plane allocation
    // can never blow the memory budget.
    let mb_cols = (c.byte() as usize % MAX_MBS) + 1;
    let mut mb_rows = (c.byte() as usize % MAX_MBS) + 1;
    while mb_cols * mb_rows > MAX_MBS {
        mb_rows -= 1;
    }
    if mb_rows == 0 {
        return;
    }
    let total = mb_cols * mb_rows;

    let mut modes = Vec::with_capacity(total);
    let mut coeffs = Vec::with_capacity(total);
    for _ in 0..total {
        let m = build_modes(&mut c);
        // Y2 is present iff the macroblock is not B_PRED (§14.3).
        let has_y2 = m.y_mode != IntraYMode::B;
        let co = build_coeffs(&mut c, has_y2);
        modes.push(m);
        coeffs.push(co);
    }

    let planes = match decode_keyframe(mb_cols, mb_rows, &modes, &coeffs) {
        Ok(p) => p,
        Err(_) => return,
    };

    // Geometry invariants the walker documents.
    assert_eq!(planes.y_stride, mb_cols * 16);
    assert_eq!(planes.uv_stride, mb_cols * 8);
    assert_eq!(planes.y.len(), mb_cols * 16 * mb_rows * 16);
    assert_eq!(planes.u.len(), mb_cols * 8 * mb_rows * 8);
    assert_eq!(planes.v.len(), planes.u.len());

    // Touch every produced byte so a short-write / stale-length bug is
    // caught under ASan.
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in planes
        .y
        .iter()
        .chain(planes.u.iter())
        .chain(planes.v.iter())
    {
        acc ^= b as u64;
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    std::hint::black_box(acc);
});
