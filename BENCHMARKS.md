# oxideav-vp8 benchmarks

Round 170 (2026-05-27) wired up `criterion` benches for the
`oxideav-vp8` encode + decode hot paths and used the resulting profiles
to retarget three concrete optimisations. This document records the
baseline + post-optimisation numbers so future rounds have a stable
A/B reference.

## Running

```sh
# Stable + default features — every bench runs the scalar paths.
cargo bench -p oxideav-vp8 --bench keyframe_encode --bench keyframe_decode \
            --bench inter_encode_short_clip --bench inverse_transform_4x4 \
            --bench motion_comp_subpel_luma --bench intra_predict_dc16 \
            --bench loop_filter_normal -- --quick

# Nightly + simd feature — `inverse_wht_4x4` goes through the
# `core::simd::Simd<i32, 4>` rewrite. Tests and other benches are
# byte-identical to the stable path.
RUSTC=$(rustup which --toolchain nightly rustc) \
RUSTDOC=$(rustup which --toolchain nightly rustdoc) \
rustup run nightly cargo bench -p oxideav-vp8 --features simd \
    --bench inverse_transform_4x4 -- --quick
```

Bench inputs are synthesised in-bench (no committed fixture files); the
encoder benches use a deterministic gradient + flat-square I420 source
that exercises both intra paths so the per-MB mode picker does real
work. The micro-benches drive a single fixed input so successive
measurements compare apples-to-apples.

## Hardware

Apple M4-class aarch64, macOS 25.1, rustc 1.95.0 (stable) / 1.97.0-nightly
(simd). `--quick` measurements (criterion's reduced-iteration mode).

## Headline results — round 170

| Bench | Baseline | Post-opt (stable) | Post-opt (nightly+simd) | Δ |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 9.06 ms | **8.30 ms** | (same as stable) | **−8.4 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 153.3 µs | 153.6 µs | 154.7 µs | ±0.2 % |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 14.35 ms | **11.24 ms** | 11.07 ms | **−21.7 %** |
| `inverse_transform_4x4/inverse_dct_4x4` | 10.23 ns | 10.06 ns | 10.57 ns | −1.7 % |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.83 ns | 9.38 ns | **7.48 ns** | **−23.9 %** (simd) |
| `motion_comp_subpel_luma/filter_block_4x4_sub3x5` | 35.40 ns | **24.65 ns** | (same) | **−30.4 %** |
| `motion_comp_subpel_luma/mb_sixtap_2d_16x4x4` | 267.6 ns | 264.3 ns | (same) | −1.2 % |
| `intra_predict_dc16/predict_y16x16_dc` | 4.88 ns | 4.77 ns | (same) | −2.3 % |
| `intra_predict_dc16/predict_y16x16_v` | 3.66 ns | 3.67 ns | (same) | ±0.2 % |
| `intra_predict_dc16/predict_y16x16_h` | 3.75 ns | 3.73 ns | (same) | −0.4 % |
| `loop_filter_normal/subblock_filter_4_4` | 2.16 ns | 2.12 ns | (same) | −1.9 % |
| `loop_filter_normal/simple_segment_4` | 2.12 ns | 2.12 ns | (same) | ±0.0 % |

Throughput on the whole-frame benches:

* `keyframe_encode_320x240_qi32`: 8.48 → **9.25 Mpx/s** (+9 %)
* `keyframe_decode_320x240_qi32`: ≈ **501 Mpx/s** (decoder was already
  cheap; the round-170 optimisations target the encode-heavy paths)
* `inter_encode_4f_128x128_qi32`: 4.57 → **5.83 Mpx/s** (+28 %)

## Profile evidence (pre-optimisation)

`sample(1)`-based PID-attach profile of the `keyframe_encode` and
`inter_encode_short_clip` bench binaries while looping under
`--measurement-time 30`. Top self-time symbols (15 s wall, 1 ms
interval):

### `keyframe_encode_320x240_qi32`

| Self-samples | Symbol |
|---:|---|
| 59 | `encoder::token_to_bit_path::descend` |
| 52 | `_xzm_free` (libsystem_malloc) |
| 50 | `encoder::estimate_block_bits` |
| 42 | `encoder::encode_mb_block_set_with_neighbors` |
| 41 | `_xzm_xzone_malloc` |
| 29 | `log2` (libsystem_m) |
| 17 | `inverse_transform::inverse_dct_4x4` |

### `inter_encode_4f_128x128_qi32`

| Self-samples | Symbol |
|---:|---|
| 60 | `log2` (libsystem_m) |
| 55 | `motion_comp::fetch_block_whole_pixel` |
| 54 | `_xzm_free` |
| 39 | `motion_comp::sixtap_2d` |
| 37 | `_xzm_xzone_malloc` |
| 35 | `motion_comp::fetch_block_halo` |
| 33 | `encoder::token_to_bit_path::descend` |
| 29 | `encoder::estimate_block_bits` |

## Optimisations landed (round 170)

### 1. `bool_bits` `log2` lookup table — encoder RD scoring

Profile evidence: `log2` was the #6 self-time symbol on the keyframe
encode and the #1 on the inter encode, called from
`encoder::estimate_block_bits` → `bool_bits` for every emitted bit of
every candidate block during RD scoring.

Rewrite: `bool_bits` now consults a 256-entry `LazyLock<[f64; 256]>`
lookup table, `BIT_COST_BY_FALSE_PROB[p] = -log2(p / 256)`, with the
`p = 0` slot floored at `-log2(1/256) = 8.0` to preserve the original
`prob.max(1)` + `1/256` clamp. The `value == true` branch reads slot
`256 - p` (with `p = 0` pre-clamped to slot 255) so a single i/o pair
per call replaces the FP log2.

**Wins:** drove the −21.7 % on `inter_encode_short_clip` and the
−8.4 % on `keyframe_encode`. Bit-identical encoder output on every
fixture in the existing test suite.

### 2. `motion_comp::fetch_block_*` in-bounds fast paths

Profile evidence: `fetch_block_whole_pixel` (#2) and `fetch_block_halo`
(#6) together accounted for ~90 self-samples on the inter bench. Each
function ran a per-pixel `isize.clamp(0, w-1)` even when the entire
block / halo landed strictly inside the plane (the common case mid-
frame).

Rewrite: both functions branch on `src_x0 >= 0 && src_y0 >= 0 &&
src_x0 + N <= w && src_y0 + N <= h`. In the in-bounds case each output
row becomes a single `copy_from_slice` over a contiguous `N`-byte run
of the source plane — no per-pixel clamps, no per-byte bounds checks.
The edge-replication slow path stays bit-identical for border MBs.

**Wins:** drove the −30 % on `filter_block_4x4` and contributed to the
inter-encode delta. Byte-exact reconstruction on the existing
encoder/decoder roundtrip + inter-stream tests.

### 3. `inverse_wht_4x4` `core::simd::Simd<i32, 4>` rewrite (simd feature)

Profile evidence: not in the top-5 self-time of either bench (the WHT
fires once per Y2-bearing MB) but it's the cleanest vectorisation
target — the §14.3 listing is two passes of independent column /
row butterflies which map 1:1 onto a 4-lane `Simd<i32, 4>` layout
(one lane per column for the first pass; transpose; one lane per row
for the second pass).

Rewrite: `inverse_wht_4x4` now dispatches at compile time between
`inverse_wht_4x4_scalar` (bit-identical to the prior RFC 6386 listing)
and `inverse_wht_4x4_simd` (the `core::simd` rewrite). The SIMD layout
is derived directly from the §14.3 spec listing — no external SIMD
reference consulted. The `simd` feature is nightly-only because
`core::simd` is itself nightly.

**Wins:** −23.9 % on the `inverse_wht_4x4` micro-bench (9.83 → 7.48 ns
on nightly+simd). Doesn't yet move the whole-frame numbers because
the WHT is a small fraction of total decode time; the same rewrite
pattern applied to `inverse_dct_4x4` in a future round is the natural
next step.

## Round 180 — `inverse_dct_4x4` SIMD

Round 180 (2026-05-29) landed the round-170 deferred follow-on: a
`core::simd::Simd<i32, 4>` rewrite of `inverse_dct_4x4` parallel to the
existing `inverse_wht_4x4_simd`. Same dispatch shape — the public
`inverse_dct_4x4` calls `inverse_dct_4x4_simd` on nightly + `simd` and
`inverse_dct_4x4_scalar` (the spec listing, unchanged) elsewhere.

Byte-exact against the scalar listing on a 21-input stress set in
`src/inverse_transform.rs::tests::dct_simd_matches_scalar_on_stress_inputs`
(DC-only over 10 magnitudes from ±8 to ±4096, single-AC at every one
of the 15 non-DC positions, plus two near-extreme mixed gradients).
The same suite was added for the WHT
(`wht_simd_matches_scalar_on_stress_inputs`) so the round-170 path now
has the same dense equivalence proof as the new round-180 path.

| Bench | r170 stable | r180 stable | r180 nightly+simd | Δ |
|---|---:|---:|---:|---:|
| `inverse_transform_4x4/inverse_dct_4x4` | 10.06 ns | 10.07–10.32 ns | **9.51–10.02 ns** | **−1 to −5 %** (simd) |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.38 ns | 9.4–10.0 ns | 7.36–8.03 ns | unchanged from r170 |

The §14.4 inverse DCT carries 8 fixed-point multiplies per pass
against the §14.3 WHT's zero — that's where the smaller r180 SIMD
margin comes from. The WHT moves end-to-end as four parallel
adds/subs per pass; the DCT has multiply-then-shift dependencies that
serialise inside each lane. Both passes still vectorise correctly
across the four-lane width, just at a lower speedup than the WHT.
The headline value of the round is the byte-exact equivalence and
the now-symmetric §14.3 / §14.4 SIMD shape, not a runtime headline
number — `inverse_dct_4x4` fires once per Y / U / V sub-block (16 +
4 + 4 = 24 calls per MB on decode) vs `inverse_wht_4x4` once per
Y2-bearing MB, so the cumulative whole-frame impact tracks the small
per-call delta scaled by the call count.

## Round 194 — rate-control `y_ac_qi` sweep

Round 194 (2026-05-31) adds a depth-mode bench
(`rate_control_qi_sweep`) that walks `KeyframeParams::y_ac_qi` — the
§9.6 baseline quantiser index, the principal rate-control knob on
`encode_keyframe` — across ten representative values on the same
deterministic 320×240 I420 source the round-170 `keyframe_encode` bench
uses. The goal is *not* a new optimisation but a published trade-off
curve readers can tune against: every other §9.6 quantiser delta
defaults to 0 in `KeyframeParams`, so a single `y_ac_qi` value moves the
DC + AC luma / chroma quantiser bank in lockstep.

```sh
CARGO_TARGET_DIR=/tmp/oxideav-vp8-target \
    cargo bench -p oxideav-vp8 --bench rate_control_qi_sweep -- --quick
```

Headline numbers (Apple M4 / aarch64, criterion `--quick`, source =
mixed luma gradient + centre flat-128 square, 320×240 I420):

| `y_ac_qi` | Output | bpp | Encode wall | Throughput |
|---:|---:|---:|---:|---:|
|   8 | 1701 B | 0.18 |  9.33 ms |  8.23 Mpx/s |
|  16 |  676 B | 0.07 |  8.48 ms |  9.05 Mpx/s |
|  24 |  612 B | 0.06 |  8.44 ms |  9.10 Mpx/s |
|  32 |  595 B | 0.06 |  8.38 ms |  9.16 Mpx/s |
|  40 |  480 B | 0.05 |  8.16 ms |  9.41 Mpx/s |
|  48 |  466 B | 0.05 |  8.21 ms |  9.35 Mpx/s |
|  56 |  461 B | 0.05 |  7.92 ms |  9.69 Mpx/s |
|  72 |  360 B | 0.04 |  7.61 ms | 10.09 Mpx/s |
|  96 |  355 B | 0.04 |  7.54 ms | 10.18 Mpx/s |
| 120 |  299 B | 0.03 |  7.31 ms | 10.51 Mpx/s |

Reading the curve:

* The byte cost falls monotonically with `y_ac_qi` — a strict
  expected-direction sanity check on the quantiser path. The jump
  between qi=8 and qi=16 (1701 → 676 B, ~−60 %) is the steepest
  segment; everything past qi=32 lives in the lossy-but-flat tail of
  the dial.
* Encode wall time falls in the same direction (9.33 → 7.31 ms, −22 %)
  because larger quantisers produce shorter token streams + more EOB
  early-exits inside `encoder::token_to_bit_path::descend` and
  `encoder::estimate_block_bits` — the two top self-time symbols
  identified in the round-170 profile (`BENCHMARKS.md` §Profile
  evidence above). The trend is consistent with that profile: the
  rate-control knob and the encoder's hot path are coupled through
  the token bit count.
* Throughput delta across the sweep is +28 % (8.23 → 10.51 Mpx/s) —
  picking a higher qi for previewing / draft modes is a real wall-time
  win, not just a bytes-on-the-wire win.
* The synthetic source compresses to small outputs at every qi
  (DC-heavy gradient + uniform flat square is the easy case for the
  intra picker) — production-content numbers will be larger in
  absolute bytes but the *shape* of the qi/bytes/throughput trade-off
  carries.

This bench is intended as a published baseline: re-run after any
encoder change and compare the per-qi byte and wall-time columns to
catch regressions on a single dial that drives the whole rate-control
surface.

## Round 204 — `token_to_bit_path` precomputation

Round 204 (2026-06-01) lands the round-170 follow-up
*"`encoder::token_to_bit_path::descend` — the function walks a small
tree and ends up RD-scoring the same token paths repeatedly. A
precomputed token-to-path table would remove the descent entirely."*

`token_to_bit_path` is hit at least three times per coefficient in the
encoder hot path — once by the block writer, once by the RD bit-cost
estimator, and once by the §13.4 token-prob counts fitter. Before
round 204 each call allocated a fresh `Vec<(usize, bool)>` and ran a
recursive `ENC_COEFF_TREE` descent. The descent is a pure function of
`(start_index ∈ {0, 2}, token ∈ 12 entries)`, so all 24 cells are now
materialised once at module load through `std::sync::LazyLock` into a
fixed-shape `[[([TokenBitStep; 7], u8); 12]; 2]` table. Path widths cap
at 7 (`Cat3..Cat6` from the root); the unreachable `(start = 2, Eob)`
cell is a `length = 0` tombstone. The function returns
`&'static [TokenBitStep]` — index-and-slice, zero per-token allocation.

| Bench | r194 stable | r204 stable | Δ |
|---|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 8.51 ms | **5.97 ms** | **−29.8 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 10.82 ms | **10.20 ms** | **−5.7 %** |

Throughput on the whole-frame benches (Apple M4 / aarch64,
criterion `--quick`):

* `keyframe_encode_320x240_qi32`: 9.03 → **12.87 Mpx/s** (+43 %)
* `inter_encode_4f_128x128_qi32`: 6.06 → **6.42 Mpx/s** (+5.9 %)

The keyframe-encode delta is the headline. The round-170 sample profile
counted `_xzm_xzone_malloc` + `_xzm_free` at ≈ 93 combined self-samples
across the encoder bench; removing the per-coefficient `Vec` allocation
collapses that into the static table's single one-shot init. The inter
delta is smaller because motion search + reconstruction dilute the
token-emission share of total time.

Equivalence proof:
`encoder::tests::token_bit_path_table_matches_tree_descent` re-runs the
original recursive descent inline and asserts every reachable cell
agrees on length, every prob_index, and every bit. Plus the full
450-test lib suite + every existing integration test reach unchanged
byte-exact output through the new dispatch.

## Round 220 — `forward_transform_4x4` micro-bench

Round 220 (2026-06-03) closes a long-standing gap in the published
A/B surface: the §14.3 forward WHT (`forward_wht_4x4`) and the §14.4
forward DCT (`forward_dct_4x4`) — the encoder partners of the
round-170 / round-180 inverse-side primitives — had no published
micro-bench. The whole-frame `keyframe_encode` and
`inter_encode_short_clip` benches drive them inside the §11 intra
picker + §13 token emit + §15 loop-filter cascade, which makes
attributing a wall-time delta to a forward-transform rewrite
ambiguous. The new `forward_transform_4x4` bench mirrors
`inverse_transform_4x4`'s input layout (the same representative
DC-heavy 4×4 residual block) so a side-by-side read of the two files
lines up the forward and inverse passes one-for-one.

Headline baseline (Apple M4 / aarch64, criterion `--quick`):

| Bench | Wall time |
|---|---:|
| `forward_transform_4x4/forward_dct_4x4` | 10.13 ns |
| `forward_transform_4x4/forward_wht_4x4` | 10.48 ns |
| `inverse_transform_4x4/inverse_dct_4x4` | 10.25 ns |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.76 ns |

Observations:

* The forward DCT and the §14.4 inverse DCT are within a percent of
  each other (10.13 vs 10.25 ns) — the §14.3 / §14.4 forward and
  inverse passes share the same fixed-point constants
  (`COSPI8_SQRT2_MINUS1`, `SINPI8_SQRT2`), the same butterfly
  shape, and the same number of `c_mul` / `s_mul` invocations per
  pass. The forward path's per-row `round_div2` (the symmetric
  round-away-from-zero step that doesn't appear in the inverse)
  costs the small remaining delta.
* The forward WHT is ~8 % more expensive than the inverse WHT
  (10.48 vs 9.76 ns) — the forward path's symmetric `round_div2`
  fires on every output sample (16 calls per `forward_wht_4x4`)
  where the §14.3 inverse path's matching `(x + 3) >> 3` rounds
  without the negative-symmetric branch.
* Per-MB call count: 24 × `forward_dct_4x4` (16 Y + 8 chroma) +
  potentially 1 × `forward_wht_4x4` (for MBs with a Y2 DC plane).
  At 10.13 + 10.48 ns this is ≈ 253 ns / MB on the forward side
  alone, which at 320 × 240 (300 MBs / frame) is ~76 µs per frame
  — ~1.3 % of the 5.81 ms `keyframe_encode` wall time. Reads as:
  the forward primitives are not the encoder hot path (token emit
  + RD scoring dwarf them), but they're now visible at criterion's
  micro-bench resolution, ready as an A/B target for any future
  SIMD / unroll work on the encoder side parallel to the round-180
  `inverse_dct_4x4_simd` rewrite.

The bench input matches `inverse_transform_4x4.rs::SAMPLE_INPUT`
verbatim so a future `forward_dct_4x4_simd` rewrite would produce
the same per-call delta shape across the two pairs.

## Round 226 — `forward_wht_4x4` / `forward_dct_4x4` SIMD rewrites

Round 226 (2026-06-04) closes the round-220 next-round candidate: the
§14.3 forward WHT and §14.4 forward DCT now dispatch through a
`core::simd::Simd<i32, 4>` rewrite under the `simd` cargo feature,
parallel to the round-180 `inverse_*_simd` rewrites. The layout
mirrors the inverse SIMD path one-for-one: hold the input as four
`Simd<i32, 4>` row-vectors (lane `j` of row `i` carries `input[i*4 +
j]`); the first pass operates as four parallel lane-wide column
butterflies + (for the DCT) `c_mul` / `s_mul` lane-wide multiplies;
transpose; run the second pass the same way; apply the symmetric
`round_div2` step lane-wide via a `Mask::select(pos_branch,
neg_branch)` over `simd_ge(0)` (matching scalar `round_div2` bit-
for-bit — the two arithmetic branches are unconditional).

Headline `--quick` numbers (Apple M4 / aarch64), comparing scalar
(public dispatcher on stable, no `simd` feature) against the new SIMD
path (public dispatcher on nightly + `simd`):

| Bench | Scalar | SIMD | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_wht_4x4` | 10.74 ns | **8.72 ns** | **−18.8 %** |
| `forward_transform_4x4/forward_dct_4x4` | 10.71 ns | 11.54 ns | +7.7 % |

Observations:

* The forward WHT SIMD path is the cleanest win in the suite next to
  the round-180 inverse WHT (also ≈ −20 %). Both transforms are
  pure butterfly chains over an i32 column axis; the four-lane
  layout collapses the per-column scalar loop into a straight-line
  add / sub sequence. The bulk of the −19 % comes from the second-
  pass `round_div2` step folding into four lane-wide `Mask::select`s
  in place of four scalar branch-pairs.
* The forward DCT SIMD path is +8 % slower than scalar — the
  multiply-heavy `c_mul` / `s_mul` chain (8 i32 lane-wide multiplies
  per pass × 2 passes) doesn't pipeline as well as the scalar
  straight-line code, and the lane-wide `round_div2_simd` adds a
  mask + select that the scalar path avoids. This is the same shape
  the round-180 inverse DCT measurement showed (≈ −1 to −5 % over
  scalar) — the forward direction's per-pass `round_div2` step is
  the extra cost, where the inverse path's `(x + 4) >> 3` is a
  pure shift. The SIMD path is kept enabled under `simd` for shape
  parity with the WHT and so the byte-exact equivalence assertion
  (`fdct_forward_simd_matches_scalar_on_stress_inputs`) is exercised
  on every nightly + `simd` test run; a future round can split the
  dispatch (route `forward_dct_4x4` to scalar even under `simd` if
  the regression matters on a target hardware).
* Per-MB call count: 24 × `forward_dct_4x4` (16 Y + 8 chroma) + at
  most 1 × `forward_wht_4x4`. Under `simd` the forward-side wall is
  24 × 11.54 + 1 × 8.72 ≈ 286 ns / MB, vs scalar 24 × 10.71 + 1 ×
  10.74 ≈ 268 ns / MB — a ~+7 % bias on the forward primitives only,
  whose round-170 share of `keyframe_encode` was ~1.3 %. That puts
  the expected `keyframe_encode` impact at sub-0.1 %, deep below the
  bench's `--quick` noise envelope; verifying empirically by
  re-running `keyframe_encode` under both feature settings is left
  to a future profile-depth round.

Equivalence proof: `forward_transform::tests::{
fdct_forward_simd_matches_scalar_on_stress_inputs,
fwht_forward_simd_matches_scalar_on_stress_inputs}` run a 21-input
stress set (the same matrix shape as the round-180 inverse tests:
all-zero / DC-only at 10 magnitudes / single-AC at every of the
15 positions / mixed gradients / high-AC mid-range, both positive
and alternating-sign) and assert public-dispatch byte-equality with
the renamed `_scalar` variants. The full 452-test lib suite and
every integration test pass on both stable (scalar dispatch) and
nightly + `simd` (SIMD dispatch).

## Round 247 — `forward_dct_4x4` SIMD dispatch split

Round 247 (2026-06-07) closes the round-226 deferred next-round
candidate. The §14.4 `forward_dct_4x4` SIMD path ran the lane-wide
`c_mul` / `s_mul` chain (8 i32 multiplies per pass × 2 passes) plus a
`round_div2_simd` mask + select; round 226 measured it ~+8 % slower
than the scalar straight-line code on `aarch64-apple-darwin` but kept
the SIMD dispatch for shape parity with the §14.3 WHT. Round 247
re-routes the public `forward_dct_4x4` dispatcher to the scalar path
under every feature configuration — `forward_wht_4x4` keeps its SIMD
dispatch unchanged (no multiplies, still −18 % under scalar).

The `_simd` implementation stays compiled under the `simd` feature
(now with `#[allow(dead_code)]` since the dispatcher no longer reaches
it) so the byte-equivalence assertion still has a symbol to call. The
test was renamed `_simd_matches_scalar`-style logic-wise and now calls
`forward_dct_4x4_simd` directly against `forward_dct_4x4_scalar` over
the 21-input stress matrix (the public-dispatch comparison would be
trivially equal after the split). A second
`fdct_public_dispatch_is_scalar` test runs on every configuration and
asserts `forward_dct_4x4 == forward_dct_4x4_scalar` so a future round
can't accidentally re-route the dispatcher without flipping that
assertion too.

Headline `--quick` numbers (Apple M4 / aarch64) comparing the
post-round-226 SIMD dispatch with the round-247 scalar dispatch:

| Bench | r226 SIMD | r247 SIMD | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_dct_4x4` | 11.69 ns | **9.81 ns** | **−16.2 %** |
| `forward_transform_4x4/forward_wht_4x4` | 8.94 ns | 8.74 ns | −2.2 % |

Stable (no `simd`): unchanged within criterion `--quick` noise (forward
DCT 10.67 → 10.82 ns; forward WHT 10.81 → 10.84 ns). Per-MB forward
side under `simd` now drops from 24 × 11.54 + 1 × 8.72 ≈ 286 ns / MB
to 24 × 9.81 + 1 × 8.74 ≈ 244 ns / MB — a ~−15 % bias on the forward
primitives, the inverse of round 226's +7 % bias.

## Round 249 — `forward_dct_4x4_scalar` canonical-butterfly refactor

Round 249 (2026-06-07) reorganises the scalar §14.4 `forward_dct_4x4_scalar`
listing into the canonical `(a1, b1, c1, d1)` partial-sum butterfly form
mirroring the §14.4 `inverse_dct_4x4_scalar` listing. The original form
matched the direct derivation from `T_fwd * input * T_fwd^T` (`o0 = i0
+ i4 + i8 + i12`; eight `c_mul` / `s_mul` calls split across `o4` and
`o12`); the refactored form groups partial sums (`a1 = i0 + i12`, `b1
= i4 + i8` for the evens, `c1` / `d1` carrying the fixed-point
multiply pair-differences for the odds) so forward and inverse paths
share visual shape. Bit-exactness is preserved by never collapsing a
multiply pair-difference into `c_mul(i0 - i12)` (the `>> 16` truncation
is non-linear under sum); each `c_mul` / `s_mul` call still evaluates
separately and the addition reorder is on associative i32 sums. A new
`forward_dct_4x4_listing` private function holds the unfactored
direct-derivation form as a regression oracle, and
`fdct_scalar_matches_direct_derivation_listing` asserts the refactored
`_scalar` produces identical bytes on the 21-input stress matrix.

`--quick` numbers comparing round-247 (post-SIMD-split) and round-249
(post-refactor) on Apple M4 / aarch64:

| Bench | r247 | r249 | Δ |
|---|---:|---:|---:|
| `forward_transform_4x4/forward_dct_4x4` (stable) | 10.82 ns | 10.83 ns | within noise |
| `forward_transform_4x4/forward_dct_4x4` (nightly + `simd`) | 9.81 ns | 10.05 ns | +2.4 % (within `--quick` noise envelope) |
| `forward_transform_4x4/forward_wht_4x4` (stable) | 10.84 ns | 10.54 ns | within noise |
| `forward_transform_4x4/forward_wht_4x4` (nightly + `simd`) | 8.74 ns | 8.93 ns | within noise |

The refactor is a readability / spec-shape parity change (LLVM was
already CSE'ing the i0/i12 partial sums under `-O3`); the bench numbers
on both configurations agree on the round-247 baseline within the
criterion `--quick` envelope. The `forward_dct_4x4_simd` listing (which
already used a butterfly-friendly lane-wide form in round 226) now
agrees visually as well as bit-exactly with the scalar.

## Round 269 — `sixtap_2d` SIMD (§18.3 six-tap sub-pixel interpolation)

Round 269 (2026-06-10) closes the long-standing round-170 candidate:
the §18.3 / §20.14 `sixtap_2d` kernel (#4 self-time on the round-170
inter-encode profile) now dispatches through a `core::simd::Simd<i32,
4>` rewrite under the `simd` feature, same dispatcher shape as the §14
transforms / §14.1 dequant / §12.2 TM_PRED rounds. Each convolution
row's four §18.3 `interp` dot products are one four-lane vector — tap
k's support lanes are the contiguous run `halo[r*9 + k ..][..4]` — and
the horizontal pass's clamped intermediate stays resident in `i32`
vectors so the vertical pass runs with zero loads.

Lane-type finding (vs the round-170 candidate note's `Simd<i16, 8>`
stripe): the §18.3 dot product over `u8` support spans `[-8160,
40800]` (the ½-displacement row `{3, -16, 77, 77, -16, 3}` has
positive-tap sum 160; 160 × 255 = 40800 > `i16::MAX`), so a single
`i16` accumulator wraps. A parity-split two-accumulator `i16×8`
two-row-stripe variant (every tap-parity class partial sum provably
fits `i16`: positive class sums cap at 128) was implemented and
measured during the round: `mb_sixtap_2d_16x4x4` ≈ 260–268 ns (no
better than scalar) and `filter_block_4x4_sub3x5` ≈ 28.3–28.7 ns
(~+15 % regression — the per-tap two-row gather eats the lane-width
win). The four-lane `i32` form, which matches the 4×4 sub-block
geometry exactly, is what shipped.

Headline `--quick` numbers (Apple M4 / aarch64, triple-run, nightly
toolchain for both columns so the compiler version cancels):

| Bench | Scalar | SIMD | Δ |
|---|---:|---:|---:|
| `motion_comp_subpel_luma/mb_sixtap_2d_16x4x4` | 271.5 ns | **248.5 ns** | **−8.5 %** |
| `motion_comp_subpel_luma/filter_block_4x4_sub3x5` | 24.87 ns | **23.55 ns** | **−5.3 %** |

Stable (no `simd`) is unchanged: 268.1 ns / 24.81 ns, within noise of
the nightly scalar column. The margin is modest next to TM_PRED's
−87.7 % because the scalar `interp` loop over fixed-size arrays was
already auto-vectorising well; the explicit kernel's win comes mostly
from keeping the two-pass intermediate in vector registers (no
narrow-to-u8 / widen-from-u8 round trip between the passes) rather
than from new parallelism.

Equivalence proof:
`motion_comp::tests::sixtap_2d_simd_matches_scalar_on_stress_inputs`
(13 halos × all 64 `(mx, my)` fraction pairs × both §18.3 filter
sets) plus `sixtap_2d_accumulator_extremes_match_scalar` (both dot-
product extremes through output column 0 under the ½-displacement
taps). The full lib suite passes on stable (463) and nightly + `simd`
(465).

## Round 270 — MB-scale §18.3 luma batching (`sixtap_mb_luma`)

Round 270 (2026-06-10) closes the round-269 next-round candidate
"MB-scale §18.3 batching". The round-269 per-4×4 `sixtap_2d` SIMD win
was capped by the 4-wide block geometry; the real headroom is one layer
up in `predict_inter_mb`, where all sixteen luma sub-blocks of a
non-SPLITMV MB share one §18.1 motion vector. Instead of sixteen
overlapping 9×9 `fetch_block_halo` + `sixtap_2d` calls, the sub-pixel
luma path now fetches one 21×21 halo (`fetch_luma_mb_halo`) and
synthesises the whole 16×16 luma block in a single two-pass convolution
(`sixtap_mb_luma`): a horizontal pass of 21 rows × 16 cols then a
vertical pass of 16 rows × 16 cols. The SIMD partner widens each pass to
`Simd<i32, 16>` — one sixteen-lane vector per output row, the clamped
horizontal intermediate resident in `i32` vectors so the vertical pass
runs with zero loads.

The new `motion_comp_subpel_luma/mb_luma_batched_16x16` bench measures
the whole-MB synthesis against the per-sub-block partner
`mb_luma_per_subblock_16x16` (sixteen `sixtap_2d` calls on the same
sub-pixel workload). Numbers on `aarch64-apple-darwin` (criterion
`--quick`, nightly toolchain for both columns so the compiler version
cancels):

| Bench | Per-sub-block | Batched scalar | Batched SIMD |
|---|---:|---:|---:|
| whole 16×16 luma block | 260–268 ns | **158.8 ns** | **140.2 ns** |

Reading the result:

* **Batched scalar vs per-sub-block: −41 %** (158.8 vs ~267.9 ns). The
  bulk of the MB-scale win is structural, not SIMD: one 21×21 fetch +
  one tight two-pass loop replaces sixteen 9×9 fetches and sixteen
  separate two-pass calls, amortising the per-block border / gather
  setup the round-170 profile attributed ~90 self-samples to
  (`fetch_block_halo` + `fetch_block_whole_pixel`).
* **Batched SIMD vs batched scalar: −12 %** (140.2 vs 158.8 ns). The
  `Simd<i32, 16>` rewrite quadruples the lane width vs round-269's
  `Simd<i32, 4>` 4×4-block form, so each horizontal-pass row is one
  sixteen-lane vector and the per-block setup the 4-wide form paid 16×
  is paid once.
* **Batched SIMD vs per-sub-block: −47 %** end-to-end (140.2 vs
  ~265 ns) — the headline. This is the path `predict_inter_mb` takes for
  every sub-pixel non-SPLITMV inter MB on nightly + `simd`.

Byte-exactness is anchored three ways:
`sixtap_mb_luma_matches_per_subblock_path` (whole-MB vs sixteen
`sixtap_2d` calls over the carved 9×9 sub-halos, every `(mx, my)` × both
filter sets), `sixtap_mb_luma_simd_matches_scalar_on_stress_inputs`
(dispatcher vs scalar over the flat / ramp / checker / LCG stress set),
and `predict_inter_mb_sub_pixel_at_border_uses_mb_halo_clamp` (a real
corner-MB prediction through the MB-halo border-clamp fallback). The
chroma path keeps the per-sub-block dispatch (the 8×8 chroma block gains
little from MB-scale batching, and the §18.1 averaged-vector / SPLITMV
cases use per-sub-block vectors).

## Round 271 — MB-scale §18.3 chroma batching (`sixtap_mb_chroma`)

Round 271 (2026-06-10) closes the round-270 next-round candidate "MB-scale
§18.3 chroma batching". The four chroma sub-blocks of each plane on a
non-SPLITMV inter MB share one §18.1 averaged motion vector, so the six-tap
support of the whole 8×8 chroma block is one contiguous 13×13 region.
`predict_inter_mb`'s sub-pixel chroma path now fetches one 13×13 halo
(`fetch_chroma_mb_halo`) and synthesises the whole 8×8 chroma block in a
single two-pass convolution (`sixtap_mb_chroma`): a horizontal pass of 13
rows × 8 cols then a vertical pass of 8 rows × 8 cols. The SIMD partner
widens each pass to `Simd<i32, 8>` — one eight-lane vector per output row,
the clamped horizontal intermediate resident in `i32` vectors so the
vertical pass runs with zero loads. This is the chroma analogue of the
round-270 16×16-luma path (`sixtap_mb_luma`, 21×21 halo, `Simd<i32, 16>`).

The new `motion_comp_subpel_luma/mb_chroma_batched_8x8` bench measures the
whole-MB synthesis against the per-sub-block partner
`mb_chroma_per_subblock_8x8` (four `sixtap_2d` calls on the same sub-pixel
workload). Numbers on `aarch64-apple-darwin` (criterion `--quick`, nightly
toolchain for both columns so the compiler version cancels):

| Bench | Per-sub-block | Batched scalar | Batched SIMD |
|---|---:|---:|---:|
| whole 8×8 chroma block | 67.7 ns | **43.9 ns** | **38.6 ns** |

Reading the result:

* **Batched scalar vs per-sub-block: −35 %** (43.9 vs 67.7 ns) — one 13×13
  fetch + one tight two-pass loop replaces four 9×9 fetches and four
  separate two-pass calls, amortising the per-block border / gather setup.
* **Batched SIMD vs batched scalar: −12 %** (38.6 vs 43.9 ns) — the
  `Simd<i32, 8>` rewrite halves round-270's `Simd<i32, 16>` lane width to
  match the 8-wide chroma block, so each horizontal-pass row is one
  eight-lane vector and the per-block setup is paid once.
* **Batched SIMD vs per-sub-block: −43 %** end-to-end (38.6 vs 67.7 ns) —
  the path `predict_inter_mb` takes for every sub-pixel non-SPLITMV inter
  MB's chroma planes on nightly + `simd`. The absolute win is smaller than
  the round-270 luma path's (8×8 vs 16×16 block) but the per-output-pixel
  ratio matches.

Byte-exactness is anchored five ways:
`sixtap_mb_chroma_matches_per_subblock_path` (whole-MB vs four `sixtap_2d`
calls over the carved 9×9 sub-halos, every `(mx, my)` × both filter sets),
`sixtap_mb_chroma_simd_matches_scalar_on_stress_inputs` (dispatcher vs
scalar over the flat / ramp / checker / LCG stress set),
`fetch_chroma_mb_halo_matches_subblock_halos_in_bounds` /
`fetch_chroma_mb_halo_clamps_at_top_left_corner` (halo containment +
border-clamp), and two real-prediction tests
(`predict_inter_mb_chroma_sub_pixel_matches_per_subblock_path` mid-plane +
`predict_inter_mb_chroma_sub_pixel_at_border_uses_mb_halo_clamp` corner) on
both U and V. Stable lib 468 → 474; nightly + `simd` lib 470 → 476.

## Round 272 — whole-pixel non-SPLITMV MB batching (`fetch_luma_mb_whole_pixel` / `fetch_chroma_mb_whole_pixel`)

Round 272 (2026-06-10) closes the round-271 next-round candidate
"whole-pixel non-SPLITMV MB batching". When the shared §18.1 motion
vector of a non-SPLITMV inter MB is *whole-pixel* (`mv & 7 == 0` per
component), the §18.3 prediction is a pure copy — no convolution. The
sixteen luma sub-blocks (or four chroma sub-blocks per plane) then share
one contiguous source region at integer offset `(mb_x, mb_y) + (mv >> 3)`,
so the whole block can be fetched in one pass instead of sixteen / four
overlapping 4×4 `fetch_block_whole_pixel` copies. Unlike the round-270 /
round-271 sub-pixel paths this is pure gather amortisation: one bounds
check + one border-straddle decision per MB rather than per sub-block, no
SIMD needed.

`predict_inter_mb`'s whole-pixel luma branch now issues one
`fetch_luma_mb_whole_pixel` (16×16, stride 16) and each chroma plane one
`fetch_chroma_mb_whole_pixel` (8×8, stride 8), both with the same
in-bounds contiguous-row fast path / per-pixel §20.14 `build_mc_border`
clamp fallback split as the per-sub-block fetch.

The new `motion_comp_subpel_luma/mb_*_whole_pixel_*` benches measure the
batched fetch against the per-sub-block assembly on a 64×64 deterministic
source (Apple M4 / aarch64, criterion `--quick`):

| Bench | Per-sub-block | Batched | Delta |
|---|---:|---:|---:|
| whole 16×16 luma copy | 46.89 ns | **13.13 ns** | **−72 %** |
| whole 8×8 chroma copy | 8.49 ns | **4.74 ns** | **−44 %** |

Reading the result: the per-sub-block luma path pays sixteen
`fetch_block_whole_pixel` calls, each with its own in-bounds test and its
own four-row `copy_from_slice` loop into a `[u8; 16]` scratch that is then
re-copied four rows at a time into `out.y`; the batched path does one
in-bounds test and sixteen direct 16-byte row copies straight into
`out.y`. The chroma ratio is smaller (four sub-blocks vs sixteen) but the
same shape. Byte-exactness is anchored five ways against the
per-sub-block `fetch_block_whole_pixel` assembly (in-bounds luma + chroma,
top-left luma + bottom-right chroma clamp, and a real corner-MB
`predict_inter_mb` covering Y + U + V); the existing
`reconstruct_inter_mb_matches_legacy_for_whole_pixel` test anchors the
full reconstruct path. Stable lib 474 → 479.

## Round 274 — SPLITMV write-strategy A/B (negative result)

Round 274 (2026-06-11) closes the rounds 270–272 next-round candidate
"SPLITMV whole-pixel sub-block batching" with a measured negative result.
SPLITMV macroblocks (RFC 6386 §16.4) carry sixteen distinct luma vectors
(plus four chroma), so the MB-scale shared-halo batch (`sixtap_mb_luma` /
`sixtap_mb_chroma`, `fetch_*_mb_whole_pixel`) cannot apply — every sub-block
is synthesised independently. The only remaining freedom is *how* each
per-sub-block result lands in the macroblock raster. Two byte-identical
strategies were benchmarked (`motion_comp_subpel_luma/splitmv_predict_*`):

| Strategy | Wall time |
|---|---:|
| `splitmv_predict_scratch_copy` (shipped) | **398.8 ns** |
| `splitmv_predict_strided_write` (`filter_block_4x4_into`) | 480.5 ns |

* **scratch_copy** builds a contiguous `[u8; 16]` block per sub-block
  (`filter_block_4x4`), then copies four contiguous 4-byte rows into the
  stride-16 (luma) / stride-8 (chroma) raster — the form `predict_split_mv`
  ships.
* **strided_write** writes each synthesised sub-block directly into the
  raster at its strided offset via the new `filter_block_4x4_into`, with no
  intermediate block.

The scratch-copy form wins by ~17 % on Apple M4 / aarch64: the contiguous
`[u8; 16]` lets the compiler vectorise the per-row writes, where the
scattered strided writes into a stride-16 raster cannot. `predict_split_mv`
therefore keeps the scratch path. `filter_block_4x4_into` is retained as a
public primitive (for callers that already own a destination raster) and is
byte-exact against `filter_block_4x4` + a strided copy
(`filter_block_4x4_into_matches_filter_block_4x4`,
`strided_into_assembly_matches_predict_split_mv`). The 16-sub-block
synthesis cost dominates; the write strategy is a ~4–5 % slice of total
per-MB SPLITMV prediction, so neither form moves whole-frame inter encode
materially — the value of the round is the documented A/B closing the
candidate.

## Round 276 — MV-cost `log2` table + allocation-free tree walks

Round 276 (2026-06-11) re-profiled the inter-encode bench per the
standing "remaining allocator churn" candidate below. The fresh
`sample(1)` profile (15 s attach, 1 ms interval, `--measurement-time
30`) put three related symbols high in self-time:

| Self-samples | Symbol |
|---:|---|
| 411 | `log2` (libsystem_m) — #6 overall |
| 200 | `motion_vector::mv_component_bits::small_mv_bits::find_path` |
| 122 | `encoder::treed_bits::find_path` |
| ~540 | `_xzm_free` / `_xzm_xzone_malloc` / `_malloc_zone_malloc` / `RawVec::finish_grow` / `grow_one` combined |

Caller attribution placed the malloc churn squarely under
`mv_component_bits` (per-candidate-MV RD costing inside motion search)
and the §11 mode-RD `treed_bits` walk — exactly the "next biggest
short-lived `Vec`" the round-272 candidate predicted. Three fixes, all
producing bit-identical encoder output:

1. **`mv_component_bits` joins the round-170 `-log2(p/256)` lookup
   table.** The §17.1 MV-component costing still computed an inline libm
   `log2` per priced bool; it now reads the same 256-entry
   `BIT_COST_BY_FALSE_PROB` table the block-RD `bool_bits` has used
   since round 170. For every `(prob, value)` pair the table entry is
   the *exact* double the inline expression produced (same clamps, same
   dyadic arguments), locked by a new full-range regression test
   (`mv_component_bits_matches_reference_over_full_range`: every value
   in `-1023..=1023` × three contexts including a 0/255 clamp-corner
   sweep, `==` on f64 — not approximate).
2. **`small_mv_bits` / `write_small_mv` drop the recursive DFS + `Vec`.**
   §17.1 `small_mvtree` is a perfect depth-3 tree whose spec listing
   comments give the leaf↔path correspondence directly (`0 = "000"` …
   `7 = "111"`), so both walks now descend the tree with the 3-bit
   binary expansion of the leaf (MSB first), still indexing
   `SMALL_MVTREE` for the node-halved probability offsets and
   debug-asserting the landing leaf. No allocation, no search.
3. **`treed_bits` / `BoolEncoder::write_treed` get a fixed-buffer DFS.**
   The shared `treed_find_path` helper writes the path into a stack
   `[bool; 16]` (a §8.1 path visits distinct internal nodes, and the
   deepest tree the crate walks — `BMODE_TREE` — has 9), replacing the
   per-call `Vec<bool>` in both the §11 mode-cost and mode-emit walks.

### Measured A/B (criterion `--quick`, Apple M4 / aarch64, stable)

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 10.32 ms | **9.38 ms** | **−9.1 %** |
| `keyframe_encode/encode_keyframe_320x240_qi32` | 5.83 ms | **5.53 ms** | **−5.2 %** |

Throughput: inter 6.35 → **6.99 Mpx/s**, keyframe 13.17 → **13.89
Mpx/s**. The post-change profile (same attach methodology) confirms the
causal chain: libm `log2` disappears from the top-of-stack list
entirely (411 → 0 samples), the `mv_component_bits` family drops 305 →
123, the tree walk shows up only as the allocation-free
`treed_find_path::dfs` (115 samples), and the malloc/free family drops
~540 → ~300 samples. Encoder output is bit-identical: the RD costs are
the same doubles, so every pick is unchanged — the full fixture /
roundtrip suite passes untouched (lib tests 481 → 482 with the new
full-range cost regression anchor).

## Round 277 — compile-time SPLITMV partition-group table

Round 277 (2026-06-11) ran the attribution pass the round-276 candidate
asked for. A fresh `sample(1)` profile of the inter-encode bench (15 s
attach, 1 ms interval, `--measurement-time 60`) counted ~335
allocator-family top-of-stack self-samples (`_xzm_free` 98,
`_xzm_xzone_malloc` 43, `_malloc_zone_malloc` 24, `finish_grow` 25,
`grow_one` 20, plus dedup/realloc tails). Call-tree attribution put
essentially **all** of it under `encoder::partition_groups` — NOT the
`near_mv` census (which is stack-only) nor the per-frame `stream`
assembly the candidate guessed. `partition_groups` rebuilt a
`Vec<Vec<usize>>` (1 outer + up to 16 inner allocations) on every call,
and the SPLITMV scorer calls it once per (MB, partition shape) — 4× per
MB per reference frame; the per-candidate `SplitMvCandidate`
`submv_modes` / `submv_new_diffs` `Vec` pair accounted for the rest.

Two changes, bit-identical encoder output:

1. **`partition_groups` becomes a compile-time table.** A `const fn`
   decomposes each row of the §20.13 `MV_PARTITIONS` constant into a
   `PartitionGroups { members: [[usize; 16]; 16], len, num_groups }`
   record; the four records live in a `static` and the function now
   returns `&'static PartitionGroups`. Same single source of truth
   (the spec table), zero runtime work, zero allocation. Member order
   inside each group stays raster-ascending so `group[0]` remains the
   §16.4 anchor.
2. **`SplitMvCandidate` drops its two `Vec`s** for fixed `[_; 16]`
   arrays indexed by group id; the emit path already walks exactly
   `num_groups` entries so the tail fill values are never read.

### Measured A/B (criterion, 30 s measurement, Apple M4 / aarch64, stable)

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 8.96 ms | **8.83 ms** | **−1.4 %** |

Throughput: inter 7.31 → **7.39 Mpx/s** (same-session A/B; criterion's
stored-baseline estimate said −2.2 %). The post-change profile confirms
the causal chain: the allocator family disappears from the ≥5-sample
top-of-stack list entirely, and the same call-tree counting script
drops 794 → **40** allocator samples — what remains is the bounded
per-frame `stream` assembly (BoolEncoder output growth), not per-MB
churn. Bit-identity was checked two ways: the full 482-test suite plus
fixture/roundtrip integration suites, and an 18-frame A/B byte-hash
(3 resolutions × 6 frames × 3 quantisers through
`Vp8InterStreamEncoder`) — identical FNV-1a before/after.

## Round 278 — whole-frame keyframe path under nightly + `simd` (measurement round)

Round 278 (2026-06-11) closes the standing "whole-frame `keyframe_encode`
re-measure under nightly + `simd`" candidate. The round-247 note predicted
a sub-percent whole-frame delta, but that arithmetic only counted the
§14.3 / §14.4 *forward* primitives — since then the `simd` feature grew
the §14.1 dequant (round 267), §12.2 TM_PRED (round 268) and §18.3
six-tap (rounds 269–271, inter-only) kernels, all of which sit on the
keyframe encode's §11 RD-reconstruct loop or the decode path. The fresh
measurement says the whole-frame win is now far from sub-percent.

Methodology: `--measurement-time 30 --warm-up-time 3` (no `--quick`),
three interleaved scalar/simd run pairs per bench so thermal drift
cancels, nightly 1.97 toolchain for both A/B columns so the compiler
version cancels, separate `CARGO_TARGET_DIR`s per config, no concurrent
load. Stable 1.95 default-features runs provide the default-path anchor.
`keyframe_decode` rides along as the other half of the keyframe path
(the SIMD kernels mostly live on reconstruct/decode).

| Bench | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ (simd vs scalar) |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 5.454 ms | 5.385 / 5.482 / 5.500 ms (mean 5.456) | 4.928 / 4.976 / 4.967 ms (mean **4.957**) | **−9.2 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 154.7 µs | 151.0 / 152.7 / 151.6 µs (mean 151.8) | 120.6 / 119.6 / 119.8 µs (mean **120.0**) | **−21.0 %** |

Throughput: keyframe encode 14.08 → **15.49 Mpx/s** (+10 %); keyframe
decode 506 → **640 Mpx/s** (+27 %). Stable default agrees with the
nightly scalar column within run-to-run spread on both benches, so the
whole delta is the `simd` dispatch, not the compiler version. Every
scalar run sits in 5.385–5.500 ms and every simd run in 4.928–4.976 ms
— the two populations don't overlap, so the deltas are far outside the
measurement envelope (criterion's own change estimates between
consecutive same-config runs were ≤ 2 %).

Attribution (`sample(1)` PID-attach, 10 s, 1 ms interval, on the encode
bench under `--measurement-time 60`): the scalar profile's #2 self-time
symbol is `inverse_transform::inverse_dct_4x4` at 1350 samples (≈ 16 %
of in-process time — it fires 24× per MB inside the §11 RD loop's
reconstruct leg *and* per coded block in the final reconstruct); under
`simd` that symbol disappears from the ≥ 5-sample top-of-stack list
entirely (the `Simd<i32, 4>` body inlines into
`encode_mb_block_set_with_neighbors`). Secondary movers:
`intra_predict::predict_y16x16_tm` 32 → 7 samples (the round-268
−87.7 % kernel) and `inverse_wht_4x4` 6 → absent. The takeaway over the
round-180 micro-bench (which measured the inverse DCT SIMD at only −1
to −5 % per isolated call): in the real inlined context — back-to-back
calls over 24 sub-blocks with surrounding dequant + add-and-clamp code
— the vectorised body wins far more than the isolated-call number
suggested. Micro-bench deltas under-predict inlined whole-frame deltas
in both directions (round 226 saw the reverse); whole-frame A/B with
extended measurement time is the deciding instrument.

No code change shipped this round (measurement + attribution only), so
bit-identity is structural; the full stable lib suite (483) + nightly +
`simd` lib suite (485) were re-run green as a sanity anchor.

## Round 279 — whole-frame inter path under nightly + `simd`, and the MB-batched sub-pixel SAD

Round 279 (2026-06-11) closes the round-278 next-round candidate: the
symmetric whole-frame `inter_encode_short_clip` A/B under nightly +
`simd`, same methodology (`--measurement-time 30 --warm-up-time 3`,
three interleaved scalar/simd run pairs, nightly 1.97 for both columns,
separate `CARGO_TARGET_DIR`s, stable 1.95 default-features anchor).

### Measurement (pre-change crate state)

| Bench | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ (simd vs scalar) |
|---|---:|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 8.907 ms | 8.977 / 9.035 / 9.061 ms (mean 9.024) | 8.823 / 8.905 / 8.850 ms (mean **8.859**) | **−1.8 %** |

The populations don't overlap (every scalar run ≥ 8.977 ms, every simd
run ≤ 8.905 ms) so the −1.8 % is real — but it's an order of magnitude
smaller than the keyframe path's −9.2 %, and the attribution explains
why. `sample(1)` PID-attach (10 s, 1 ms interval) on the **scalar**
build puts `motion_comp::sixtap_2d` at #1 self-time (2162 of ~7700
in-process samples); on the **simd** build the same symbol is *still*
#1 at 2186 samples — flat. The call tree shows where it lives: the §17
sub-pixel refinements (`half_pixel_refine_luma` +
`quarter_pixel_refine_luma`) are ≈ 38 % of in-process time, almost all
inside `mb_luma_sad_at_mv`, which synthesised each of the 17 candidate
predictions per searched MB (center + 8 half-pel + 8 quarter-pel) as
**sixteen separate `filter_block_4x4` calls** — sixteen overlapping 9×9
`fetch_block_halo` fetches + sixteen 4×4 `sixtap_2d` convolutions per
candidate. In that shape the 4×4 `sixtap_2d_simd` kernel buys nothing
whole-frame (the rounds 269–271 MB-batched kernels only served the
*reconstruct* leg, one MB-pass per coded MB; the search leg dwarfs it
at 17 candidate syntheses per MB). The simd column's whole −1.8 % is
the §14 transform/dequant kernels on the RD/reconstruct leg
(`inverse_dct_4x4` 277 scalar samples → absent under `simd`), the
round-278 finding scaled to a 4-frame inter clip.

### The flagged hotspot, and the one targeted change

The measurement flags a clear scalar-shape hotspot that is *not* a
missing SIMD kernel: `mb_luma_sad_at_mv` should synthesise a candidate
the same way `predict_inter_mb`'s luma half already does. All sixteen
luma sub-blocks of a non-SPLITMV candidate share the one MV (§18.1), so
the round-270/271 MB-batched primitives apply verbatim:

* whole-pixel candidate (`stored_luma_mv(mv) & 7 == 0` both axes) →
  one contiguous [`fetch_luma_mb_whole_pixel`] 16×16 fetch;
* sub-pixel candidate → one 21×21 [`fetch_luma_mb_halo`] fetch + one
  whole-MB [`sixtap_mb_luma`] §18.3 pass.

Both are byte-exact with the per-sub-block tiling by the existing
equivalence proofs (`predict_inter_mb_sub_pixel_matches_per_block`,
`fetch_luma_mb_whole_pixel_matches_per_subblock_in_bounds`, the border
clamp tests, and the `sixtap_mb_luma` scalar/simd stress tests), so
every candidate SAD — and therefore every MV decision and every emitted
bit — is unchanged.

### Measured A/B (same interleaved extended-measurement methodology)

| Bench (post-change) | Stable default | Nightly scalar (3 runs) | Nightly + `simd` (3 runs) | Δ vs pre-change |
|---|---:|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 7.431 ms (**−16.6 %**) | 7.459 / 7.320 / 7.326 ms (mean 7.368, **−18.3 %**) | 6.955 / 6.952 / 6.953 ms (mean **6.953**, **−21.5 %**) | — |

Throughput: stable default 7.36 → **8.82 Mpx/s**; nightly scalar 7.26 →
**8.90 Mpx/s**; nightly simd 7.40 → **9.43 Mpx/s**. The simd-vs-scalar
whole-frame gap widens from −1.8 % to **−5.6 %** — the search leg now
runs through `sixtap_mb_luma`, whose SIMD body finally gets to act on
the dominant call site. Post-change simd attribution: `sixtap_2d`
disappears from the ≥ 5-sample top-of-stack list entirely;
`fetch_luma_mb_halo` shows 410 samples and `mb_luma_sad_at_mv` 1510
(the inlined batched kernel), down from the ~2960-sample refinement
cluster.

Bit-identity was checked three ways: the full stable suite (483 lib +
integration, 36 green targets), the nightly + `simd` lib suite (485),
and a 54-frame byte-hash A/B — 3 resolutions (64×48 / 128×128 /
176×144) × 6 frames × 3 quantisers (qi 12 / 32 / 96), keyframe
interval 3 so each stream carries two K + four P frames, all through
`Vp8InterStreamEncoder` — FNV-1a `2f655ee6d2a8a303` identical
pre-/post-change on stable *and* under nightly + `simd`.

## Round 281 — fused whole-pixel SAD scoring (`mb_luma_sad_at_whole_mv` / `group_sad_at_whole_mv`)

Round 281 (2026-06-12) closes the round-279 standing candidate
"whole-pixel SAD scoring batching". A fresh `sample(1)` PID-attach
profile (12 s, 1 ms interval, nightly + `simd` build under
`--measurement-time 60`) reproduced the round-279 finding as the
current #1 / #3 self-time symbols:

| Self-samples (of ~9276) | Symbol |
|---:|---|
| 2290 | `motion_comp::fetch_block_whole_pixel` |
| 1784 | `motion_search::mb_luma_sad_at_mv` (sub-pixel leg, round-279 shape) |
| 1371 | `encoder::group_sad_at_whole_mv` |
| 590 | `encoder::encode_p_frame_multi_ref_inner_with_counts_and_pick` |
| 458 | `motion_comp::fetch_luma_mb_halo` |

The §17 integer-pixel diamond descent (`mb_luma_sad_at_whole_mv`) and
the SPLITMV group scorer (`group_sad_at_whole_mv`) still fetched
sixteen (or per-group fewer) separate 4×4 patches per whole-pixel
candidate — sixteen `fetch_block_whole_pixel` bounds checks + scratch
copies per candidate, plus (on the SPLITMV side) a per-member 4×4
source extraction copy. Both scorers now run a **fused fetch-and-SAD**
(the round-279 candidate's parenthetical "a fused fetch-and-SAD that
never materialises the patch"): a whole-pixel candidate's §18.3
prediction is "simply copied", i.e. a direct window into the reference
plane at integer offset `(mb_x, mb_y) + (eighth-pixel mv >> 3)`, so
when the 16×16 MB-extent source region lands strictly in-bounds (the
dominant case mid-frame — and a conservative gate for every SPLITMV
group member) the SAD accumulates straight off the reference and
source rows. No prediction block is built and no source sub-block is
extracted. Border-straddling candidates fall back to the batched
`fetch_luma_mb_whole_pixel` (non-SPLITMV) or the original per-member
`fetch_block_whole_pixel` assembly (SPLITMV groups) — §20.14
`build_mc_border` edge replication preserved bit-for-bit.

### Measured A/B (`--measurement-time 30 --warm-up-time 3`, three interleaved pre/post pairs per config, Apple M4 / aarch64)

| Config | Pre (3 runs) | Post (3 runs) | Δ (means) |
|---|---:|---:|---:|
| stable 1.95 default | 7.403 / 7.183 / 7.130 ms (mean 7.239) | 6.317 / 6.152 / 6.130 ms (mean **6.200**) | **−14.4 %** |
| nightly 1.97 scalar | 7.347 / 7.242 / 7.153 ms (mean 7.247) | 6.308 / 6.247 / 6.139 ms (mean **6.231**) | **−14.0 %** |
| nightly 1.97 + `simd` | 6.910 / 6.818 / 6.775 ms (mean 6.834) | 5.876 / 5.815 / 5.758 ms (mean **5.816**) | **−14.9 %** |

Throughput: stable 9.05 → **10.57 Mpx/s**; nightly scalar 9.04 →
**10.52 Mpx/s**; nightly + `simd` 9.59 → **11.27 Mpx/s**. Every pre
run sits ≥ 7.119 ms (scalar) / ≥ 6.775 ms (simd) and every post run
≤ 6.376 ms / ≤ 5.904 ms — the populations don't overlap in any
config, so the deltas are far outside the measurement envelope. The
pre columns agree with the round-279 post-change numbers (7.43 /
7.37 / 6.95 ms) within session drift.

Post-change attribution (same `sample(1)` methodology):
`fetch_block_whole_pixel` collapses 2290 → **356** self-samples (the
residue is the SPLITMV border fallback + the per-sub-block reconstruct
paths); the fused SAD work now shows in its callers' own bodies
(`group_sad_at_whole_mv` 2007, `mb_luma_sad_at_whole_mv` 179 — tight
row loops instead of call + copy + re-read), and the new #1 is the
round-279-shaped sub-pixel `mb_luma_sad_at_mv` leg (2103), which
already runs the MB-batched §18.3 kernels.

Bit-identity was checked three ways: the full stable suite (38 green
targets, lib 483 → 486 with the new equivalence anchors
`whole_mv_sad_matches_per_subblock_fetch_assembly` — every MB of a
3×2-MB plane × 13 whole-pixel candidates including §17.1-extreme
border-straddlers in all four directions —
`whole_mv_sad_matches_assembly_with_padded_stride`, and
`group_sad_fused_fast_path_matches_per_subblock_fetch` — all four
§16.4 partition shapes × every group × 11 candidates × 3 MB
positions); the nightly + `simd` lib suite (485 → 488); and a 54-frame
byte-hash A/B — 3 resolutions (64×48 / 128×128 / 176×144) × 6 frames
× 3 quantisers (qi 12 / 32 / 96), keyframe interval 3, all through
`Vp8InterStreamEncoder` — FNV-1a `1495730e2f66b0a3` (17 273 bytes)
identical pre-/post-change on stable *and* under nightly + `simd`.
(The hash differs from the round-279 record because the round-281
harness drives a different deterministic source — a drifting gradient
plus a moving high-contrast square so SPLITMV does real work; what
matters is pre == post within the round, on both toolchains.)

## Round 282 — decoder-side bench coverage + full-suite refresh (bench round)

Round 282 (2026-06-12) is a bench-only round: no `src/` change (the
decoder and encoder are byte-identical to round 281). Three things
landed.

### 1. Two new decoder-side benches

The bench suite had whole-frame coverage for keyframe decode but
nothing for the §16 **inter decode** path, and the §15 loop filter was
only covered at the per-edge primitive layer (`loop_filter_normal`,
`loop_filter_mb_edge`) — never as the whole-frame §20.6 pass the
decoder runs per frame.

* **`inter_decode_short_clip/inter_decode_4f_128x128_qi32`** — a
  `Vp8DecoderState` consuming the one-K + three-P 128×128 stream that
  `Vp8InterStreamEncoder` produces from the same deterministic drift
  clip the `inter_encode_short_clip` bench encodes (the stream is built
  once in setup, unmeasured). Per iteration this runs §16.1 ref
  selection, §16.2/§17 MV decode, §18 motion compensation (whole-pixel
  copies + §18.3 six-tap synthesis), §14 transforms, and the §15.1
  inter loop filter — the decode half of the inter roundtrip.
* **`loop_filter_frame/{filter_frame_keyframe_320x240_normal,
  filter_frame_keyframe_320x240_simple,
  filter_inter_frame_320x240_normal}`** — the whole-frame §15 pass
  (per-MB §20.6 level resolution + §15.1 skip rules + the raster
  MB-edge / sub-block-edge cascade over Y, U, V) on a 20×15-MB frame
  whose every MB carries a coded coefficient (the fully-coded worst
  case; level 26, sharpness 0). The planes carry a per-MB checkerboard
  DC offset so every edge does real clamp work. Each iteration runs on
  a fresh clone via `iter_batched` (clone unmeasured). Note the
  whole-frame decode benches don't exercise this worst case — their
  synthetic streams resolve to light filtering (the §15 symbols don't
  reach the decode profiles' top-of-stack lists) — so this bench is
  the standing A/B instrument for any future §15 rewrite.

### 2. Full-suite refresh (post r279–r281 state)

All numbers re-measured this round on the same machine: Apple
M4-class aarch64, macOS 25.1; **stable** = rustc 1.95.0 default
features, **nightly + `simd`** = 1.97.0-nightly with the `simd`
feature. Whole-frame benches under `--measurement-time 30
--warm-up-time 3`; micro-benches under `--quick`; separate
`CARGO_TARGET_DIR` per config. Micro-bench rows compare across the
two columns only loosely (different compiler versions); rows whose
kernels have no `simd` dispatch (loop filter, intra DC/V/H, SAD,
SPLITMV write strategies) differ by toolchain alone.

| Bench | Stable (1.95) | Nightly + `simd` (1.97) |
|---|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 5.297 ms (14.4 Mpx/s) | **5.118 ms** (15.0 Mpx/s) |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 154.2 µs (500 Mpx/s) | **122.9 µs** (622 Mpx/s, −20 %) |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 6.182 ms (10.6 Mpx/s) | **5.947 ms** (11.0 Mpx/s) |
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` (NEW) | 188.2 µs (348 Mpx/s) | **161.7 µs** (405 Mpx/s, −14 %) |
| `loop_filter_frame/filter_frame_keyframe_320x240_normal` (NEW) | 334.6 µs | 376.2 µs |
| `loop_filter_frame/filter_frame_keyframe_320x240_simple` (NEW) | 78.5 µs | 84.9 µs |
| `loop_filter_frame/filter_inter_frame_320x240_normal` (NEW) | 346.9 µs | 352.4 µs |
| `inverse_transform_4x4/inverse_dct_4x4` | 10.14 ns | 9.82 ns |
| `inverse_transform_4x4/inverse_wht_4x4` | 9.41 ns | 7.73 ns |
| `forward_transform_4x4/forward_dct_4x4` | 10.97 ns | 10.80 ns |
| `forward_transform_4x4/forward_wht_4x4` | 10.80 ns | 9.00 ns |
| `intra_predict_dc16/predict_y16x16_dc` | 4.76 ns | 4.96 ns |
| `intra_predict_dc16/predict_y16x16_v` | 3.64 ns | 3.74 ns |
| `intra_predict_dc16/predict_y16x16_h` | 3.73 ns | 3.95 ns |
| `intra_predict_dc16/predict_y16x16_tm` | 44.72 ns | **5.89 ns** |
| `loop_filter_normal/subblock_filter_4_4` | 2.14 ns | 2.23 ns |
| `loop_filter_normal/simple_segment_4` | 2.10 ns | 2.33 ns |
| `loop_filter_mb_edge/mb_filter_wide` | 5.71 ns | 5.83 ns |
| `loop_filter_mb_edge/mb_filter_hev` | 5.28 ns | 6.26 ns |
| `loop_filter_mb_edge/subblock_filter_low_variance` | 5.06 ns | 5.31 ns |
| `loop_filter_mb_edge/common_adjust_outer_taps` | 2.87 ns | 3.01 ns |
| `loop_filter_mb_edge/common_adjust_no_outer` | 2.66 ns | 2.76 ns |
| `motion_comp_subpel_luma/filter_block_4x4_sub3x5` | 24.73 ns | 26.37 ns |
| `motion_comp_subpel_luma/mb_sixtap_2d_16x4x4` | 264.8 ns | 270.3 ns |
| `motion_comp_subpel_luma/mb_luma_batched_16x16` | 158.8 ns | **142.3 ns** |
| `motion_comp_subpel_luma/mb_luma_per_subblock_16x16` | 264.5 ns | 265.6 ns |
| `motion_comp_subpel_luma/mb_chroma_batched_8x8` | 43.7 ns | **40.3 ns** |
| `motion_comp_subpel_luma/mb_chroma_per_subblock_8x8` | 66.4 ns | 70.0 ns |
| `motion_comp_subpel_luma/mb_luma_whole_pixel_batched_16x16` | 13.4 ns | 14.1 ns |
| `motion_comp_subpel_luma/mb_luma_whole_pixel_per_subblock_16x16` | 47.1 ns | 49.5 ns |
| `motion_comp_subpel_luma/mb_chroma_whole_pixel_batched_8x8` | 4.79 ns | 5.27 ns |
| `motion_comp_subpel_luma/mb_chroma_whole_pixel_per_subblock_8x8` | 8.53 ns | 9.46 ns |
| `motion_comp_subpel_luma/splitmv_predict_scratch_copy` | 379.0 ns | 388.1 ns |
| `motion_comp_subpel_luma/splitmv_predict_strided_write` | 455.3 ns | 484.6 ns |
| `motion_search_descent/small_diamond_search_luma_iters_8` | 82.2 ns | 88.6 ns |
| `motion_search_descent/half_pixel_refine_luma_8_offsets` | 1.494 µs | 1.409 µs |
| `motion_search_descent/quarter_pixel_refine_luma_8_offsets` | 1.490 µs | 1.461 µs |
| `motion_search_descent/full_descent_whole_half_quarter` | 3.065 µs | 2.867 µs |
| `motion_search_descent/block_sad_16x16_single_pair` | 6.31 ns | 6.83 ns |

Headline reads: the whole-frame numbers confirm the r278–r281 state
within session drift (keyframe encode ≈ 5.3 ms, inter encode ≈ 6.2 ms
stable, keyframe decode −20 % under `simd` matching round 278's
−21 %). The new inter-decode number establishes the baseline at
**188 µs / 348 Mpx/s stable, 162 µs / 405 Mpx/s under `simd`** for
the 4-frame 128×128 clip. The `rate_control_qi_sweep` byte column was
re-run as an encoder regression sanity check: all ten outputs are
byte-identical to the round-194 record (1701 / 676 / 612 / 595 / 480 /
466 / 461 / 360 / 355 / 299 B) while the wall column dropped from
7.3–9.3 ms to 5.3–5.7 ms — the accumulated r204–r281 encoder wins at
constant output.

### 3. Decoder profile evidence + ranked candidates

`sample(1)` PID-attach profiles (12 s, 1 ms interval, bench looping
under `--measurement-time 60`) of `keyframe_decode` (stable),
`inter_decode_short_clip` (stable), and `inter_decode_short_clip`
(nightly + `simd`). Top self-time symbols:

| keyframe stable | inter stable | inter simd |
|---|---|---|
| `inverse_dct_4x4` 2178 | `decode_block` 1919 | `decode_block` 2265 |
| `decode_block` 1823 | `inverse_dct_4x4` 1919 | `reconstruct_inter_mb` 2086 |
| `decode_keyframe_mb_non_bpred` 1406 | `memmove` 1125 | `memmove` 1054 |
| `memmove` 945 | `reconstruct_inter_mb` 1037 | `decode_mb_coeffs` 616 |
| `decode_mb_coeffs` 570 | `decode_mb_coeffs` 559 | `decode_frame` 550 |
| `predict_y16x16_tm` 507 | `decode_frame` 487 | `parse_token_prob_update` 445 |
| `parse_mb_modes` 333 | `parse_token_prob_update` 361 | `decode_keyframe_mb_non_bpred` 309 |

Call-tree attribution of the inter-stable `memmove`/`memset` family
(~1100 samples): ≈ 624 under `Vp8DecoderState::decode_frame`'s own
body and ≈ 393 under `RefFrameSlot::clone` — i.e. essentially all of
it is per-frame reference-slot plane copying, plus ≈ 59 in
`crop_to_visible`. Ranked next decoder candidates:

1. **§13 token decode (`dct_tokens::decode_block` +
   `decode_mb_coeffs`)** — #1 under `simd` (2265 + 616 of ~9300
   in-process samples, ≈ 31 %) and #1/#2 in every profile. The
   per-coefficient bool-decoder tree descent is the decoder's mirror
   of the encoder hot path that rounds 170/204/276 collapsed; the same
   playbook applies (branch-reduced descent, precomputed
   tree-walk tables, batched context fetch for the §13.2 band/has-coeff
   state). Biggest single lever on both decode benches.
2. **Per-frame reference-slot copy churn in
   `Vp8DecoderState::decode_frame`** — the `memmove` family is #3 in
   both inter profiles (≈ 11–14 % of inter decode), attributed to
   `RefFrameSlot::clone` + in-body plane copies when refreshing LAST /
   GOLDEN / ALTREF. Candidate: share or swap the plane buffers
   (slot-swap when refresh flags allow, or reference-counted planes
   with copy-on-write) instead of cloning whole planes per frame.
   Relative weight grows as frames shrink; still visible at 320×240.
3. **`coded_header::parse_token_prob_update`** — 361 (stable) / 445
   (simd) self-samples ≈ 5 % of inter decode: the per-frame fixed cost
   of reading the 4×8×3×11 §13.4 update-flag bools. Candidate: a
   specialised flag-read loop (the flags are overwhelmingly false at
   these probs — hoist the bool-decoder state and inline the
   common-path renormalise), amortised once per frame so it matters
   most for small-frame / high-fps streams.

The stable-only `inverse_dct_4x4` #1 (keyframe profile, 2178 samples
≈ 16 %) is already solved by the existing `simd` feature (round 278
measured it gone under `simd`); it stays closed rather than ranked.

## Round 283 — fused §13 token descent + batched bool-decoder renormalisation

Round 283 (2026-06-12) takes the round-282 ranked list's #1: §13 token
decode (`dct_tokens::decode_block` + `decode_mb_coeffs`, ≈ 31 % of
inter decode under `simd`, #1/#2 self-time in every decoder profile).
The decoder-side mirror of the round-204 encoder playbook ("the
descent is a pure function of a fixed tree — stop walking the table"),
in three pieces, all output-bit-identical:

1. **Fused branch-coded §13.2 descent** (`decode_block_core`). The
   per-coefficient `treed_read_coef` walked the 22-entry `COEFF_TREE`
   table step-by-step, returned a `DctToken` enum, then re-dispatched
   on it twice (EOB check + magnitude match). The tree is fixed, so
   the descent is now written out branch-by-branch — each `read_bool`
   sits at one named tree node reading `probs[node >> 1]` exactly as
   the generic §8.1 walk would, and each leaf flows straight into its
   consequence (zero / small value / `DCTextra` category / EOB). The
   §13.2 "skip dct_eob after DCT_0" rule becomes an inner zero-run
   loop re-entering at the DCT_0 node with the §13.3 zero-class row,
   eliminating the `prevCoeffWasZero` flag and the per-position
   restart entirely.
2. **Write-order-table raster output**. `decode_mb_coeffs` decoded
   each block into a scan-order scratch `[i16; 16]`, then ran a
   16-lane `scan_to_raster` permute and copied the result into the
   `MbCoeffs` block. The core now takes a write-order table
   (`ZIGZAG` for the MB walk, identity for the public scan-order
   `decode_block`) and lands every coefficient in its raster slot as
   it is decoded — no scratch, no permute pass, no return copy.
3. **Batched bool-decoder renormalisation**. The §7.3 listing doubles
   `range`/`value` one bit at a time until `range >= 128` — up to 7
   dependent loop iterations per `read_bool`. The doubling count is a
   pure function of `range` (`leading_zeros() - 24`), and at most one
   input byte can be needed per renormalisation, so the loop is now a
   single shift + one conditional byte splice at bit offset
   `bit_count + shift - 8`. This one is decoder-wide: every §9/§10/§11
   header bool, §13.4 `parse_token_prob_update` flag, mode/MV tree and
   token bit goes through it.

### Measured A/B (`--measurement-time 30 --warm-up-time 3`, three interleaved pre/post pairs per config, Apple M4 / aarch64)

| Bench / config | Pre (3 runs) | Post (3 runs) | Δ (means) |
|---|---:|---:|---:|
| `keyframe_decode` stable 1.95 | 154.50 / 152.85 / 151.60 µs (mean 152.98) | 143.32 / 142.73 / 139.81 µs (mean **141.95**) | **−7.2 %** |
| `keyframe_decode` nightly 1.97 + `simd` | 123.58 / 121.95 / 118.67 µs (mean 121.40) | 110.96 / 110.11 / 107.91 µs (mean **109.66**) | **−9.7 %** |
| `inter_decode_short_clip` stable 1.95 | 192.45 / 189.07 / 185.85 µs (mean 189.12) | 177.10 / 175.18 / 170.84 µs (mean **174.37**) | **−7.8 %** |
| `inter_decode_short_clip` nightly 1.97 + `simd` | 161.13 / 159.99 / 158.18 µs (mean 159.77) | 144.70 / 144.54 / 149.52 µs (mean **146.25**) | **−8.5 %** |

Throughput: keyframe decode 502 → **541 Mpx/s** stable, 633 →
**700 Mpx/s** simd; inter decode 347 → **376 Mpx/s** stable, 410 →
**448 Mpx/s** simd. The pre/post populations don't overlap in any
config (worst margin: inter simd, every pre ≥ 158.18 µs vs every post
≤ 149.52 µs), so the deltas sit far outside the measurement envelope.
The pre columns agree with the round-282 baselines within session
drift.

Post-change attribution (same `sample(1)` methodology, 12 s @ 1 ms on
the nightly + `simd` inter bench, ~9237 in-process samples): the §13
pair drops from 2881 (≈ 31 %) to `decode_block_core` 2060 +
`decode_mb_coeffs` 175 = **2235 (≈ 24 %)** and cedes #1 to
`reconstruct_inter_mb` (2242); the per-frame reference-slot `memmove`
family (1236 + memset/bzero ≈ 274) is now the clearest next lever,
with `parse_token_prob_update` at 534 (≈ 6 %) behind it.

Bit-identity was checked three ways: the full stable suite (38 green
targets, lib 486 → 488 with the new equivalence anchors
`batched_renormalize_matches_bit_at_a_time_listing` — a 4096-bool
xorshift prob/bit stream decoded in lockstep against the literal §7.3
bit-at-a-time listing with the full decoder state asserted after every
read — and `fused_descent_matches_generic_tree_walk` — 400 randomised
blocks across every plane type, neighbour-context seed, and magnitude
class, fused vs generic `COEFF_TREE` walk, asserting coefficients,
non-zero counts, and the complete bool-decoder end state); the
nightly + `simd` lib suite (488 → 490); and a 55-frame decode-side
byte-hash A/B — 3 resolutions (64×48 / 128×128 / 176×144) × 6 frames
× 3 quantisers (qi 12 / 32 / 96) at keyframe interval 3 decoded
through `Vp8DecoderState`, plus the bench's exact 320×240 keyframe
through `decode_vp8`, hashing every decoded Y/U/V plane byte
(1 324 800 bytes) — FNV-1a `ec93aa4f7f728ebe` identical
pre-/post-change on stable *and* under nightly + `simd`.

## Round 285 — harnesses for the r283/r284 hot paths + ranked hotspot table (bench round)

Round 285 (2026-06-12) is a bench-only round: no `src/` change (decoder
and encoder are byte-identical to round 284 — structural bit-identity,
re-anchored by the full stable lib suite, 488 green). Three new bench
binaries close the harness gaps the round-283 profile left, and two
fresh `sample(1)` profiles produce the ranked table below.

### 1. `token_decode` — the fused §13.2 descent in isolation + an inter-heavy stream

The round-283 fused descent (`decode_block_core`) had no isolated A/B
target — both whole-frame decode benches wrap it in prediction +
transforms + loop filter. The new harness drives `decode_mb_coeffs`
directly over 64-MB token partitions produced by the crate's own §13
encoder (`TokenEncoder`), decoded once at setup with a full
coefficient-equality assertion so the measured loop is a proven-valid
bitstream walk:

* **dense** — every block carries an 11-coefficient run sweeping the
  §13.2 token classes through Cat4 (the descent + extra-bits + sign
  worst case);
* **sparse** — zero-runs + early/immediate EOBs (the §13.2
  "skip dct_eob after DCT_0" inner-loop shape of well-predicted inter
  residue).

Plus `inter_decode_12f_176x144_token_heavy`: a whole-stream decode of a
1-keyframe + 11-P-frame 176×144 clip (textured drifting gradient + a
travelling high-contrast checker square) — the round-283 target
workload at a P-frame share three times `inter_decode_short_clip`'s.

| Bench | Stable (1.95) | Nightly + `simd` (1.97) |
|---|---:|---:|
| `token_decode/decode_mb_coeffs_dense_64mb` | 501.7 µs (7.84 µs/MB) | 494.8 µs |
| `token_decode/decode_mb_coeffs_sparse_64mb` | 48.16 µs (0.75 µs/MB) | 47.90 µs |
| `token_decode/inter_decode_12f_176x144_token_heavy` | 1.359 ms (224 Mpx/s) | **1.091 ms** (279 Mpx/s, −20 %) |

The dense/sparse 10× spread is the per-token cost surface: a dense MB
decodes ~330 tokens (≈ 24 ns/token end-to-end through context updates),
a sparse MB ~50. The micro pair is toolchain-flat (no `simd` dispatch
in the token layer), while the whole-stream number inherits the §14/§18
kernel wins.

### 2. `reconstruct_inter_mb` — the round-283 #1 decoder symbol

The §14.2/§18 per-MB inter reconstruction orchestrator in its three
workload shapes (interior MB, in-bounds fast paths):

| Bench | Stable (1.95) | Nightly + `simd` (1.97) |
|---|---:|---:|
| `reconstruct_inter_mb/subpel_full_residue` | 569 ns | **464 ns** (−18 %) |
| `reconstruct_inter_mb/whole_pixel_full_residue` | 424 ns | **344 ns** (−19 %) |
| `reconstruct_inter_mb/subpel_skip` | 280 ns | 273 ns |

Decomposition reading: the whole-pixel prediction is ~19 ns of the
344 ns whole-pixel row (`mb_luma_whole_pixel_batched_16x16` 14 ns +
chroma 5 ns), so the §14.3 WHT + 24 × §14.4 IDCT + §14.5
extract-add-insert residue chain is ≈ **325 ns — ~95 % of the
whole-pixel reconstruct and ~70 % of the sub-pixel one**. At 24 IDCTs
× ~10 ns ≈ 240 ns, the remaining ~85 ns is the per-sub-block
`extract_4x4` → `add_residue_4x4` → `insert_4x4` scratch round-trip —
the concrete fusion surface the ranked table below names.

### 3. `subpel_sad_scoring` — the encoder's #1 cluster, decomposed

The micro-bench the standing "sub-pixel SAD without patch
materialisation" candidate asked for — one §17 refinement candidate
through `mb_luma_sad_at_mv`, plus the 21×21 halo fetch alone:

| Bench | Stable (1.95) | Nightly + `simd` (1.97) |
|---|---:|---:|
| `subpel_sad_scoring/mb_luma_sad_at_mv_half_pel` | 187.8 ns | 179.9 ns |
| `subpel_sad_scoring/mb_luma_sad_at_mv_quarter_pel` | 184.5 ns | 183.0 ns |
| `subpel_sad_scoring/mb_luma_sad_at_mv_whole_pel` | 18.5 ns | 20.0 ns |
| `subpel_sad_scoring/fetch_luma_mb_halo_21x21_in_bounds` | 22.2 ns | 21.9 ns |

A sub-pixel candidate costs ≈ 184 ns: fetch 22 + whole-MB §18.3
convolution ≈ 156 (`mb_luma_batched_16x16`) + SAD ≈ 6
(`block_sad_16x16_single_pair`). The fused row-SAD candidate can only
attack the convolution's narrow-to-u8 store + reload and the final
256-byte SAD read-back — call it ≤ 20 ns/candidate of headroom (~10 %),
× 16 sub-pixel candidates per searched MB. These rows are the A/B
instrument for that change; the whole-pel row (the round-281 fused
fetch-and-SAD) is the 10× cheaper shape the §17 ladder's diamond stage
rides.

### 4. Ranked hotspot table (fresh `sample(1)` profiles, 12 s @ 1 ms, nightly + `simd`, `--measurement-time 60`)

Decoder — `inter_decode_short_clip` (~10.1k in-process samples), with
the token-heavy stream's shift in parentheses:

| Rank | Symbol | Self-samples | Share | Status |
|---:|---|---:|---:|---|
| 1 | `motion_comp::reconstruct_inter_mb` | 2447 | ≈ 24 % | **next PROFILE-OPT target** (see below) |
| 2 | `dct_tokens::decode_block_core` | 2225 | ≈ 22 % | round-283-fused; grows to ≈ 47 % on the token-heavy stream (4340 of ~9.3k samples) — re-ranks #1 on dense-residue content |
| 3 | `memmove`/`memset` family | ≈ 1600 | ≈ 16 % | per-frame reference-slot plane copying (r283 attribution: `decode_frame` body + `RefFrameSlot::clone`) |
| 4 | `Vp8DecoderState::decode_frame` body | 542 | ≈ 5 % | slot bookkeeping around #3 |
| 5 | `coded_header::parse_token_prob_update` | 522 | ≈ 5 % | per-frame fixed cost; specialised flag-read loop still open |

Encoder — `inter_encode_short_clip` (~9.3k in-process samples):

| Rank | Symbol | Self-samples | Share | Status |
|---:|---|---:|---:|---|
| 1 | `mb_luma_sad_at_mv` + `fetch_luma_mb_halo` | 1992 + 564 | ≈ 28 % | sub-pixel SAD cluster — now decomposed by `subpel_sad_scoring`; fused row-SAD headroom ≈ 10 % of the cluster |
| 2 | `encoder::group_sad_at_whole_mv` | 2205 | ≈ 24 % | round-281 fused loops (work moved into the caller's own body — tight row SAD, not call churn) |
| 3 | RD/emit trio (`…_pick` 651 / `encode_mb_block_set_with_neighbors` 569 / `estimate_block_bits` 437) | 1657 | ≈ 18 % | token-emission side, already table-driven (r204/r276) |
| 4 | `forward_dct_4x4` + `transform_whole_block_luma` | 509 | ≈ 5 % | scalar-dispatch by design (r247) |

**Named next PROFILE-OPT target: `reconstruct_inter_mb` §14.4/§14.5
residue fusion.** #1 decoder self-time (≈ 24 %, and #2 even on
token-heavy content at ≈ 19 %), and the new micro-bench shows ~95 % of
its whole-pixel cost is the residue chain, of which ~85 ns/MB is pure
data movement: each of the 24 sub-blocks runs `extract_4x4` (strided
raster → `[u8; 16]` scratch), `add_residue_4x4`, `insert_4x4` (scratch
→ strided raster), plus a `[i16; 16]` residue scratch per IDCT. Fusing
the §14.4 inverse transform's output pass with the §14.5 add-clamp
directly over the prediction raster (strided, per sub-block row —
4-lane wide, matching the existing `inverse_dct_4x4_simd` layout)
removes both scratch round-trips at unchanged arithmetic. Byte-identity
provable with the existing stress-matrix pattern; expected impact spans
both decode benches and the encoder's RD-reconstruct leg
(`reconstruct_inter_mb` is 131 encoder samples, but
`encode_mb_block_set_with_neighbors` shares the same add-residue
shape). Runner-up: the rank-3 reference-slot copy churn (slot-swap /
copy-on-write planes), unchanged from the round-283 list.

## What didn't get touched yet (next-round candidates)

* **~~Remaining allocator churn (`malloc` / `free`)~~** — CLOSED in
  round 277 (see above): the churn was `partition_groups`'
  per-call `Vec<Vec<usize>>` + the `SplitMvCandidate` `Vec` pair, both
  now allocation-free. The residual ~40 call-tree allocator samples are
  the per-frame partition assembly (`BoolEncoder` output `Vec` growth
  in `stream`) — bounded per frame, not per MB, and not worth a
  speculative pre-reserve without a fresh profile pointing at it.
* **~~SPLITMV whole-pixel sub-block batching~~** — closed negative in round
  274 (see the round-274 section above): SPLITMV sub-blocks carry distinct
  vectors so the shared-halo batch can't apply, and the only remaining
  freedom (the per-sub-block write strategy) measured ~17 % SLOWER as a
  strided write than the shipped contiguous `[u8; 16]`-scratch copy. The
  candidate is retired.
* **~~Whole-frame `keyframe_encode` re-measure under nightly + `simd`~~**
  — CLOSED in round 278 (see above): measured **−9.2 %** whole-frame
  keyframe encode and **−21.0 %** keyframe decode under nightly +
  `simd`, attributed primarily to `inverse_dct_4x4_simd` in the §11
  RD-reconstruct loop (scalar profile's #2 self-time symbol at ≈ 16 %,
  gone under `simd`). The round-247 sub-percent prediction predated the
  round-267/268 dequant + TM_PRED kernels and under-counted the inlined
  inverse-DCT win.
* **~~Whole-frame `inter_encode_short_clip` re-measure under nightly +
  `simd`~~** — CLOSED in round 279 (see above): the pre-change simd
  dispatch was worth only **−1.8 %** whole-frame because the dominant
  §18.3 six-tap work sat in the *search* scoring path
  (`mb_luma_sad_at_mv`, 17 per-4×4-tiled candidate syntheses per MB),
  not the reconstruct leg the MB-batched kernels served. Routing that
  scoring path through the round-270/271 MB-batched primitives (bit-
  identical, 54-frame byte-hash proof) measured **−16.6 %** stable /
  **−18.3 %** nightly scalar / **−21.5 %** nightly simd whole-frame,
  and widened the simd-vs-scalar gap to **−5.6 %**.
* **~~Whole-pixel SAD scoring batching~~** — CLOSED in round 281 (see
  above): both whole-pixel scorers (`mb_luma_sad_at_whole_mv` and the
  SPLITMV `group_sad_at_whole_mv`) now run a fused fetch-and-SAD
  straight off the reference rows when the MB-extent source region is
  in-bounds, with the §20.14 border fallback unchanged. Measured
  **−14.4 %** stable / **−14.0 %** nightly scalar / **−14.9 %**
  nightly + `simd` whole-frame on `inter_encode_short_clip`;
  `fetch_block_whole_pixel` self-time collapsed 2290 → 356 samples;
  bit-identical (3 new equivalence anchors + 54-frame byte-hash A/B).
* **Decoder-side trio (round-282 profile)** — ~~§13 token decode~~
  CLOSED in round 283 (see above): fused branch-coded §13.2 descent +
  write-order-table raster output + batched bool-decoder
  renormalisation, **−7.2/−9.7 %** whole-frame keyframe decode and
  **−7.8/−8.5 %** inter decode (stable / nightly + `simd`),
  bit-identical (two new equivalence anchors + 55-frame decode-side
  byte-hash A/B). Still open from the trio: per-frame reference-slot
  plane copying in `Vp8DecoderState::decode_frame` (`memmove` family —
  post-round-283 it is the #3 self-time cluster at ≈ 1510 of ~9237
  samples with `reconstruct_inter_mb` now #1; candidate shape:
  slot-swap when refresh flags allow, or reference-counted planes
  with copy-on-write) and the §13.4 `parse_token_prob_update`
  per-frame fixed cost (534 samples ≈ 6 % — already cheapened in
  absolute terms by the round-283 batched renormalise; the remaining
  shape is the specialised flag-read loop).
* **Sub-pixel SAD without patch materialisation** — the post-round-281
  profile's #1 is the sub-pixel `mb_luma_sad_at_mv` leg (2103
  self-samples) + `fetch_luma_mb_halo` (590): each half-/quarter-pixel
  candidate still materialises the full 16×16 `sixtap_mb_luma` output
  before `block_sad_16x16` reads it back. A fused variant could SAD
  each output row as the vertical pass produces it (the row lives in
  an `i32` vector already under `simd`), skipping the narrow-to-u8
  store + reload — but the clamp-to-u8 must stay bit-exact, and the
  scalar path's auto-vectorisation may already cover most of the
  margin. The micro-bench it needed landed in round 285
  (`subpel_sad_scoring`): a sub-pixel candidate costs ≈ 184 ns
  (fetch 22 / convolve ≈ 156 / SAD ≈ 6), bounding the fusion headroom
  at roughly 10 % of the cluster — the round-258 lesson (SIMD leaf
  inlining regressing the surrounding descent) applies before
  committing.
* **`reconstruct_inter_mb` §14.4/§14.5 residue fusion** — NAMED next
  profile-opt target by the round-285 ranked table (see above): #1
  decoder self-time (≈ 24 %), with the round-285 micro-bench showing
  ~95 % of the whole-pixel reconstruct cost in the residue chain and
  ~85 ns/MB of it pure `extract_4x4`/`insert_4x4`/scratch data
  movement. Fuse the §14.4 inverse-DCT output pass with the §14.5
  add-clamp directly over the strided prediction raster; arithmetic
  unchanged, byte-identity provable with the existing stress-matrix
  pattern. Runner-up: reference-slot copy churn (slot-swap /
  copy-on-write planes, rank 3 at ≈ 15 %).

## Round 286 — fused §14.4 IDCT + §14.5 add-clamp residue pass

Profile-opt round landing the round-285 #1 decoder hotspot. The
per-sub-block residue chain in the three inter-reconstruction entry
points (`reconstruct_inter_mb`, `reconstruct_inter_mb_whole_pixel`,
`reconstruct_split_mv_mb`) was a four-buffer round-trip:

```
inverse_dct_4x4(coeffs → [i16;16] residue)
extract_4x4(strided plane → [u8;16] pred)
add_residue_4x4(pred, residue → [u8;16] summed)
insert_4x4([u8;16] summed → strided plane)
```

A single `inverse_dct_4x4_add_into(coeffs, plane, stride, sr, sc)`
helper now folds the second inverse-DCT pass with the §14.5 add-clamp
written straight into the strided prediction raster. Arithmetic is
unchanged; output is bit-identical (proved by
`fused_idct_add_into_matches_unfused_sequence` over a randomized
coefficient/position/predictor sweep on both paths, plus every in-tree
inter fixture still decoding byte-exact against its `expected.yuv`).

### Path split (round-274 lesson)

The §14.4 IDCT structurally emits the second pass one raster row at a
time, so a naive in-place strided fold stores 4 bytes per row. On the
**SIMD** path this folds cleanly: each row's predictor is loaded into a
`Simd<i32,4>`, the residue added lane-wide, `simd_clamp(0,255)` applied,
and the four bytes narrowed — faster than the unfused store. On the
**scalar** path the unfused sequence already auto-vectorises the full
16-element add-clamp as one chunk, and a row-at-a-time fold regressed it
(same reason the round-274 SPLITMV scratch-then-copy beat a strided
write ~23 %). So `inverse_dct_4x4_add_into` keeps the contiguous-buffer
sequence on the scalar path behind the same helper signature — callers
see one function, the default (stable) build is byte- and
speed-identical.

### Measured A/B (criterion, Apple M4 / aarch64; shared box)

`reconstruct_inter_mb` per-MB bench, nightly + `simd` (the path the
round-283/285 profile measured):

| shape                     | before  | after   | Δ      |
| ------------------------- | ------- | ------- | ------ |
| `subpel_full_residue`     | 462 ns  | 426 ns  | −8 %   |
| `whole_pixel_full_residue`| 323 ns  | 285 ns  | −12 %  |
| `subpel_skip`             | 266 ns  | 261 ns  | ~0 (no residue) |

Residue-pass-only cost (full minus skip): sub-pixel 196 → 165 ns
(−16 %), whole-pixel 57 → 24 ns (−58 %).

The new `idct_add_residue_fusion` bench times the unfused vs fused
24-sub-block MB residue pass in one binary (load-immune A/B): fused
179 ns vs unfused 187 ns under `simd` (−4 %); scalar parity (218 vs
216 ns, within noise — the scalar path is the unchanged sequence).

Next profile-opt target stays the round-285 runner-up: per-frame
reference-slot plane-copy churn in `Vp8DecoderState::decode_frame`
(slot-swap / copy-on-write planes, ≈ 15 % self-time).

## Round 288 — `ref_slot_rotation` micro + whole-stream bench (bench round)

Round 288 (2026-06-13) is a bench-only round: no `src/` change (decoder
and encoder byte-identical to round 286 — the full stable lib suite
re-anchors structural bit-identity). It builds the isolated harness for
the round-285/286 ranked decoder **runner-up** — the §9.7 / §9.8
reference-frame slot rotation (the `memmove` / `memset` family, #3
self-time cluster at ≈ 15–16 % on the inter decode profile). Every
whole-frame decode bench (`keyframe_decode`, `inter_decode_short_clip`,
`token_decode`) folded this `RefFrameSlot::clone` churn behind header
parse + token decode + reconstruct + loop filter, so the per-frame copy
cost had no A/B instrument.

### 1. `rotate_*` — the §20 page-147 rotation walk in isolation

The new `ref_slot_rotation` binary replicates the exact temporaries
`Vp8DecoderState::decode_frame` stages (`current_slot` +
`pre_{last,golden,altref}` + `new_{altref,golden,last}`, the
`copy_arf → copy_gf → refresh_gf → refresh_arf → refresh_last` order),
driven over public `RefFrameSlot` fields at the 320×240 geometry the
`keyframe_decode` bench uses (115 200 plane bytes/slot). Three flag
combinations partition the decoder's per-frame cost (Apple M4 / aarch64,
rustc 1.95.0 stable, `--measurement-time 12`):

| Bench | Time | Throughput |
|---|---:|---:|
| `ref_slot_rotation/rotate_refresh_last_only` | 12.2 µs | 8.76 GiB/s |
| `ref_slot_rotation/rotate_refresh_all` | 15.2 µs | 7.05 GiB/s |
| `ref_slot_rotation/rotate_copy_gf_arf` | 16.2 µs | 6.62 GiB/s |

Reading: even the round-149 hardwired P-frame ladder (`refresh_last = 1`,
everything else 0 — the overwhelmingly common case) costs **12.2 µs of
pure slot copying per frame**, because the staged sequence still clones
the three pre-rotation `Option<RefFrameSlot>` slots
(`pre_{last,golden,altref}`) plus `new_{altref,golden}` plus the
`current.clone()` into LAST — six populated-slot `Vec` clones where the
output keeps only one fresh LAST and two pass-through Options. The
heavier `refresh_all` (+25 %) and cross-slot `copy_gf_arf` (+33 %) shapes
add at most ~4 µs on top — the rotation cost is dominated by the
**unconditional** `pre_*` / `current` clones on every frame, not the rare
golden/altref branches.

### 2. `decode_*` — whole-stream, two refresh cadences

Two whole-stream decodes through `Vp8DecoderState` over the crate's own
§16 encoder output (1 keyframe + 8 P-frames, 320×240 drifting clip),
exercising the *shipped* rotation unmodified:

| Bench | Time | Throughput |
|---|---:|---:|
| `ref_slot_rotation/decode_1k8p_last_only` | 1.86 ms | 372 Mpx/s |
| `ref_slot_rotation/decode_1k8p_golden_altref` | 1.89 ms | 367 Mpx/s |

`golden_altref` adds a periodic `refresh_golden_frame` (every 2nd P) /
`refresh_alternate_frame` (every 3rd P) cadence on top of the default
refresh-last ladder. It is only **~1.3 % slower** whole-frame than
`last_only` — confirming the micro's finding from the whole-frame side:
the rotation is a near-**fixed** per-frame cost driven by `refresh_last`'s
unconditional current-slot clone, not by refresh frequency. The 12.2 µs
micro × 9 frames ≈ 110 µs of the 1.86 ms whole-stream number (≈ 6 % of
this stream's decode wall, scaling with frame area / MB count toward the
profile's ≈ 15 % at the inter bench's higher P-frame density).

### Ranked hotspot table (carried from round 285/286, re-confirmed)

The round-285 fresh `sample(1)` profiles still stand (no `src/` change
since round 286 touched only the §14.4/§14.5 residue fold, which moved
`reconstruct_inter_mb` down, not the rotation). Decoder inter profile,
post-round-286:

| Rank | Symbol | Share | Status |
|---:|---|---:|---|
| 1 | `dct_tokens::decode_block_core` | ≈ 22 % (≈ 47 % token-heavy) | round-283-fused; re-ranks #1 on dense residue |
| 2 | `motion_comp::reconstruct_inter_mb` | ≈ 20 % | round-286-fused residue pass (−8/−12 % per-MB) |
| 3 | `memmove`/`memset` — **ref-slot rotation** | ≈ 15 % | **now isolated by `ref_slot_rotation`** — NAMED next PROFILE-OPT target |
| 4 | `Vp8DecoderState::decode_frame` body | ≈ 5 % | slot bookkeeping around #3 |
| 5 | `coded_header::parse_token_prob_update` | ≈ 5 % | per-frame fixed cost |

**Named next PROFILE-OPT target: §9 reference-slot rotation copy-on-write
/ slot-swap.** The micro proves ~12.2 µs/frame of slot copying on the
common refresh-last path, six populated `Vec` clones where the rotation
semantically needs only a move of the current reconstruction into LAST
and pointer-aliasing of the unchanged GOLDEN / ALTREF slots. Candidate
shapes (any one bit-identical by construction — the rotation only ever
*selects* and *replaces* whole slots, never mutates plane bytes):

1. **`Rc`/`Arc`-backed planes with copy-on-write** — store
   `RefFrameSlot` planes behind a refcount so `pre_*` capture and the
   GOLDEN/ALTREF pass-through become pointer bumps; only the
   `current.clone()` into LAST stays a real copy (and even that can be a
   move of the just-decoded planes when `refresh_last` is the only bit).
2. **Slot-swap when refresh flags allow** — `std::mem::swap` /
   `Option::take` the current planes into LAST instead of cloning, since
   `planes` is owned and dropped right after the rotation.

The `ref_slot_rotation` micro is the A/B instrument for that change; the
two whole-stream rows bound the whole-frame ceiling (≈ 6 % on this
stream, ≈ 15 % at inter-bench P-frame density). Byte-identity provable
with the existing decode-side byte-hash A/B (the round-283 / round-286
55-frame FNV-1a harness). Runner-up after that: the §13.4
`parse_token_prob_update` per-frame flag-read loop (rank 5).

## Round 289 — move-minimising reference-slot rotation

Profile-opt round landing the round-288 named target: the §9 / §20-page-147
reference-frame slot rotation in `Vp8DecoderState::decode_frame` (the
`memmove` / `memset` #3 decoder self-time cluster, ≈ 15 %; round-288 micro
proved ~12.2 µs/frame of slot copying on the common refresh-last ladder).

### The change

The pre-289 rotation staged every input as a clone: the just-decoded frame
(`current_slot = { y: planes.y.clone(), … }`), the three entry slots
(`pre_last`/`pre_golden`/`pre_altref` = `self.*.clone()`), and again a clone
into each refreshed destination — up to six populated `Vec` clones per
frame even when only `refresh_last` is set. The rotation only ever *selects*
and *replaces* whole slots (it never mutates plane bytes), so it now:

1. **Resolves each destination to a symbolic source first**
   (`RotationSource ∈ {PreLast, PreGolden, PreAltRef, Current}`) following
   the §20 ordering (copy_arf → copy_gf → refresh_gf → refresh_arf →
   refresh_last; both copy cases read the *pre*-rotation slot set).
2. **Materialises by move**, cloning a source only when it genuinely feeds
   more than one destination (`is_last_use` test over the resolved source
   list). On the dominant `refresh_last`-only path this is **zero** plane
   copies: `Current` moves into `LAST`, `PreGolden`/`PreAltRef` move through.
3. **Crops the output frame from `planes` before the rotation**, so the
   just-decoded slot consumes `planes` by move instead of cloning it. The
   keyframe path applies the same move (one of its three slots takes
   `planes`; the other two still clone, since a key frame populates all
   three).

### Measured A/B (`ref_slot_rotation/decode_1k8p_*`, criterion `--measurement-time 8`, same-session, Apple M4 / aarch64, stable 1.95)

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `ref_slot_rotation/decode_1k8p_last_only` | 1.904 ms | **1.710 ms** | **−9.3 %** |
| `ref_slot_rotation/decode_1k8p_golden_altref` | 1.962 ms | **1.715 ms** | **−10.6 %** |

Throughput: refresh-last 363 → **404 Melem/s**; golden/altref cadence
352 → **403 Melem/s**. Criterion's own change estimate (stored-baseline,
same session) reports −9.3 % / −10.6 % at p < 0.05 ("Performance has
improved") on the post-change run. The `rotate_*` micro rows are
**unchanged** — that bench keeps a standalone clone-everything replica of
the old ladder inside the bench file (it does not call the crate's private
`rotate_reference_slots`), so it stays the documented baseline for the
copy cost the whole-stream change removes; only the `decode_1k8p_*` rows,
which drive the real `decode_frame` rotation, move.

### Bit-identity

The rotation only selects and replaces whole slots, so output is
bit-identical by construction. Anchored by the new
`rotation_matches_clone_everything_reference` (every entry-slot
population — each of LAST/GOLDEN/ALTREF populated-or-empty — ×
every refresh-control combination: `copy_buffer_to_alternate` ∈
{none, 0, 1, 2} × `copy_buffer_to_golden` ∈ {none, 0, 1, 2} ×
`refresh_golden`/`refresh_alternate`/`refresh_last` ∈ {none, false, true},
tagged-plane slots so the assertion checks *which* source landed in each
destination, `==` against the verbatim pre-289 clone-everything ladder),
plus the full stable lib suite (490) and nightly + `simd` lib suite (492),
the `i_frame_then_p_frame_64x64` / `golden_update_cycle` / `altref_arnr`
bit-exact decode tests, every in-tree decode/roundtrip integration test
(37 green binaries), and the `blackbox_oracle` black-box validator.

Next profile-opt target stays the round-288 runner-up: the §13.4
`parse_token_prob_update` per-frame flag-read loop (decoder rank 5, ≈ 5 %).
A copy-on-write (`Rc`/`Arc`-backed) plane representation remains a deeper
follow-on for the residual rotation cost (it would also remove the two
unavoidable keyframe clones and any genuine fan-out clone the move path
still pays), but is a public-API change to `RefFrameSlot`'s `Vec<u8>` plane
fields and so is out of scope for a single bit-identical profile-opt round.

## Round 291 — §13.4 `token_prob_update()` lockstep flag-read loop

Profile-opt round landing the round-288/289 named decoder target: the
§13.4 `coded_header::parse_token_prob_update` per-frame flag-read loop
(the standing decoder rank-5 self-time cluster, ≈ 5 %; 1056 update flags
= 4 planes × 8 bands × 3 prev-token classes × 11 positions read once per
decoded frame, each at its position-specific `COEFF_UPDATE_PROBS`
probability — NOT a flat 128).

### The change

The pre-291 loop carried a 4-deep `enumerate()` whose `(i, j, k, t)`
indices existed only to re-index `COEFF_UPDATE_PROBS[i][j][k][t]` — a
four-level bounds-checked re-traversal of the probability table on every
one of the 1056 flags, even though the output array was already being
walked structurally. The walk order is fixed, so the output and the
probability table are now traversed in exact lockstep by zipping the two
leaf `[…; 11]` rows: the inner read indexes neither array, replacing the
per-flag 4-level index arithmetic + four bounds checks with a single
forward step over the already-flat probability row. Arithmetic, flag
order, and per-position probabilities are unchanged.

### New `token_prob_update` micro-bench

The whole-frame decode benches fold this loop inside the rest of the §9
header parse, so the loop-shape change had no isolated A/B target. The
new `token_prob_update` binary drives `bench_parse_token_prob_update`
directly over a header partition pre-encoded by the crate's own §13.4
writer (`write_no_token_prob_updates` / `write_token_prob_updates`),
decoded once at setup with a full payload-equality assertion so the
measured loop is a proven-valid bitstream walk. Two shapes:

* `parse_no_updates` — the all-`None` frame (every flag false): the
  overwhelmingly common per-frame payload, the loop's hot shape, no
  `L(8)` literal reads;
* `parse_sparse_updates` — a scattering of `Some(prob)` replacements
  exercising the `read_literal(8)` branch on top of the flag walk.

### Measured A/B (criterion stored-baseline, Apple M4 / aarch64, stable 1.95, `--measurement-time 8–10`, same session)

`after` = zipped lockstep loop; `before` = the verbatim pre-291
4-level-indexed loop, both timed against the same stored `after`
baseline:

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `token_prob_update/parse_no_updates` | 3.04 µs (348 Melem/s) | **2.73 µs** (385 Melem/s) | **−10.6 %** (p < 0.05) |
| `token_prob_update/parse_sparse_updates` | 3.17 µs (333 Melem/s) | **2.93 µs** (361 Melem/s) | **−15.7 %** (p < 0.05) |

Criterion's own change estimate reports "Performance has regressed" when
the old indexed loop is run against the `after` baseline (i.e. the new
loop is the faster one) at p < 0.05 on both rows; re-running the
optimised loop against its own baseline shows no change on
`parse_no_updates` (the low-noise hot shape) and stays within the
shared-box drift on the sparse row. The `parse_no_updates` row is the
clean signal and the dominant real-world payload (most frames carry no
prob updates). On the whole-frame decode this loop is ≈ 5 % of decoder
self-time, so the −10.6 % loop-level win is a ≈ 0.5 % whole-frame
decoder floor reduction, layered under the round-283 token descent and
round-286/289 reconstruct/rotation wins.

### Bit-identity

The loop reads the same flags in the same order at the same per-position
probabilities, so output is bit-identical by construction. Anchored by
the new `parse_token_prob_update_matches_indexed_reference` (a mixed
`Some`/`None` payload — ~1 in 5 of the 1056 positions carrying a
replacement so the `L(8)` literal branch fires across all four planes —
encoded at the §13.4 per-position probabilities, decoded through both the
production parser and a verbatim copy of the pre-291 4-level-indexed
reference loop, asserting equal payloads *and* equal remaining-input
stream positions), plus the full stable lib suite (491) and nightly +
`simd` lib suite (493), every in-tree decode/roundtrip integration test
(37 green binaries), and the `blackbox_oracle` black-box validator —
decoded bytes unchanged across the entire corpus.

Next profile-opt target: the round-288 deeper follow-on — a copy-on-write
(`Rc`/`Arc`-backed) plane representation for the residual reference-slot
rotation cost (also removing the two unavoidable keyframe clones), which
is a public-API change to `RefFrameSlot`'s `Vec<u8>` plane fields and so
needs its own round. Decoder ranks 1–2 (`decode_block_core` token
descent, `reconstruct_inter_mb`) remain the heaviest clusters but were
each fused in rounds 283 / 286.

## Round 292 — `predict_b4x4` §12.3 sub-block intra coverage + decoder hotspot refresh (bench round)

Round 292 (2026-06-14) is a **bench-coverage** round: no behaviour
change, decoded bytes identical. The §12 intra predictors had isolated
A/B instruments for the 16×16 whole-block modes (`intra_predict_dc16`:
DC/V/H/TM) but **not** for the §12.3 4×4 B_PRED sub-block predictor
(`intra_predict::predict_b4x4`) — the per-sub-block directional intra
kernel that a B_PRED macroblock invokes sixteen times (once per 4×4 luma
sub-block). B_PRED is the dominant luma intra mode on detailed keyframe
content, so this path is folded into the whole-frame keyframe-decode
bench on every run, yet had no isolated target for a future per-mode
optimisation.

### New `intra_predict_b4x4` micro-bench

`benches/intra_predict_b4x4.rs` drives `predict_b4x4` over a non-flat
ramp neighbour context (an 8-pixel `above` so the right-edge diagonal
modes exercise their extension pixels, a 4-pixel `left`, and a distinct
top-left `p`) chosen so no directional kernel collapses to a constant.
Two layers:

* the ten §12.3 modes individually (`b_dc … b_hu`), giving each
  directional kernel its own A/B anchor — the diagonal `Vr`/`Vl`/`Hd`/
  `Hu` modes do ~16 `avg3p`/`avg2p` calls each, `Dc` is the cheap one;
* `bpred_mb_16_subblocks` — sixteen calls cycling through all ten modes
  with a per-sub-block rotated context, the realistic per-macroblock
  B_PRED decode unit (call count and mode spread match a real 16×16
  luma block).

### Measured (criterion `--quick`, Apple M4 / aarch64, stable 1.95)

| Bench | Time |
|---|---:|
| `intra_predict_b4x4/b_dc` | 4.53 ns |
| `intra_predict_b4x4/b_tm` | 4.32 ns |
| `intra_predict_b4x4/b_ve` | 4.50 ns |
| `intra_predict_b4x4/b_he` | 4.50 ns |
| `intra_predict_b4x4/b_ld` | 4.71 ns |
| `intra_predict_b4x4/b_rd` | 4.60 ns |
| `intra_predict_b4x4/b_vr` | 4.50 ns |
| `intra_predict_b4x4/b_vl` | 4.69 ns |
| `intra_predict_b4x4/b_hd` | 4.69 ns |
| `intra_predict_b4x4/b_hu` | 4.73 ns |
| `intra_predict_b4x4/bpred_mb_16_subblocks` | **84.6 ns** |

The ten modes cluster tightly at 4.3–4.7 ns; the per-mode spread is
small because every mode terminates in the same 16-byte `copy_from_slice`
and the directional arithmetic (4–16 `avg3`/`avg2`/`clamp255` ops) is
cheap relative to call + buffer-fill overhead. The full 16-sub-block
macroblock is ≈ 84.6 ns — i.e. a B_PRED MB's luma intra prediction is a
sub-100-ns per-MB cost, well below its §13 token-decode cost (the
`token_decode` ranked #1 below) and its reconstruct cost. This confirms
B_PRED intra prediction is **not** a decoder hotspot worth a kernel
rewrite; the new bench's value is a permanent A/B floor so a future
regression (or a speculative SIMD per-row rewrite of the diagonal modes)
is measurable.

### Refreshed ranked decoder hotspot map (post r283–r291, this session)

Whole-frame + targeted decode benches re-measured this round
(`--quick`, same box):

| Bench | Time | Throughput |
|---|---:|---:|
| `keyframe_decode/decode_keyframe_320x240_qi32` | 141.4 µs | 543 Mpx/s |
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 165.2 µs | 397 Mpx/s |
| `token_decode/decode_mb_coeffs_dense_64mb` | 473.7 µs | — |
| `token_decode/inter_decode_12f_176x144_token_heavy` | 1.206 ms | 252 Mpx/s |
| `reconstruct_inter_mb/subpel_full_residue` | 571 ns | — |
| `reconstruct_inter_mb/whole_pixel_full_residue` | 440 ns | — |
| `reconstruct_inter_mb/subpel_skip` | 274 ns | — |
| `token_decode/decode_mb_coeffs_sparse_64mb` | 45.6 µs | — |
| `intra_predict_b4x4/bpred_mb_16_subblocks` | 84.6 ns | — |

Ranked next-round decoder candidates, carried forward from the round-282
profile and the rounds 283/286/289/291 fuse history:

1. **§13 token decode (`dct_tokens::decode_block` + `decode_mb_coeffs`)**
   — still the heaviest decoder cluster (`token_decode` dense 64-MB =
   473.7 µs; the token-heavy 12-frame inter clip = 1.206 ms, the single
   most expensive bench in the suite). Rounds 283 fused the descent and
   batched bool-decoder renormalisation; the remaining lever is the
   §13.2 per-coefficient context-fetch (band / has-coeff state) and the
   token-tree leaf dispatch. Biggest single decode lever.
2. **Per-frame reference-slot copy churn (`RefFrameSlot::clone` /
   `Vp8DecoderState::decode_frame` plane copies)** — the `memmove`
   family the round-282 profile put at ≈ 11–14 % of inter decode. Round
   289 cut the rotation cost; the residual is the two unavoidable
   keyframe clones + the copy-on-write plane representation, which is a
   public-API change to `RefFrameSlot`'s `Vec<u8>` fields and so needs
   its own (versioned) round.
3. **`reconstruct_inter_mb` sub-pel path** — `subpel_full_residue` =
   571 ns/MB, ≈ 30 % above the whole-pixel path (440 ns); the §18.3
   six-tap + IDCT-add fusion landed in round 286, so the residual is the
   sub-pel fetch-halo + filter, already SIMD under the `simd` feature.
4. **`coded_header::parse_token_prob_update`** — the round-291 zipped
   lockstep loop took the ≈ 5 % per-frame §13.4 flag-read cost down
   −10.6 %; now a small fixed floor, de-prioritised below the three
   above.

B_PRED intra prediction (this round's new bench) is explicitly **not**
on the candidate list: at 84.6 ns/MB it is two-to-three orders of
magnitude below the per-MB token-decode + reconstruct cost. The bench
exists as a regression floor, not a flagged hotspot.

### Bytes-identical

Pure additive change: one new `benches/` binary + its `[[bench]]`
registration + this document. No `src/` or `tests/` edits, so every
decode path is byte-for-byte the pre-292 path by construction. The new
bench synthesises its own inputs in-bench (no fixture files). The full
stable lib suite (491) and nightly + `simd` lib suite (493), the in-tree
decode/roundtrip integration tests, and the `blackbox_oracle` black-box
validator are unchanged and green.

## Round 294 — §14.4 DC-only IDCT-add fast path (profile-opt round)

Round 294 (2026-06-14) re-profiled the decode hot paths and took the
result's clearest lever. A fresh `sample(1)` PID-attach profile (12 s @
1 ms, nightly + `simd`, `--measurement-time 60`):

* **token-heavy inter stream** (`token_decode/inter_decode_12f…`):
  `decode_block_core` rank-1 at 5466 samples (≈ 53 %),
  `inverse_dct_4x4_add_into_simd` rank-2 at 1676 (≈ 16 %). The §13
  token descent is already fully fused (round 283) and resists a
  bit-identical kernel change.
* The §14.4 IDCT-add (rank-2) was the next actionable target: it is
  applied **unconditionally** to every coded residual sub-block — even
  the very common case where only the DC coefficient is non-zero (a
  well-predicted inter sub-block).

### The change

A §14.4 **DC-only fast path** at the top of `inverse_dct_4x4_add_into`.
When every AC coefficient is zero (`input[1..]` all zero), both
separable passes carry only the DC term, so the full transform reduces
to a single uniform residue `(input[0] + 4) >> 3` added (clamped) to all
sixteen predictor pixels — derived directly from the §14.4 listing
(pass 1 leaves `tmp[0]=tmp[4]=tmp[8]=tmp[12]=input[0]`, pass 2's row
butterfly then yields the same rounded value for all outputs). The
butterfly, transpose, and SIMD lane work are skipped entirely; an
all-zero DC (`(input[0]+4)>>3 == 0`) returns immediately with the
prediction untouched. The guard sits in the shared dispatcher so both
the scalar and `simd` paths benefit, and the general path is unchanged
for any block with a non-zero AC coefficient.

### Measured A/B (`--measurement-time 10 --warm-up-time 2`, four interleaved pre/post pairs per config, Apple M4 / aarch64, nightly + `simd`)

| Bench / config | Pre (4-run mean) | Post (4-run mean) | Δ (means) |
|---|---:|---:|---:|
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 123.73 µs | **106.45 µs** | **−14.0 %** |
| `token_decode/inter_decode_12f_176x144_token_heavy` | 944.7 µs | 940.7 µs | −0.4 % |
| `reconstruct_inter_mb/subpel_full_residue` (all-coded) | 411.96 ns | 427.86 ns | +3.9 % |
| `reconstruct_inter_mb/whole_pixel_full_residue` (all-coded) | 274.1 ns | 288.5 ns | +5.2 % |
| `reconstruct_inter_mb/subpel_skip` | 253.9 ns | 254.3 ns | +0.2 % |

The headline is the whole-frame inter decode: **−14.0 %**, with every
one of the four post runs (106.02 / 106.71 / 106.54 / 106.51 µs) below
every pre run (123.04 / 123.84 / 123.77 / 124.25 µs) — far outside the
measurement envelope. Throughput **530 → 616 Mpx/s**. The token-heavy
12-frame stream (textured drifting content, mostly fully-coded blocks
where the DC-only path rarely fires) is noise-bounded at −0.4 %
(per-run −1.6 % / −0.6 % / +1.8 % / −2.1 %). The `reconstruct_inter_mb`
synthetic benches feed **all-coded** blocks (every AC non-zero — the
fast path never fires) so they show only the one-comparison guard cost
(+4…5 %); that shape does not occur in real decode where the
predicted-residue DC-only fraction dominates, which is exactly why the
whole-frame number swings the other way. `subpel_skip` short-circuits
before the IDCT and is flat.

### Bytes-identical

The fast path produces bit-identical output to the full transform. New
equivalence test `dc_only_add_into_matches_general_path` sweeps every
DC magnitude across the §14.2 envelope (both clamp saturations) at
every strided sub-block position, asserting the fast path equals an
independent full-`inverse_dct_4x4` → add-clamp reference. The full
stable lib suite (492 → 493 with the new anchor) and nightly + `simd`
lib suite (494 → 495), plus the in-tree decode/roundtrip integration
tests, are green on both paths. A 10-fixture decode-side byte-hash A/B
through `Vp8DecoderState` (162 048 Y/U/V plane bytes, FNV-1a
`cf33bace1d44adff`) is identical pre-/post-change on **stable scalar
and nightly + `simd`**.

### Refreshed ranked decoder hotspot map (post r294)

| Rank | Symbol | inter-decode share | Notes |
|---|---|---:|---|
| 1 | `dct_tokens::decode_block_core` | ≈ 26 % (≈ 53 % on dense streams) | round-283 fused descent; resists a bit-identical change |
| 2 | `inverse_transform::inverse_dct_4x4_add_into` | ≈ 10 % | **this round's target** — DC-only path now folds out the SIMD butterfly for predicted-residue blocks |
| 3 | `coded_header::parse_token_prob_update` | ≈ 7 % | round-291 zipped loop; small fixed per-frame floor |
| 4 | per-frame reference-slot plane copy | ≈ 11–14 % (memmove family) | copy-on-write plane representation — a versioned `RefFrameSlot` API change, own round |

Named next PROFILE-OPT target: the per-frame reference-slot plane-copy
churn (`memmove` family, ≈ 11–14 % of inter decode) — a copy-on-write
plane representation gated behind a versioned `RefFrameSlot` change, so
it carries its own round.

## Round 295 — §7.3 boolean entropy decoder primitive in isolation (bench round)

Round 295 (2026-06-14) added `bool_decoder_read`, the first bench to
isolate the §7.3 `bool_decoder` primitive itself. Every coded bit in a
VP8 frame — header flags, macroblock modes, motion-vector components,
and the dominant share, DCT-coefficient tokens — flows through
`BoolDecoder::read_bool` and its inner batched renormalisation step.
That primitive was previously measured only folded inside the §13 token
descent (`token_decode`), §12/§16 prediction, §14 transforms, and the
§15 loop filter, so the renormalisation batching documented in
`bool_decoder.rs` (the bit-at-a-time §7.3 loop collapsed into a single
`range.leading_zeros()`-derived shift with at most one byte-pull per
read) had no isolated A/B target. A future change to the renorm shape
would only show up diluted inside the whole-frame numbers.

### The bench

Three regimes that bracket the §7.3 cost, each driven by a partition
produced by the crate's own §7.3 `BoolEncoder` (the exact inverse of
the decoder under test) and decoded once at setup with a full
bit/byte-equality assertion, so the measured loop is always a
proven-valid bitstream walk:

* `read_bool_skewed_64k` — 65 536 booleans at probability 248
  (≈ 97 % skew). The interval split rarely needs a doubling, so the
  renormalisation fast-path (`shift == 0` early return) dominates — the
  well-modelled regime (a confident coefficient context / skip flag).
* `read_bool_balanced_64k` — 65 536 booleans at probability 128 (a fair
  coin). The split lands mid-interval, so nearly every read triggers a
  one/two-bit renormalisation and frequent byte-refills — the §7.3
  renorm worst case.
* `read_literal_8b_8k` — 8 192 `read_literal(8)` calls (65 536
  flat-probability-128 reads assembled MSB-first), the §9 header / §17
  MV-magnitude `L(n)` idiom, measuring the `read_literal` accumulator
  loop on top of `read_bool`.

### Measured (`--quick`, Apple M4-class aarch64, macOS 25.1, rustc 1.95.0 stable)

| Bench | Time | Throughput (bits/s) |
|---|---:|---:|
| `bool_decoder_read/read_bool_balanced_64k` | **182.28 µs** | 359.5 Melem/s |
| `bool_decoder_read/read_literal_8b_8k` | 152.79 µs | 428.9 Melem/s |
| `bool_decoder_read/read_bool_skewed_64k` | 151.81 µs | 431.7 Melem/s |

### Ranked §7.3 hotspot map

| Rank | Regime | Per-bool cost | What dominates |
|---|---|---:|---|
| 1 | balanced (prob 128, fair coin) | ≈ 2.78 ns | renormalisation shift + byte-refill on nearly every read — the §7.3 worst case |
| 2 | literal `L(n)` (flat 128, skewed bits) | ≈ 2.33 ns | the `read_literal` accumulator loop; per-bool cost tracks the skewed `read_bool` (its bits decode through the renorm fast path) |
| 3 | skewed (prob 248) | ≈ 2.32 ns | the modelled fast path — `shift == 0` early return on most reads, no byte pull |

The headline finding: the **balanced (fair-coin) regime is ≈ 20 %
slower per bool** than the skewed/literal regimes (2.78 ns vs ≈ 2.32
ns). The renormalisation shift and its byte-refill — the part the
in-tree batching already collapses to a single `leading_zeros()` shift
— is the dominant per-bit cost, and it scales directly with how often
the interval split forces a doubling. `read_literal` adds no measurable
overhead beyond its constituent `read_bool` calls: its 2.33 ns/bit is
within noise of the skewed `read_bool`'s 2.32 ns/bit, confirming the
accumulator loop (`v = (v << 1) | bit`) is free relative to the entropy
read it wraps.

### No behavioural change

This is a measurement-only round: bench harness plus the partitions it
synthesises, with no edit to any decode path. The renormalisation is
already collapsed to a single leading-zeros shift, so the profile
surfaces no obvious byte-identical micro-opt; the value of the bench is
the isolated A/B target it now provides for any future change to the
§7.3 read or renorm shape (e.g. a multi-bit batched read, or a
different byte-refill cadence). The full lib test suite is unchanged
and green.

### Next-round candidates surfaced

The balanced-regime renorm cost is the actionable §7.3 lever a future
round could target — but only behind a bit-exact A/B, since the
primitive is on the critical path of every decode. The named
whole-frame PROFILE-OPT target remains the per-frame reference-slot
plane-copy churn (r294's rank-4, ≈ 11–14 % of inter decode, its own
round).

## Round 298 — §14.1 dequantization layer in isolation (bench round)

Round 298 (2026-06-14) added `dequantize_mb`, isolating the §14.1
dequantization layer — the step that turns the raw quantized
coefficients `decode_mb_coeffs` recovers from the token partition into
the pre-dequantized `MbCoeffs` the §14.2 inverse transforms consume.
Two layers run on the decode path and neither had an isolated harness;
both were only ever measured folded inside the whole-frame decode
benches (`keyframe_decode`, `inter_decode_short_clip`, `token_decode`):

* **factor derivation** — `MbDequantFactors::from_quant_indices`
  (per-frame, no segmentation) and `MbDequantFactors::for_segment` (the
  §10 per-segment override). Both run the §20.4 `dequant_init` body:
  six `dc_qlookup` / `ac_qlookup` table reads through `clamp_qindex`,
  the `*2` Y2-DC and `*155/100` Y2-AC scalings, and the `>132`
  chroma-DC / `<8` Y2-AC clamps. `from_quant_indices` fires once per
  frame; `for_segment` once per active segment.
* **the apply** — `MbDequantFactors::dequantize`, the per-MB hot loop.
  It scales all twenty-five 4×4 blocks of one macroblock (the Y2 block,
  the sixteen Y sub-blocks, the four U and four V chroma sub-blocks):
  400 coefficient×factor multiplies per non-skip MB, computed in `i32`
  and stored back as `i16` (§14.1 page 76). This is the SIMD/unroll A/B
  target — `dequant_block` already carries an optional `core::simd`
  path — so a stable per-MB baseline at criterion resolution was
  overdue.

### The bench

Five functions, inputs synthesised in-bench (no committed fixtures):

* `from_quant_indices` — the per-frame derivation on a mid-range
  quantiser header (`y_ac_qi = 64`) with **every** per-plane delta
  present, so all six factors take the full derivation path (none
  short-circuit a `None` delta to 0).
* `for_segment_delta` / `for_segment_absolute` — the §10 per-segment
  override in both modes (delta: `base = yac_qi + segment_quant`;
  absolute: `base = segment_quant`).
* `dequantize_sparse_mb` — the per-MB apply over a macroblock whose
  twenty-five blocks each carry a DC plus three low-frequency AC
  coefficients (the common post-quantisation residual shape).
* `dequantize_dense_mb` — the same apply over fully-dense blocks
  (every lane non-zero, the per-coefficient worst case).

### Measured (`--warm-up-time 1 --measurement-time 3`, Apple M4-class aarch64, macOS 25.1, rustc 1.95.0 stable)

| Bench | Time |
|---|---:|
| `dequantize_mb/from_quant_indices` | **2.32 ns** |
| `dequantize_mb/for_segment_absolute` | 2.86 ns |
| `dequantize_mb/for_segment_delta` | 2.95 ns |
| `dequantize_mb/dequantize_sparse_mb` | 183.0 ns |
| `dequantize_mb/dequantize_dense_mb` | 184.1 ns |

### Findings

The two layers sit three orders of magnitude apart in absolute cost:
factor derivation is a once-per-frame (or once-per-segment) ≈ 2–3 ns
table-lookup-and-scale, while the per-MB apply is ≈ 183 ns and fires on
every coded macroblock — so the apply is where any §14.1 throughput win
has to come from. `for_segment` costs ≈ 0.5 ns more than
`from_quant_indices` (the extra base-index add and `absolute` branch);
delta and absolute mode are within noise of each other.

The headline for the apply: **sparse and dense are within noise** (183.0
vs 184.1 ns, < 1 %). The scalar `dequant_block` walks all sixteen lanes
of every block unconditionally — `block[0] * dc_factor` then a
`skip(1)` loop over lanes 1..=15 — so a block of mostly-zero
coefficients costs the same as a fully-dense one (`0 * factor` is still
a multiply-and-store). The per-MB cost is therefore set by the fixed
400-multiply block count, not by coefficient occupancy. That is exactly
the shape that maps cleanly onto the 16-wide `core::simd` path
`dequant_block` already carries (one widen + one lane-wise multiply +
one truncate per block, no data-dependent branching), which is why the
SIMD path exists and stays byte-exact against the scalar listing on
every fixture (`dequant_block_simd_matches_scalar_on_stress_inputs`).

### No behavioural change

This is a measurement-only round: bench harness plus the inputs it
synthesises, with no edit to any decode or encode path. The full lib
test suite is unchanged and green. The bench is the isolated A/B target
for any future change to the §14.1 apply (a wider SIMD lane group, a
zero-run skip, or a fused dequant→IDCT pass).

### Next-round candidates surfaced

The per-MB apply's occupancy-independent cost is the actionable §14.1
lever: a fused dequant→inverse-transform pass (the apply currently
materialises a fully-scaled `MbCoeffs` that the IDCT then re-reads)
could amortise the block walk, but only behind a bit-exact A/B since
the layer is on every decoded macroblock. The named whole-frame
PROFILE-OPT target remains the per-frame reference-slot plane-copy churn
(r294's rank-4, ≈ 11–14 % of inter decode, its own round).

## Round 300 — drop the dead per-frame coefficient/side-band default-fill (profile-opt round)

Round 300 (2026-06-14) re-profiled the decode hot paths and took the
clearest non-API-gated lever the §13 → §15 frame-decode flow surfaced.

### Profile evidence

A fresh `sample(1)` PID-attach profile of `inter_decode_short_clip`
(12 s @ 1 ms, nightly + `simd`), top-of-stack self-time samples:

| Symbol | Samples | Note |
|---|---:|---|
| `dct_tokens::decode_block_core` | 2899 | round-283 fused descent; resists a bit-identical change |
| `inverse_transform::inverse_dct_4x4_add_into` | 1095 | r294 DC-only fast path already landed |
| `_platform_memmove` | 988 | the named ref-slot plane-copy churn (API-gated, own round) |
| `coded_header::parse_token_prob_update` | 748 | r291 zipped loop, small fixed floor |
| **`__bzero` + `_platform_memset`** | **274 + 231 = 505** | **this round's target — buffer default-fills** |

The `__bzero` / `memset` cluster (≈ 5 % of decode self-time) traced to a
**doubly-wasted** per-frame default-fill. Both the §13 residual decode
(`decoder::decode_residuals`, `state::decode_intra_residuals`) and the
§16 interframe driver (`Vp8DecoderState::decode_frame`) built their
per-MB output vectors with `vec![default(); mb_rows*mb_cols]` (or a
`with_capacity` + `resize(default())`) and then **overwrote every slot**
via an indexed write inside the raster-order decode loop. The bulk
default-fill — for the `Vec<MbCoeffs>` lane that is 800 bytes × every MB,
plus a `sentinel_mode` clone per `modes_out` slot on the inter path —
is pure dead work: not one default-initialised slot is ever read (the
§15 loop filter consumes the vectors only after the whole frame is
decoded and every slot has been written).

### The change

Replace `vec![default(); N]` / `with_capacity` + `resize` with
`Vec::with_capacity(N)` followed by `push` inside the decode loop, in all
three sites:

* `decoder::decode_residuals` — keyframe `Vec<MbCoeffs>` (stateless path);
* `state::decode_intra_residuals` — keyframe `Vec<MbCoeffs>` (stateful
  `Vp8DecoderState` path);
* `Vp8DecoderState::decode_frame` — the four interframe side-band vectors
  (`modes_out`, `coeffs_out`, `ref_frames_out`, `inter_modes_out`), and
  the now-unused `sentinel_mode` constant is removed.

Every MB is decoded exactly once in raster order (`mb_row` outer,
`mb_col` inner, `raster = mb_row*mb_cols + mb_col`), so the `push`
sequence reproduces the **identical** vector contents the indexed write
produced, with the capacity reserved up front (no reallocation). A
`debug_assert_eq!(vec.len(), raster, …)` at each push site pins the
ordering invariant. On any decode error the `?` propagates and the
partially-filled vectors are simply dropped, exactly as before.

### Measured A/B (`--warm-up-time 2 --measurement-time 8`, three interleaved pre/post runs each, Apple M4-class aarch64, nightly + `simd`)

| Bench | Pre (run medians) | Post (run medians) | Δ |
|---|---:|---:|---:|
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 109.2 / 113.2 / 108.9 µs | 105.9 / 106.3 / 106.3 µs | **≈ −3 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 118.3 / 109.5 / 108.7 µs | 106.8 / 107.1 / 106.7 µs | **≈ −2 %** |

Every post run sits below every pre run on both benches — the
improvement is outside the measurement envelope. The win is the removed
bulk frame-init memset (the `__bzero` / `memset` cluster the profile
flagged); it grows with frame size (the fill is `O(mb_count)` and was
`800 bytes`-per-MB on the coeff lane alone).

### Bytes-identical

Pure allocation-shape refactor: the decoded vector contents are
bit-for-bit what the indexed-write path produced (same values, same
raster order). The full stable lib suite (493) and nightly + `simd` lib
suite (495), plus all 37 in-tree integration test binaries (encode→
decode pixel lockstep, keyframe/P-frame roundtrips, inter-stream,
two-pass roundtrip, the `blackbox_oracle` black-box validator), are green
on both toolchains. `cargo clippy --all-targets -D warnings` is clean on
stable and nightly + `simd`.

### Refreshed ranked decoder hotspot map (post r300)

| Rank | Symbol | Note |
|---|---|---|
| 1 | `dct_tokens::decode_block_core` | round-283 fused descent; resists a bit-identical change |
| 2 | `inverse_transform::inverse_dct_4x4_add_into` | r294 DC-only fast path landed |
| 3 | per-frame reference-slot plane copy (`_platform_memmove`) | the named PROFILE-OPT target — copy-on-write plane representation behind a versioned `RefFrameSlot` change, own round |
| 4 | `coded_header::parse_token_prob_update` | r291 zipped loop, small fixed floor |

The named whole-frame PROFILE-OPT target remains the per-frame
reference-slot plane-copy churn (the `_platform_memmove` rank, a
copy-on-write plane representation gated behind a versioned
`RefFrameSlot` API change, so it carries its own round).

## Round 302 — collapse the loop-filter coefficient side-band to a per-MB flag (profile-opt round)

Round 302 (2026-06-14) took a non-API-gated lever the §13 → §15 inter
frame-decode flow surfaced: the stateful interframe decoder kept a
whole-frame `Vec<MbCoeffs>` (`coeffs_out`) alive past the per-MB decode
loop solely to feed the §15 loop filter — yet the loop filter reads those
coefficients only through `mb_has_coeffs`, a **single boolean per MB**.
Every MB's coefficients are otherwise fully consumed by reconstruction
inside the per-MB loop (`reconstruct_inter_mb` / `reconstruct_split_mv_mb`
/ the intra reconstructors), so the frame-length `Vec<MbCoeffs>` was ≈
800 bytes / MB of pure carry traffic for one boolean reduction per MB.

### The change

The inter decode loop now computes `mb_has_coeffs(&mb_coeffs)` inline —
while the freshly-decoded bundle is still hot in cache, where the `any()`
short-circuits on the first non-zero — and pushes only the `bool` into a
`Vec<bool> has_coeffs_out`. The §15 pass dispatches to a new internal
`filter_inter_frame_flags` (and its keyframe analogue `filter_frame_flags`)
that takes a `&[bool]` "has-coeffs" slice instead of `&[MbCoeffs]`. The
public `filter_frame` / `filter_inter_frame` keep their `&[MbCoeffs]`
signatures as thin wrappers (`coeffs.iter().map(mb_has_coeffs).collect()`
→ flag core), so no public API changes. The keyframe path is left on the
public wrapper because it already materialises the full `Vec<MbCoeffs>`
for `decode_keyframe` reconstruction — no buffer to remove there.

### Measured A/B (`--warm-up-time 2 --measurement-time 8`, three interleaved pre/post runs each, Apple M4-class aarch64)

| Bench | Pre (run medians) | Post (run medians) | Δ |
|---|---:|---:|---:|
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 116.2 / 116.5 / 118.6 µs | 115.5 / 115.1 / 113.5 µs | **≈ −1…−3 %** |
| `keyframe_decode/decode_keyframe_320x240_qi32` | 140.0 / 142.0 / 140.8 µs | 140.9 / 139.3 / 139.9 µs | ≈ flat |

Every inter post-run median sits below the baseline band; the keyframe
path is flat by design (it retains the full coeff vector for
reconstruction). The win is modest on the small 4-frame 128×128 inter
clip because the eliminated buffer is `O(mb_count)` — it grows with frame
size (the removed `Vec<MbCoeffs>` is 800 bytes × every MB plus the
associated write-then-read memory traffic, replaced by 1 byte × MB
written at decode time).

### Bytes-identical

The flag path is byte-for-byte the coeffs path: `filter_*_flags` differs
only in reading `has_coeffs[i]` where the public path computes
`mb_has_coeffs(&coeffs[i])`. Two new exhaustive equivalence tests
(`filter_frame_flags_matches_coeffs`, `filter_inter_frame_flags_matches_coeffs`)
sweep all 2⁶ per-MB occupancy masks × every Y-mode × simple/normal config
on a textured multi-MB plane set and assert the two paths produce
identical planes. The full stable lib suite (493 → 495 with the two
anchors) and nightly + `simd` lib suite (495 → 497), plus all in-tree
integration test binaries (encode→decode pixel lockstep, keyframe /
P-frame roundtrips, inter-stream, two-pass roundtrip, the `blackbox_oracle`
black-box validator), are green on both toolchains. `cargo clippy
--all-targets -D warnings` is clean on stable.

### Refreshed ranked decoder hotspot map (post r302)

| Rank | Symbol | Note |
|---|---|---|
| 1 | `dct_tokens::decode_block_core` | round-283 fused descent; resists a bit-identical change |
| 2 | `inverse_transform::inverse_dct_4x4_add_into` | r294 DC-only fast path landed |
| 3 | per-frame reference-slot plane copy (`_platform_memmove`) | the named PROFILE-OPT target — copy-on-write plane representation behind a versioned `RefFrameSlot` change, own round |
| 4 | `coded_header::parse_token_prob_update` | r291 zipped loop, small fixed floor |

The named whole-frame PROFILE-OPT target remains the per-frame
reference-slot plane-copy churn (the `_platform_memmove` rank, a
copy-on-write plane representation gated behind a versioned `RefFrameSlot`
API change, so it carries its own round).

## Round 304 — contiguous-plane crop fast path + memmove-caller profile (profile-opt round)

Round 304 (2026-06-14) re-profiled `inter_decode_short_clip` with a fresh
`sample(1)` PID-attach pass (12 s @ 1 ms, nightly + `simd`) and broke down
the standing #3 `_platform_memmove` rank by caller. The top-of-stack
self-time table reconfirmed the r302 ordering (`decode_block_core` 1431,
`inverse_dct_4x4_add_into` 674, `_platform_memmove` 605,
`parse_token_prob_update` 342). Attributing every memmove sample to its
nearest `oxideav_vp8` caller:

| memmove caller | Samples | Note |
|---|---:|---|
| `state::decode_frame` | 308 | slot-rotation + per-MB side-band pushes (r289/r300/r302 already move-minimised) |
| `dct_tokens::decode_mb_coeffs` | 112 | the per-MB `MbCoeffs` return-by-value + `default()` zero-init — **not** dead work: `decode_block_core` leaves coefficients past the §13.3 EOB untouched and relies on the caller pre-zeroing, so the fill is load-bearing |
| `decoder::crop_to_visible` | 52 | **this round's target** — the per-row visible-crop copy |
| `motion_comp::reconstruct_inter_mb` | 46 | residue fold (r286 fused) |
| `state::RefFrameSlot::from_keyframe_planes` | 32 | unavoidable key-frame triple-slot populate (2 clones, r289 move-min) |

### The change

`decoder::crop_to_visible` packed the visible region with `h` (resp. `uvh`)
separate row-sliced `extend_from_slice` calls regardless of whether a
per-row stride gap existed. When the visible width already equals the
macroblock-padded stride (`w == y_stride` / `uvw == uv_stride` — the common
case for 16-multiple dimensions, e.g. the 128×128 and 320×240 benches), the
whole leading `w * h` region is already the packed output verbatim, so the
crop is now one contiguous `to_vec()` instead of a per-row loop. The strided
path is retained byte-for-byte for the genuine-crop case (visible < stride).

### Measured A/B (interleaved 3× pre/post, Apple M4-class aarch64, nightly + `simd`)

| Bench | Pre (run medians) | Post (run medians) | Δ |
|---|---:|---:|---:|
| `keyframe_decode/decode_keyframe_320x240_qi32` | 141.3 / 139.7 / 143.6 µs | 140.7 / 141.3 / 139.9 µs | within noise |
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 117.2 µs | 115.4 µs | within noise |

The change is **safe and bit-identical** (a new lib test
`crop_to_visible_contiguous_matches_strided` sweeps aligned / cropped /
height-truncated cases against an always-strided reference) but its
whole-frame benefit sits below the measurement floor at benched sizes:
`crop_to_visible` is ≈ 0.7 % of decode self-time (31 / ≈ 4500 samples on the
12 s profile), so collapsing one per-row memcpy loop into a single copy can
not move the whole-frame number out of noise here. The win scales with frame
area (one contiguous copy vs `h` row copies + bounds checks) and the path is
never slower than the prior loop, so it is kept as a no-regression
micro-improvement rather than reverted. No risky change was forced.

### Bytes-identical

Pure copy-shape refactor of the output emit; decoded planes are byte-for-byte
what the strided loop produced. Stable lib suite 496 (+1), nightly + `simd`
lib suite 498 (+1), all in-tree integration test binaries (encode→decode
pixel lockstep, keyframe / P-frame roundtrips, inter-stream, two-pass
roundtrip, the `blackbox_oracle` black-box validator) green on both toolchains.
`cargo clippy --all-targets --no-deps -- -D warnings` clean on stable and
nightly + `simd`.

### Refreshed ranked decoder hotspot map (post r304)

| Rank | Symbol | Note |
|---|---|---|
| 1 | `dct_tokens::decode_block_core` | round-283 fused descent; resists a bit-identical change |
| 2 | `inverse_transform::inverse_dct_4x4_add_into` | r294 DC-only fast path landed |
| 3 | per-frame reference-slot plane copy (`_platform_memmove`) | the named PROFILE-OPT target — copy-on-write plane representation behind a versioned `RefFrameSlot` change, own round |
| 4 | `coded_header::parse_token_prob_update` | r291 zipped loop, small fixed floor |

The remaining memmove headroom is concentrated in `state::decode_frame`'s
slot rotation + the `decode_mb_coeffs` return-by-value, both already
move-minimised / load-bearing; the named whole-frame PROFILE-OPT target is
still the copy-on-write reference-slot representation gated behind a
versioned `RefFrameSlot` API change (its own round).

## Round 306 — `read_literal` register-local fixed-prob-128 loop + literal A/B coverage (profile-opt round)

Round 306 (2026-06-15) revisited the §7.3 boolean-decoder primitive. The
named r304 PROFILE-OPT target (copy-on-write reference-slot representation
behind a versioned `RefFrameSlot` API) was assessed as too large / too
risky for a single safe round — the §9 slot rotation is already
move-minimised (r289 `rotate_reference_slots` does zero plane copies on the
common refresh-last path), so a CoW rewrite would touch the whole
`RefFrameSlot` ownership model for headroom the r304 profile put at a few
hundred memmove samples. Per the round's "pick a smaller safe win OR extend
a benchmark — do not force a risky change" guidance, this round took the
smaller bool-decoder win plus bench coverage instead.

### The change

`BoolDecoder::read_literal` previously called the generic
`read_bool(128)` `num_bits` times. Each call reloaded the `range` /
`value` / `bit_count` / `input` fields from `self`, recomputed the §7.2
interval split with a 32-bit multiply (`(range - 1) * 128`), and re-entered
`renormalize`. At the fixed probability 128 the split collapses to a pure
shift — `1 + (((range - 1) * 128) >> 8)` is exactly `1 + ((range - 1) >> 1)`
(`* 128 >> 8 == >> 1`, an algebraic identity) — and hoisting the four
registers into locals lets the whole accumulator loop run without touching
`self` until it commits the final state. The mid-literal `EndOfStream`
error path commits the consumed registers exactly where the generic loop
would leave them.

### Bit-exactness

Two new lib tests pin the fast path state-for-state against the generic
`num_bits × read_bool(128)` reference loop:
`read_literal_fast_matches_generic_loop` (2 048 mixed-width literals, full
`range`/`value`/`bit_count`/`input` agreement after every literal across
widths 1..=16) and `read_literal_fast_matches_generic_on_end_of_stream`
(identical failure + committed state at the EOS boundary). Stable + nightly
lib suite 498 tests, all in-tree integration binaries (including the
`blackbox_oracle` validator) green.

### Measured A/B (interleaved 3× pre/post, Apple M4-class aarch64, stable)

| Bench | Pre (run min) | Post (run min) | Δ |
|---|---:|---:|---:|
| `bool_decoder_read/read_literal_8b_8k` | ≈ 177 µs | ≈ 176 µs | within noise |

The change removes one multiply and three field reloads per coded bit, but
on a loaded machine the per-run variance (±10–20 %) swamped the difference:
successive identical-code runs swung 177–217 µs. The improvement is a strict
instruction-count reduction with zero behavioural change, kept as a
no-regression micro-improvement (never slower than the generic loop) rather
than a claimed speedup. `read_literal` is a small share of whole-frame
decode time (header / partition-size / MV-magnitude reads, not the DCT-token
hot loop, which uses context-probability `read_bool` not flat-128 literals),
so no whole-frame bench moves either.

### Bench coverage extended

`bool_decoder_read` gained two cases so the `read_literal` /
`read_signed_literal` register-local loops have width-varied A/B targets the
flat width-8 `read_literal_8b_8k` lacked:

* `read_literal_mixed_width` — 8 192 `read_literal` calls of widths spread
  across 1..=16 (the real §9 header / §13 partition-size width mix).
* `read_signed_literal_7b_8k` — 8 192 `read_signed_literal(7)` calls (sign
  bit + six magnitude bits), the §17 MV-component idiom the unsigned bench
  never touched.

Both partitions are produced by the crate's own §7.3 `BoolEncoder` and
decoded once at setup with a full value-equality assertion, so each measured
loop is a proven-valid bitstream walk.

`cargo fmt --check` and `cargo clippy --all-targets --no-deps -- -D
warnings` clean. No risky change was forced; the CoW reference-slot target
remains deferred to its own round.

## Round 308 — shipped vs clone-everything A/B for the §9 slot rotation (profile-opt round)

Round 308 (2026-06-15) closed a measurement gap the prior `ref_slot_rotation`
harness left. The round-288 bench measured only a **clone-everything**
stand-in (`rotate_*` rows): a local `rotate()` helper that clones all three
entry slots plus the current reconstruction, reporting ~12–13 µs/frame as the
"slot copy cost" and naming copy-on-write as the next target sized against
that number. That stand-in over-states the cost: the shipped
`rotate_reference_slots` (r289) is a move-based **minimal-copy** rotation —
it moves each owned source into the last destination that selects it and
clones only a genuinely multi-fed source, so on the common refresh-last path
it performs **zero** plane clones.

This round exposes the shipped rotation to the bench via a `#[doc(hidden)]`
pass-through shim `bench_rotate_reference_slots` (no logic change to the
private rotation) and adds three `shipped_*` rows under the same three flag
combinations as the clone-everything rows.

### A/B (320×240 = 20×15 MB, criterion median, macOS aarch64)

| flags               | `rotate_*` (clone-all) | `shipped_*` (4 input clones) | Δ     |
| ------------------- | ---------------------- | ---------------------------- | ----- |
| `refresh_last_only` | 13.19 µs               | 8.71 µs                      | −34 % |
| `refresh_all`       | 16.21 µs               | 14.31 µs                     | −12 % |
| `copy_gf_arf`       | 17.20 µs               | 8.40 µs                      | −51 % |

### Interpreting the floor

The `shipped_*` rows are **not** zero because the bench clones the current
reconstruction plus the three entry slots into owned arguments each iteration
(the shim consumes by value). `decode_frame` does not pay those four clones —
it passes `self.{last,golden,altref}.take()`, already-owned values it moves —
so the decoder's true per-frame rotation cost is *below* the `shipped_*`
floor. The residual genuine copy work the shipped path still performs:

* `refresh_all` clones the current reconstruction twice (one source → three
  destinations), hence its smaller Δ.
* `copy_gf_arf` clones one pre-rotation slot once (LAST feeds both new-LAST
  and new-GOLDEN), the GOLDEN source moving into ALTREF — far below the six
  clones the stand-in pays.
* The key-frame init (not in this micro) clones twice (one reconstruction →
  three slots).

Only a copy-on-write (`Rc`/`Arc`-backed) slot representation can remove the
`refresh_all` / `copy_gf_arf` / key-frame clones; the move-based rotation
already removes the common refresh-last-path clones. The standing
copy-on-write profile-opt target should be sized against the `shipped_*`
floor (and below, given the input-clone caveat), **not** the `rotate_*`
ceiling — the headroom is materially smaller than the round-288 numbers
suggested.

`cargo fmt --check` and `cargo clippy --all-targets --no-deps -- -D warnings`
clean; 498 lib tests pass. No decoder/encoder logic change; no risky change
forced.

## Round 314 — §15 loop-filter SIMD (`core::simd::Simd<i32, 4>`, simd feature)

Round 314 (2026-06-15) closes the standing round-170 / round-269 candidate
"SIMD the deblock path". Before this round the `simd` feature accelerated the
§14 transforms, §14.1 dequant, §12 intra and §18.3 sub-pixel interpolation —
but the §15 loop filter, the single heaviest decode-side stage on a fully
coded frame, stayed scalar on every build. This round adds 4-lane vector
kernels for the §15.2 simple filter and the §15.3 normal subblock / MB
filters and routes the frame-geometry edge loops through them in groups of
4 rows (vertical edge) / columns (horizontal edge).

### Why 4-lane works here

The §15.1 geometry fires the per-segment kernel once per row of a vertical
edge and once per column of a horizontal edge. Those rows / columns are fully
independent — each gathers its own 8-pixel window from a distinct stride row
(vertical) or distinct column (horizontal), and the §15.2 / §15.3 arithmetic
on one window never reads another. That independence maps directly onto a
`Simd<i32, 4>` where lane `r` holds segment `r`'s value at a given tap. Both
§15.1 edge lengths (16 luma, 8 chroma) are multiples of 4, so the whole edge
vectorises with no scalar tail.

### Byte-exactness discipline

The scalar kernels early-out per segment when the §15.3 `filter_yes` gate (or
the §15.2 edge metric) fails. The vector path computes the filtered result
for all four lanes unconditionally and then, per lane, selects between the
filtered and the original pixel with a `Mask` derived from the same gate.
Every lane therefore performs the identical i32 add / `clamp_s8` / shift
sequence the scalar code performs, so the selected output is bit-for-bit the
scalar output. Three parity tests
(`subblock_filter_simd_matches_scalar`, `mb_filter_simd_matches_scalar`,
`simple_segment_simd_matches_scalar`) drive a 70-window stress matrix (ramps,
flats, spikes, an LCG pseudo-random fill) across five `(hev, interior,
edge)`-limit combinations through both the vector kernel and the
always-compiled scalar per-segment reference, asserting equality on every
lane.

### A/B (`loop_filter_frame`, 320×240 = 20×15 MB, criterion median, M4-class aarch64)

| Bench | Scalar (stable) | SIMD (nightly) | Δ |
|---|---:|---:|---:|
| `filter_frame_keyframe_320x240_normal` | 352.7 µs | **196.5 µs** | **−44 %** |
| `filter_frame_keyframe_320x240_simple` | 81.0 µs | **49.3 µs** | **−39 %** |
| `filter_inter_frame_320x240_normal` | 342.6 µs | **205.6 µs** | **−40 %** |

The micro-benches in `loop_filter_normal.rs` / `loop_filter_mb_edge.rs` call
the per-segment scalar kernels directly and so are unchanged; the SIMD path
is only reachable through the `filter_frame` / `filter_inter_frame` geometry
loops, which `loop_filter_frame.rs` exercises — the correct A/B target.

`cargo fmt --check` clean on stable; `clippy --all-targets --no-deps -D
warnings` clean on both the stable scalar build and the nightly `simd` build;
691 tests pass under `--features simd` (686 under the stable scalar build —
the 5 extra are the simd-gated parity tests). No decoder/encoder logic change
on the stable path; the stable build is byte-identical to before this round.

## Round 409 — ARNR + §7.3 bool-encoder write harnesses (bench round, part 1)

Round 409 (2026-07-11) opens the bench/profile axis by covering the two
public hot layers that still had no isolated criterion harness:

* **`arnr_build_altref`** — the motion-compensated temporal filter that
  builds the §9.7 altref anchor (`build_arnr_altref`). Per 16×16 block
  of the center frame it runs a whole-pel three-round refinement search
  (±15 px, step 8 → 4 → 2 → 1) against every other window frame, drops
  blocks above the occlusion SAD cutoff, and blends surviving pixels
  with a difference-driven weight. Three workloads: a five-frame
  static-noise window (the steady-state denoise shape), a three-frame
  translating window (the motion-search-heavy shape), and the
  strength-0 pass-through floor (plane copies only — the filter's
  marginal cost is the delta against this row).
* **`bool_encoder_write`** — the §7.3 boolean *encoder* twin of the
  round-295 `bool_decoder_read` bench. Every coded bit of an encoded
  frame passes through `BoolEncoder::write_bool` and its
  carry-propagating renormalisation loop, but the write side was only
  ever measured folded inside whole-frame encode. Regimes mirror the
  decoder bench: skewed (prob 248) and balanced (prob 128, the renorm
  worst case) `write_bool` streams, the `write_literal` L(8) idiom, and
  the `write_signed_literal` §9.3 magnitude+sign idiom. Each measured
  iteration builds and finishes a fresh partition (vector growth is
  inseparable from real frame emission); every stream is round-trip
  decoded at setup by the crate's own `BoolDecoder`.

### Baselines (`--warm-up-time 2 --measurement-time 6`, Apple M4-class aarch64, macOS 25.1, rustc stable)

| Bench | Time | Per-element |
|---|---:|---:|
| `arnr_build_altref/arnr_5f_128x128_static_noise` | **1.294 ms** | 12.66 Mpx/s |
| `arnr_build_altref/arnr_3f_128x128_translating` | 656.9 µs | 24.94 Mpx/s |
| `arnr_build_altref/arnr_5f_128x128_strength0` | 535.5 ns | (plane-copy floor) |
| `bool_encoder_write/write_bool_skewed_64k` | 120.98 µs | 1.85 ns/bool |
| `bool_encoder_write/write_bool_balanced_64k` | 190.95 µs | 2.91 ns/bool |
| `bool_encoder_write/write_literal_8b_8k` | 182.23 µs | 2.78 ns/bool |
| `bool_encoder_write/write_signed_literal_7b_8k` | 137.06 µs | 2.39 ns/bool |

Findings feeding the rest of the round: the ARNR filter costs ≈ 2 400×
its pass-through floor on a five-frame window — every candidate probe
of the refinement search fetches its displaced block through a per-pixel
edge-clamping `pel()` even when the block is fully in-bounds (the
overwhelmingly common case away from frame edges), and the blend
accumulation pays the same per-pixel clamp. The write-side bool coder
shows the same balanced-vs-skewed spread as the decoder (≈ +57 % per
bool at prob 128) with the renormalisation loop + byte-emit dominating
the balanced regime.

## Round 409 — ARNR fast paths (profile-opt, part 2)

The new `arnr_build_altref` baseline showed the temporal filter paying a
per-pixel edge-clamping `pel()` fetch (two `clamp`s + a row-base
multiply) on **every** SAD probe pixel and every blend-accumulation
pixel, plus an integer division per blended pixel — even though away
from frame edges (the overwhelmingly common case) every displaced block
is fully in-bounds. Three levers, none of which changes a single output
pixel:

* **in-bounds fast paths** — `block_sad` and both accumulation loops
  (`accumulate_luma_block` / `accumulate_chroma_block`) now test the
  displaced block's bounds once and take straight row slices when it is
  interior; the original clamped per-pixel form is kept verbatim as the
  edge path (chroma additionally requires an overhang-free output block,
  preserving the deliberate double-accumulation into clamped edge pixels
  on odd-dimension planes).
* **monotone SAD early exit** — the refinement search passes its
  incumbent `best_sad` into `block_sad`, which abandons a candidate as
  soon as the row-partial SAD reaches it. Only a *strictly smaller* SAD
  ever wins, so a winning candidate can never trigger the exit and a
  losing one is rejected either way: identical `(mv, sad)` selection.
* **weight lookup table** — the difference-driven blend weight
  `W_MAX·S / (S + d²)` depends on `d` only through `|d| <= 255`, so one
  256-entry table per `build_arnr_altref` call replaces the per-pixel
  division.

### Bit-exactness

Four new pins: `weight_table_matches_pixel_weight` (every `(d,
strength)` pair), `block_sad_fast_matches_generic` (interior + all four
corner blocks × every MV in ±16), `refine_search_matches_no_early_exit_reference`
(the full descent vs an exhaustive generic-SAD reference across
convergent / walking / hopeless scenes, every block of the frame), and
`accumulate_luma_block_fast_matches_generic` /
`accumulate_chroma_block_fast_matches_generic` (verbatim copies of the
original per-pixel loops as references; the chroma pin runs on a 25×23
odd-geometry plane so the overhang path is exercised). Full lib suite
551 (+8), all integration test binaries green — including
`altref_arnr_on_decodes_bit_exact`.

### Measured A/B (`--warm-up-time 2 --measurement-time 6`, criterion stored-baseline delta, Apple M4-class aarch64, stable)

| Bench | Pre | Post | Δ |
|---|---:|---:|---:|
| `arnr_build_altref/arnr_5f_128x128_static_noise` | 1.294 ms | **513.2 µs** | **−60.4 %** |
| `arnr_build_altref/arnr_3f_128x128_translating` | 656.9 µs | **278.0 µs** | **−57.8 %** |
| `arnr_build_altref/arnr_5f_128x128_strength0` | 535.5 ns | 548.1 ns | within noise (untouched floor) |

Throughput on the steady-state denoise shape rises from 12.7 to
31.9 Mpx/s (2.52×).

## Round 409 — fused §13.3 decode → §14.1 dequant (profile-opt, part 3)

The standing r298 candidate. `MbDequantFactors::dequantize` re-walked
all twenty-five 4×4 blocks of every coded macroblock (400
occupancy-independent multiplies + an 800-byte load/store pass) after
the token descent had already visited exactly the non-zero lanes.
`decode_block_core` is now monomorphised on a `const DQ: bool`: under
`DQ = true` each non-zero coefficient is scaled by its plane's DC/AC
factor — the identical `i32` product and `i16` truncation the second
pass performed — at the moment it is written, and the zero-run /
untouched lanes need no scaling at all (`0 × factor` truncates to `0`,
which is what the second pass stored). `decode_and_dequantize_mb`
routes through the new fused `pub(crate) decode_mb_coeffs_dequant`
walk, so both stateless and stateful whole-frame decode drivers drop
the second pass on every coded MB. Public surfaces
(`decode_mb_coeffs`, `decode_block`, `dequantize`) are unchanged; the
`DQ = false` monomorphisation is the raw walk as before.

### Bit-exactness

`fused_dequant_walk_matches_decode_then_dequantize` pins the fused walk
stream-for-stream against decode-then-dequantize over four §13 shapes:
a dense MB whose cat6 magnitudes × large factors overflow `i16` (the
truncation case), a sparse zero-run MB, a no-Y2 (B_PRED-shaped) MB,
and a skip MB — asserting coefficients, both entropy contexts, and
trailing-literal bit-position lockstep. Full suite 808 tests green on
stable; 557 lib tests green on nightly + `simd` (the SIMD §14.1 block
apply remains for the public `dequantize` API and its parity tests).

### Measured A/B (two interleaved pre/post pairs, `--warm-up-time 2 --measurement-time 6`, Apple M4-class aarch64, stable)

| Bench | Pre (pair medians) | Post (pair medians) | Δ |
|---|---:|---:|---:|
| `keyframe_decode/decode_keyframe_320x240_qi32` | 125.7 / 126.7 µs | **116.1 / 119.6 µs** | **≈ −6…−8 %** |
| `inter_decode_short_clip/inter_decode_4f_128x128_qi32` | 115.2 / 114.8 µs | **105.9 / 108.6 µs** | **≈ −5…−8 %** |
| `token_decode/inter_decode_12f_176x144_token_heavy` | 776.5 µs | 776.0 µs | ≈ −2 % (p = 0.06, token-dominated stream) |
| `token_decode/decode_mb_coeffs_{dense,sparse}_64mb` | — | — | within noise (raw walk, unchanged by design) |

Every post run sits below every pre run on both whole-frame benches.
The win is the removed occupancy-independent second pass; it is largest
on ordinary mixed-density frames and smallest on streams whose time is
already dominated by the token descent itself.

### Refreshed ranked decoder hotspot map (post r409 fusion)

| Rank | Symbol | Note |
|---|---|---|
| 1 | `dct_tokens::decode_block_core` | now also carries the fused §14.1 multiply; resists a bit-identical change |
| 2 | `inverse_transform::inverse_dct_4x4_add_into` | r294 DC-only fast path landed |
| 3 | per-frame reference-slot plane copy (`_platform_memmove`) | CoW slot representation, API-gated, own round (r308 sized the true headroom below the `shipped_*` floor) |
| 4 | `coded_header::parse_token_prob_update` | r291 zipped loop, small fixed floor |

## Round 409 — encoder profile refresh + trellis hoisting negative result (part 4)

A fresh `sample(1)` PID-attach profile of `keyframe_encode` (12 s @ 1 ms,
stable) re-ranked the encode side for the first time since the §13
trellis landed. Top self-time symbols:

| Symbol | Samples | Share |
|---|---:|---:|
| `encoder::trellis_quantize_block` | 7 396 | **≈ 80 %** |
| `encoder::estimate_block_bits` | 418 | ≈ 4.5 % |
| `encoder::treed_find_path::dfs` | 370 | ≈ 4.0 % |
| `encoder::encode_mb_block_set_with_neighbors_strength` | 358 | ≈ 3.9 % |
| `inverse_transform::inverse_dct_4x4_scalar` | 242 | ≈ 2.6 % |
| `forward_transform::forward_dct_4x4` | 224 | ≈ 2.4 % |

The §13 trellis — run per candidate mode × per block by the RD pickers —
now dominates keyframe encode outright (the r170-era `log2` / malloc
rows are long gone). The natural bit-identical restructure was attempted
and **regressed ≈ +28 %**: hoisting the context-invariant work out of
the per-(context, candidate) scan (distortion + token classification
once per candidate, probability row + §13.2 start node once per
context, the rate term assembled by a factored row-level helper with the
identical f64 accumulation order) benched at 26.5 ms vs the unhoisted
20.5 ms on `keyframe_encode/encode_keyframe_320x240_qi32`, reproducibly
across interleaved pre/post pairs and with `#[inline(always)]` on the
helper. A bisect showed the distortion-only hoist is neutral (≈ 21.0 ms,
within the shared-box noise band) — the regression comes from the
row/token hoisting itself, i.e. the current call shape (the
per-call-site `coeff_token_bits` with everything resolved inside) is
what LLVM already specialises best. Both variants were reverted; the
shipped trellis is byte-for-byte untouched. Per the r274 precedent the
negative result is recorded so the next encoder round does not re-walk
this path.

Ranked next encoder candidates (in resistance order): the
`treed_find_path::dfs` per-emission tree search (a pure function of
`(tree, leaf)` — memoisable per call site, ≈ 4 %), `estimate_block_bits`
(same f64-order constraints as the trellis), and the trellis itself
(resists hoisting; a genuinely different formulation — e.g. integer
rate units — would change decisions and is barred by the bit-identity
guard).

## Round 409 — precomputed mode-tree paths for the RD scorers (profile-opt, part 5)

The rank-3 encoder symbol from this round's profile
(`treed_find_path::dfs`, ≈ 4 % of keyframe-encode self-time) was a
depth-first search re-run on **every candidate mode scored**: `treed_bits`
walked the compile-time-constant §8.1 mode tree from scratch to find the
leaf's bit path, then chased the tree nodes again to derive the
probability indices — per candidate, per sub-block, per macroblock (the
§11.2/§11.4 B_PRED scorer alone prices ten `BMODE_TREE` walks per 4×4
sub-block). The path is a pure function of `(tree, leaf)`, so the four
priced trees (`KF_YMODE_TREE`, `UV_MODE_TREE` — shared by the keyframe
and interframe chroma scorers — `BMODE_TREE`, `IF_YMODE_TREE`) now carry
a once-per-process `LazyLock` table of per-leaf `(bit, prob_index)` step
sequences, and `treed_bits_cached` replays the table row with the same
`bool_bits` accumulation in the same step order — every priced `f64` is
bit-for-bit the DFS-walk value, so no RD decision (and no emitted byte)
can move. The DFS form is retained as the reference;
`treed_bits_cached_matches_dfs_walk` pins every leaf of every table
under both the real probability tables and a synthetic non-uniform
lookup (a wrong `prob_index` would change the sum, not alias into it).

### Measured A/B (interleaved pre/post pairs, `--warm-up-time 2 --measurement-time 8`, Apple M4-class aarch64, stable)

| Bench | Pre (run medians) | Post (run medians) | Δ |
|---|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 21.57 / 21.67 ms | **20.63 / 20.69 ms** | **≈ −4.5 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 12.24 ms | 12.14 ms | within noise (B_PRED pricing is rare on the P-frame path) |

Every keyframe post run sits below every pre run. The emission-side
`BoolEncoder::write_treed` keeps its DFS (once per *winning* mode, not
per candidate — off the hot path).

## Round 409 — `write_literal` register-local fixed-prob-128 loop (profile-opt, part 6)

The write-side twin of the round-306 `read_literal` specialisation,
now measurable thanks to this round's `bool_encoder_write` harness.
`BoolEncoder::write_literal` iterated the generic `write_bool(128)`
per bit: each call reloaded `range` / `bottom` / `bit_count` from
`self` and recomputed the §7.3 interval split with a 32-bit multiply.
At the fixed probability 128 the split collapses to a pure shift —
`1 + (((range - 1) * 128) >> 8)` is exactly `1 + ((range - 1) >> 1)` —
and hoisting the three registers into locals lets the whole MSB-first
loop run without touching `self` except to emit bytes / propagate a
carry. `write_signed_literal` (the §9.3 magnitude+sign idiom) rides the
same loop for its magnitude bits.

### Bit-exactness

`write_literal_fast_matches_generic_loop` drives 4 096 mixed-width
literals (widths 0..=32, xorshift values) through the fast path and a
reference encoder running the generic `num_bits × write_bool(128)`
loop, asserting identical emitted bytes AND identical
`(range, bottom, bit_count)` after every literal, plus equal
`finish()` output. Full suite 810 tests green.

### Measured A/B (interleaved pre/post, `--warm-up-time 2 --measurement-time 6`, Apple M4-class aarch64, stable)

| Bench | Pre | Post | Δ |
|---|---:|---:|---:|
| `bool_encoder_write/write_literal_8b_8k` | 173.5 µs | **99.4 µs** | **−43 %** |
| `bool_encoder_write/write_signed_literal_7b_8k` | 137.1 µs (baseline) | 123.0 µs | ≈ −9 % (sign bit still generic) |
| `bool_encoder_write/write_bool_{skewed,balanced}_64k` | — | — | untouched code; run-to-run drift only |

Whole-frame encode impact is negligible by design (literals carry
headers / partition sizes / MV magnitudes, not the DCT-token hot loop)
— this is a primitive-layer win in the same class as the r306 decoder
change, but this time with an isolated harness that can actually
resolve it.

## Round 441 — per-frame §13 trellis rate tables + sub-threshold early exit (profile-opt round)

A fresh `sample(1)` PID-attach profile of `keyframe_encode` (15 s @ 1 ms,
stable) confirmed the r409 ranking unchanged: `trellis_quantize_block`
at **9 328 of ≈ 11 540 self-samples (≈ 81 %)**, with
`encode_mb_block_set_with_neighbors_strength` (8 %),
`estimate_block_bits` (3.4 %), the §14.4/§14.3 transform pair (≈ 5 %
combined) trailing far behind. r409 recorded that hoisting work *inside*
the per-candidate rate call regresses (+28 %); this round removes the
per-candidate rate call instead.

Two output-invariant changes to `trellis_quantize_block`:

* **Per-frame rate tables.** `coeff_probs` is constant for a whole
  frame, and for the extra-bit-free levels (`abs ∈ 0..=4` — every
  candidate the Viterbi scan generates until `m_round ≥ 5`) the §13.3
  token rate is a pure function of
  `(plane, band, ctx3, prev_was_zero, abs)`. A `TrellisRateTables`
  (960 + 96 `f64`s) is built **once per frame** by calling the reference
  `coeff_token_bits` / `eob_token_bits` themselves, so a hot-loop lookup
  returns the bit-for-bit identical `f64` the direct call would have
  produced — costs, chosen levels, and emitted bytes cannot move. Levels
  ≥ 5 (cat extra bits, value-dependent) keep the direct call. The tables
  ride `MbRdCtx` through every trellis site (whole-block luma, Y2,
  B_PRED sub-blocks, chroma, inter residual, SPLITMV); the per-MB public
  entry keeps a `None` fallback that routes to the unchanged direct
  calls.
* **Sub-threshold early exit.** When every in-scan coefficient rounds to
  magnitude 0 the candidate set at every position is exactly `{0}`, no
  terminating position can qualify (it needs a non-zero level), and the
  traceback provably writes an all-zero block — so the scan is skipped
  and the zero block emitted directly. At working quantisers this fires
  constantly (chroma + smooth-area luma candidates during the RD mode
  fan-out).

### Bit-exactness

* `trellis_rate_tables_match_direct_calls` — every table slot bitwise
  (`f64::to_bits`) equal to the direct reference call, across all
  4 planes × 16 scan positions × 3 contexts × 2 EOB-skip states ×
  5 levels + EOB, under the §13.5 defaults AND a deterministically
  perturbed non-uniform table.
* `trellis_lut_and_early_exit_match_reference` — the pre-r441 trellis is
  retained verbatim as `trellis_quantize_block_reference`; 6 912 cases
  (2 probability tables × 3 quantisers × 4 block types × 4 magnitude
  regimes from far-sub-threshold to cat-token range × 12 seeds ×
  3 lambdas) assert level-for-level identical output for both the
  `Some(tables)` and `None` paths.
* A 54-entry golden-hash harness (decode of the 13 stream fixtures +
  keyframe/inter/altref encode matrices + their self-decodes + raw ARNR
  output) pinned before the change matches after it, hash for hash.
* Full suite 812 tests green.

### Measured A/B (`--warm-up-time 2 --measurement-time 8`, Apple M4-class aarch64, stable, shared box)

| Bench | Pre | Post (2 runs) | Δ |
|---|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 19.92 ms | **9.39 / 9.14 ms** | **≈ −53 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 12.11 ms | **8.39 / 8.12 ms** | **≈ −31 %** |

The pre numbers sit on the r409 recorded band (20.5–21.6 ms keyframe),
so the baseline is credible despite the shared box. Keyframe throughput
moves from ≈ 3.9 to ≈ **8.3 Mpx/s**.

## Round 441 — trellis scan truncation + hoisted candidate distortion (part 2)

The post-part-1 re-profile still ranks `trellis_quantize_block` #1
(6 582 of ≈ 11 540 self-samples, ≈ 57 %). Two further output-invariant
cuts to the Viterbi scan itself:

* **Scan truncation.** Positions past the last scan slot whose rounded
  candidate magnitude is non-zero admit only the level-0 candidate:
  they can never become a terminating (non-zero last-coded) slot, the
  traceback never starts past `best_term_pos`, and their drop-to-zero
  distortion is already priced into every termination cost via
  `suffix_zero_dist`. The loop now stops at `last_cand_nonzero` — on
  typical energy-compact blocks that skips most of the 16 positions
  (the survivor updates it skips are provably dead work).
* **Hoisted candidate distortion.** The per-candidate distortion term
  depends only on the magnitude, not the carried context; it was being
  recomputed per `(in_ctx, candidate)` pair (up to 3× each). It is now
  computed once per candidate, with the `m = 0` slot reusing the
  drop-to-zero value already produced for `suffix_zero_dist` —
  bit-for-bit the same f64s, consumed in the same iteration order (the
  `in_ctx × candidate` scan order, and therefore strict-`<` tie
  behaviour, is untouched).

### Bit-exactness

Same three-layer pin as part 1: the 6 912-case
`trellis_lut_and_early_exit_match_reference` stress against the
retained pre-r441 reference (which never truncates), the 54-entry
golden-hash harness (unchanged hash-for-hash), full suite 812 green.

### Measured A/B (`--warm-up-time 2 --measurement-time 8`, Apple M4-class aarch64, stable, shared box)

| Bench | Part-1 | Part-2 | Δ (part) | Δ (cumulative vs 19.92 / 12.11 ms pre) |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 9.14–9.39 ms | **6.53 ms** | **≈ −29 %** | **≈ −67 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 8.12–8.39 ms | **6.92 ms** | **≈ −15 %** | **≈ −43 %** |

## Round 441 — DC-only / all-zero fast path in the RD reconstruct leaf (part 3)

The part-2 re-profile puts `inverse_dct_4x4_scalar` at ≈ 12 % of
keyframe-encode self-time — almost all of it under
`reconstruct_block_4x4`, the per-candidate RD leaf that dequantises and
reconstructs every scored block exactly as the decoder will. The
decoder's fused kernel (`inverse_dct_4x4_add_into`, r286/r294) already
carries a DC-only shortcut; the encoder leaf now gets its twin: after
the dequant multiplies, if every AC coefficient is zero both §14.4
passes carry only the DC term, all sixteen residue outputs are
`(dq[0] + 4) >> 3`, and the reconstruction collapses to one uniform
add-clamp — or a pure predictor copy when that value is 0, which also
covers the all-zero blocks the RD fan-out and the §13 trellis produce
constantly. The shortcut reads the *truncated* `i16` DC exactly as the
transform would, so the wrap-around edge behaves identically.

### Bit-exactness

`reconstruct_block_4x4_dc_only_matches_general` — 4 096 cases (all-zero
/ small-DC / huge-DC `i16`-truncation-edge / general-block regimes ×
random predictors × random `(dc, ac)` factors) against the pre-shortcut
chain kept verbatim in the test; 54-entry golden-hash harness unchanged
hash-for-hash; full suite 813 green.

### Measured A/B (`--warm-up-time 2 --measurement-time 8`, Apple M4-class aarch64, stable, shared box)

| Bench | Part-2 | Part-3 | Δ (part) | Δ (cumulative vs pre) |
|---|---:|---:|---:|---:|
| `keyframe_encode/encode_keyframe_320x240_qi32` | 6.53 ms | **5.88 ms** | **≈ −10 %** | **≈ −70 %** |
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 6.92 ms | **6.69 ms** | ≈ −3.4 % | **≈ −45 %** |

## Round 441 — bounded early-abandon whole-pixel SAD (negative result, part 4)

Attempted and **reverted**. The inter profile's #2/#3 self-time symbols
(`group_sad_at_whole_mv` ≈ 20 %, with `sixtap_mb_luma` at ≈ 24 %) are
fed by the §17.1 whole-pixel diamond descents, whose selection is pure
strict-`<` on the SAD — so a bounded scorer (abandon a candidate once
its row-partial reaches the incumbent best; a winner always returns its
exact full SAD because partials are monotone) is provably
decision-identical, the same trick the r409 ARNR refinement search
lands. Both descents were routed through bounded variants
(per-16-pixel-row abandon for the whole-MB scorer, per-sub-block for
the SPLITMV group scorer), contract + verbatim-descent-replica
equivalence tests green, 54-entry golden hashes unchanged.

It measured **flat to slightly negative**:
`inter_encode_short_clip` 6.69 ms (part-3) → 6.73–6.78 ms over three
runs. On the descent workloads these benches drive, neighbouring
candidates sit close in SAD, so the abandon threshold is rarely crossed
before the final rows — while the per-row `acc >= limit` compare taxes
every scored candidate. Reverted per the r274 flat-candidate doctrine;
recorded so the next inter round doesn't re-walk it. (If a future
corpus shows large-displacement content where hopeless candidates
dominate, the shape to revive is the bounded scorer pair + strict-`<`
contract as described here.)

## Round 451 — §18.3 identity-pass elision + §16.4 rectangle group SAD (profile round)

Round 451 (2026-08-24) profiled both whole-frame encode paths afresh
(`sample(1)` PID-attach, 12–15 s @ 1 ms, stable, Apple M4-class
aarch64). Keyframe encode still ranks `trellis_quantize_block` #1
(≈ 48 % post-r441, down from 81 % pre-r441), with the RD driver
(≈ 15 %), `estimate_block_bits` (≈ 14 %) and the §14 transform pair
(≈ 10 %) behind it. Inter encode ranked `sixtap_mb_luma` #1 (≈ 23 %),
`group_sad_at_whole_mv` #2 (≈ 20 %), the trellis #3 (≈ 18 %). The
decode side is saturated (token core ≈ 28 %, fused IDCT ≈ 13 %,
plane-copy memmove ≈ 8 % — all previously ground or API-bound). Two
optimisations landed, both gated on a 200-entry whole-corpus
golden-hash harness (13 fixture stream decodes frame-by-frame, a
36-point keyframe-encode matrix + self-decodes, keyframe/inter/altref
stream drives ×3 encode-frame variants + stateful self-decodes, raw
ARNR at three strengths) that was byte-identical before/after every
step, plus the full 816-test suite.

### Landed: identity-pass elision in the scalar six-tap dispatchers

The §18.3 whole-pixel filter row `{0,0,128,0,0,0}` makes a §20.14 pass
the exact identity (`clamp255((128·p + 64) >> 7) = p`), so a fraction
pair with one zero component only ever needed one pass. The stable
`sixtap_2d` / `sixtap_mb_luma` / `sixtap_mb_chroma` dispatchers now
gate on the resolved tap values and run only the surviving pass
(monomorphic widened-`i32`-row loops); other pairs take the unchanged
two-pass listing. The §17 half-pixel refinement probes four such
shapes per center; decode hits them on every inter MV with exactly one
whole-pixel axis.

| Bench | Two-pass (3,5) | Elided | Δ |
|---|---:|---:|---:|
| `mb_luma_batched_16x16_horiz_only_3x0` (new) | 158 ns | **67 ns** | **−58 %** |
| `mb_luma_batched_16x16_vert_only_0x5` (new) | 158 ns | **92 ns** | **−41 %** |

### Landed: row-merged rectangle SAD for the SPLITMV group descents

Every §20.13 partition group is a rectangle of sub-blocks, but the
in-bounds group-SAD fast path accumulated one 4-byte member row at a
time. `PartitionGroups` now precomputes each group's pixel rectangle at
compile time and the §16.4 diamond descent + `sub_mv_ref` scorer route
through a shaped evaluator whose in-bounds path SADs the rectangle's
contiguous 16/8/4-byte pixel rows, dispatched once per call onto a
constant-width kernel (row-merging reorders an exact non-overflowing
`u32` sum only). Border candidates keep the per-member edge-replicating
fallback.

### Whole-frame A/B (criterion saved-baseline, `--warm-up-time 2 --measurement-time 8`, stable)

| Bench | r451 pre | post (both commits) | Δ |
|---|---:|---:|---:|
| `inter_encode_short_clip/inter_encode_4f_128x128_qi32` | 6.971 ms | **6.706 ms** | **−4.2 %** |
| `keyframe_encode/encode_keyframe_320x240_qi32` | 6.07 ms | 6.08–6.25 ms | unchanged (no MC/SAD on this path) |
| `keyframe_decode` / `inter_decode_short_clip` | 120.8 / 107.3 µs | — | unchanged (bench streams carry whole-pixel MVs; the elide pays off only on single-axis sub-pel content) |

### Negative results (recorded per the r274 flat-candidate doctrine)

* **Widened-row restructure of the general two-pass scalar kernels** —
  rewriting the two-pass luma/chroma scalar listings as widened
  `i32`-row loops benched −5 % on `mb_luma_batched_16x16` but **+26 %**
  on `mb_chroma_batched_8x8` (43.4 → 54.7 ns): LLVM already
  auto-vectorises the shipped listings well, and the chroma geometry
  regressed outright. Reverted; only the single-pass elision arms keep
  the widened-row shape.
* **Per-call rectangle detection inside `group_sad_at_whole_mv`** —
  detecting the bounding box + ascending-members gate per call (and
  dispatching per row) measured **+19 %** on `inter_encode_short_clip`:
  the extra body destroyed the inlined descent loop. The shipped form
  moves the geometry to compile time and leaves the original function
  intact as the fallback.
* **Per-step rate-table replay in `estimate_block_bits`** — extending
  the r441 `TrellisRateTables` with per-(plane, band, ctx3,
  prev-was-zero, level) sequences of the individual `bool_bits` step
  values (replayed one add at a time in emission order, so the running
  `f64` is bit-for-bit the direct walk's) built + threaded + pinned
  clean, but benched **flat** (p = 0.52) on `keyframe_encode`: the
  estimator's cost is the `f64` adds and table traffic themselves, not
  the token classification/dispatch around them. Reverted; recorded so
  the next encoder round does not re-walk it. (A summed-subtotal LUT —
  the shape that IS a win inside the trellis — is barred here: the
  estimator's reference accumulates per-step into one running `f64`, so
  subtotal grouping changes ulps and could flip RD near-ties.)
* **Super-halo fused refinement scorer** — prefetching one in-bounds
  22×22 window per §17 refinement center and scoring all 9 candidates
  through a strided fused sixtap+SAD kernel (skipping the 8 per-probe
  21×21 halo fetches and the 256-byte prediction materialisation)
  measured **+7.6 %** on `inter_encode_short_clip`: the strided
  reimplementation loses more to codegen than the fetch amortisation
  recovers — the contiguous-halo monomorphic kernels + `block_sad_16x16`
  are simply better compiled. Reverted (equivalence pin and all) — the
  shape to revive, if ever, is contiguous per-probe sub-halo copies,
  not strided reads.

Fuzz: a bounded ASan session over the five touched-path targets
(≈ 31 M combined executions) found nothing; 226 session-discovered
coverage inputs were folded into committed seed sets (see
`fuzz/README.md`).
