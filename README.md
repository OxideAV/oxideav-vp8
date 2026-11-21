# oxideav-vp8

[![CI](https://github.com/OxideAV/oxideav-vp8/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-vp8/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-vp8.svg)](https://crates.io/crates/oxideav-vp8) [![docs.rs](https://docs.rs/oxideav-vp8/badge.svg)](https://docs.rs/oxideav-vp8) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust VP8 video codec (RFC 6386) for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Decoder and encoder are both at production status.

## Status

* **Decode** — RFC 6386 key-frame and inter-frame decode, bit-exact
  against the reference output on a multi-frame fixture corpus
  (mid-GOP golden refresh, multi-frame auto-alt-ref, ARNR).
  `Vp8DecoderState::last_frame_shown()` surfaces the §9.1 `show_frame`
  bit so players can suppress invisible altref-update frames; the
  framework `Vp8Decoder` never returns them. The framework decoder
  states its output shape through `output_params()` and announces
  mid-stream size switches with one `Frame::StreamParams` event each
  (see below).
* **Encode** — full encoder with SPLITMV, GOLDEN / ALTREF,
  multi-partition output, refresh controls, loop-filter deltas, the
  §11 intra mode picker, **§13 trellis coefficient-level RDO
  quantisation** on every encode path (keyframe, intra-in-P-frame, and the
  motion-compensated / SPLITMV inter emit) — a Viterbi `J = D + λ·R` search
  over each block's level assignment, modelling the §13.3 token-context
  chain and the optimal EOB position; keyframes shed 12–13 % of bytes at
  the working quantisers while holding PSNR — the §13.4 token-probability
  fitter, §9.3 / §10
  segment-based adaptive quantisation (`encode_keyframe_adaptive_quant`),
  **§9.4 RD loop-filter `(level, sharpness)` auto-selection** on both the
  keyframe (`encode_keyframe_auto_loop_filter`) and inter
  (`encode_p_frame_multi_ref_auto_loop_filter`) paths, also exposed as a
  `Vp8TwoPassConfig::auto_loop_filter` stream knob — searches the §9.4
  level / sharpness ranges for the pair that minimises post-§15
  reconstruction SSD against the source, writing the winner to the header
  so the decoder runs the identical filter; +0.19 dB on a coarse-quantised
  blocky frame while holding byte-exact encoder↔decoder lockstep, and a
  two-pass
  rate-control family offering both a complexity-aware
  constant-quality schedule and a closed-loop **bitrate-targeting** mode
  (solve a per-GOP base qindex to a byte budget, then correct
  frame-by-frame from the running rate debt). The per-coefficient
  token emission hot path is allocation-free (§13.2 token bit paths
  precomputed into a static table).
* **B-frame-less GOLDEN / ALTREF management** — **invisible
  altref-update frames** (§9.1 `show_frame = 0`, §9.7
  `refresh_alternate_frame = 1` / `refresh_last = 0`,
  `encode_invisible_altref_update`), an **ARNR motion-compensated
  temporal filter** that synthesizes the anchor picture
  (`arnr::build_arnr_altref` — per-16×16 whole-pel alignment, SAD
  occlusion guard, difference-weighted blend), and the lagged
  **`Vp8AltrefStreamEncoder`** that assembles them: lookahead groups
  each ship one invisible anchor plus visible multi-ref P-frames, the
  §9.7 copy ladder promotes ALTREF → GOLDEN at group ends (bracketing
  reference pairs for two header bits), and a MAD scene-cut detector
  closes groups and forces keys at content changes. On 12 noisy
  translating frames at qi 44 the anchored P-frames total −31 % bytes
  vs a no-anchor multi-ref baseline, anchors included the stream is
  still smaller. Round 387 made the per-frame quality features
  **composable** (`KeyframeCodingOptions` / `InterCodingOptions` — the
  §11 intra picker, §9.4 auto loop-filter and two-pass §13.4 fitted
  token-prob updates now combine freely) and threaded them through the
  driver as `AltrefStreamConfig` toggles: on the 12-frame ARNR sequence
  the fitter alone is −8.2 % stream bytes at equal PSNR, all three
  toggles −17.5 % at equal-or-better PSNR, black-box verified by an
  `ffmpeg` cross-decode that matches our decoder **byte-exactly** on
  every visible frame. Also reachable as `Vp8Encoder::encode_sequence`
  (consuming `Vp8EncoderConfig::alt_ref_interval` /
  `lookahead_window` / `golden_interval` / `auto_loop_filter`) and, on
  the framework side, as the **lagged `oxideav_core::Encoder` adapter**
  behind `make_encoder_with_config` (`send_frame` buffers lookahead
  groups, `receive_packet` surfaces `NeedMore` mid-group, `flush`
  drains the tail; invisible anchors carry `pts = None`, only key
  frames set `flags.keyframe`; `alt_ref_interval = 0` restores zero-lag
  K/P streaming — the still-image `make_encoder` /
  `make_encoder_with_quality` doors keep their historical
  keyframe-per-frame contract).
* **Segmentation depth (round 387)** — the §9.3
  `update_segment_feature_data` block is now complete on both frame
  kinds: the adaptive-quant keyframe gained the per-segment
  **loop-filter feature**
  (`encode_keyframe_adaptive_quant_with_segment_lf_deltas`), and
  P-frames gained the full **segment-based adaptive quantisation**
  layer (`InterSegmentationConfig` /
  `encode_p_frame_multi_ref_adaptive_quant`, also an
  `AltrefStreamConfig.segmentation` stream knob): per-MB §10 variance
  classification, per-segment quantisers driving the RD walk,
  distribution-fitted `mb_segment_tree_probs`, per-MB `segment_id` in
  the §11 / §16 mode layer, and the §20.6 per-segment filter override —
  every shape `ffmpeg`-cross-decoded byte-exact. Round 387 also fixed
  two latent encoder bugs: fitted key frames now write
  `refresh_entropy_probs = 0` so the carried entropy base stays at the
  §13.5 defaults every inter encoder assumes (was: silent pixel drift
  on the first P-frame after a fitted key frame), and the intra-pick
  walk no longer double-pushes its §16.4 `split_candidates` slot (was:
  wrong split data — or a panic — whenever an intra MB preceded a
  SPLITMV MB).
* The full public surface is reachable both with the default
  `registry` build and under `--no-default-features`; a compile-only
  assertion suite lives in
  [`tests/api_compat_0_1_13.rs`](./tests/api_compat_0_1_13.rs).

VP8 has no remaining bitstream gaps in this crate; ongoing work is
encoder rate-distortion quality (the §13 trellis quantiser now covers
every encode path — keyframe, intra-within-P-frame, and the
motion-compensated / SPLITMV inter emit — and its aggressiveness is now
exposed as the **`TrellisStrength`** quality/size knob on
`KeyframeParams` / `Vp8EncoderConfig`: `DEFAULT` holds the calibrated
"shave-bits-hold-PSNR" trade, higher values spend more PSNR for fewer
bytes, `OFF` reverts to plain round-quantisation; a strength of `4.0`
shrinks a coarse-quantised keyframe ~24–40 % below the no-trellis wire for
~0.3 dB), performance tuning (SIMD primitives, profile-guided fast paths;
the trellis prices tokens from per-frame precomputed rate tables, the
whole-frame encode benches run ≈ 3× faster than the pre-r441 wire at
byte-identical output, the r451 pass added §18.3 identity-pass
elision on single-axis sub-pel fractions plus rectangle-row SPLITMV
group SAD — a further −4 % on the inter-encode bench — and r486 cut the
default key-frame encode a further ~35–40 % at identical bytes, added the
`EncodeEffort` levels and the threaded wavefront described below, and
sped the stable-build loop filter ~2.5×; r490 gave inter frames their own
effort ladder — 16–30× the previous frame rate at a third of the bits —
with wavefront threads, a one-pass rate control and a threaded decoder),
and bench / fuzz coverage — see `BENCHMARKS.md`.

## Install

```toml
# With oxideav-core for use under the OxideAV runtime:
[dependencies]
oxideav-vp8 = "0.2"

# Or standalone, without any framework dependency:
[dependencies]
oxideav-vp8 = { version = "0.2", default-features = false }
```

| Feature | Default | What it does |
|---|---|---|
| `registry` | on | Pulls `oxideav-core` and the framework-trait factories (`make_encoder` / `make_decoder` returning `Box<dyn Encoder>` / `Box<dyn Decoder>`), the `Vp8Decoder` (`oxideav_core::Decoder`) impl, and the `register*` entry points. |
| `simd` | — | **Nightly-only.** Routes selected §14 transform, §14.1 dequantize, §12 intra (TM_PRED row kernel across the §12.2 16/8 blocks + the §12.3 4×4 sub-block + §12.2 DC edge-sum reduction), §15 loop-filter, and §18.3 sub-pixel-interpolation kernels through `core::simd` rewrites. Every SIMD primitive is byte-exact against its scalar partner on a stress matrix. Default (stable) builds use the scalar paths unchanged. See `BENCHMARKS.md` for the A/B numbers. |

## Standalone use (no `oxideav-core`)

### Decode

`decode_vp8` is the one-shot keyframe entry point; multi-frame inter
sequences drive the stateful `Vp8DecoderState`, which keeps the LAST /
GOLDEN / ALTREF reference slots:

```rust
use oxideav_vp8::{decode_vp8, Vp8DecoderState};
# let vp8_keyframe_bytes: Vec<u8> = std::fs::read("key.vp8")?;
# let vp8_frame_packets: Vec<&[u8]> = vec![&vp8_keyframe_bytes];

// One-shot single-frame decode of a VP8 keyframe.
let frame = decode_vp8(&vp8_keyframe_bytes)?;
let (w, h) = (frame.width as usize, frame.height as usize);
assert_eq!(frame.y.len(), w * h);
assert_eq!(frame.u.len(), w.div_ceil(2) * h.div_ceil(2));
assert_eq!(frame.v.len(), w.div_ceil(2) * h.div_ceil(2));

// Multi-frame inter decode — one Vp8DecoderState per stream.
let mut state = Vp8DecoderState::new();
for packet in vp8_frame_packets {
    let frame = state.decode_frame(packet)?;
    // consume frame.y / frame.u / frame.v (8-bit I420, tightly packed)
    let _ = frame;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Vp8DecodedFrame` (also re-exported as `Vp8Frame`) carries the visible
`width` / `height` plus three tightly-packed `Vec<u8>` planes.

#### Decoder threads

`Vp8DecoderState::new().with_threads(n)` grants a thread budget to a
stream decode (serial by default; the registry decoder follows the
`ExecutionContext`). A still picture takes it through the one-shot
`decode_vp8_with_options`:

```rust
use oxideav_vp8::{Vp8DecodeOptions, decode_vp8_with_options};
# use oxideav_vp8::{I420Frame, KeyframeParams, encode_keyframe};
# let (w, h) = (64u32, 48u32);
# let y: Vec<u8> = (0..w * h).map(|i| (i * 5 % 253) as u8).collect();
# let c = vec![128u8; ((w / 2) * (h / 2)) as usize];
# let vp8_keyframe_bytes =
#     encode_keyframe(&I420Frame::packed(w, h, &y, &c, &c), &KeyframeParams::default())?;
let options = Vp8DecodeOptions::new()      // serial, default pixel cap
    .with_threads(4);                      // caller-granted budget
let picture = decode_vp8_with_options(&vp8_keyframe_bytes, &options)?;
# let _ = picture;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Token rows decode on up to one worker per token partition, macroblock
rows reconstruct concurrently and the loop filter runs in row bands; the
pictures are identical at every budget. **Every wait is a blocking
wait**: a row that catches up with the row above sleeps on a condition
variable until that row is a batch of macroblocks further — no worker
spins or polls, so a budget larger than the free cores costs nothing
but the idle threads.

`decode_vp8_with_options` (and every key frame of `Vp8DecoderState`)
works a macroblock row at a time — a row's coefficients are decoded,
reconstructed and dropped — where `decode_vp8` holds the dequantised
coefficients of the whole frame until reconstruction is over (800 bytes
per macroblock: 38 MiB for a 12-megapixel picture). The pictures are
bit-identical and the serial speed is the same.

Key-frame decode of the photo-like bench picture, loop filter on
(`cargo run --release --example keyframe_decode_timing`; shared machine,
minimum of 7–21 runs):

| picture | token partitions | 1 thread | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 800×533 | 1 | 4.5 ms | 4.0 ms | 3.6 ms | — |
| 4000×3000 | 1 | 78 ms | 50 ms | 46 ms | 42 ms |
| 4000×3000 | 8 | 80 ms | — | 34 ms | 28 ms |

(With one token partition the token rows are one serial chain; the
budget then only spreads reconstruction and the loop filter.)

### Encode a keyframe

```rust
use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};
# let (width, height) = (64u32, 48u32);
# let y_plane = vec![128u8; (width * height) as usize];
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();

let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let params = KeyframeParams::default();        // y_ac_qi 32; dimensions come from the frame
let vp8_bytes: Vec<u8> = encode_keyframe(&frame, &params)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The output is a raw VP8 keyframe bitstream (3-byte tag + 7-byte start
code + partitions); wrap it in an IVF / RIFF-WEBP / Matroska container
as needed.

#### Encode options: effort, threads, probability fit, loop filter

`encode_keyframe_with_options` takes a `KeyframeEncodeOptions` next to
the `KeyframeParams`. **Its default is the recommended still-picture
encode** — about twice as fast as the option-less `encode_keyframe`, 9 %
smaller and slightly closer to the source:

```rust
use oxideav_vp8::{
    EncodeEffort, I420Frame, KeyframeEncodeOptions, KeyframeParams, encode_keyframe_detailed,
    encode_keyframe_with_options,
};
# let (width, height) = (64u32, 48u32);
# let y_plane: Vec<u8> = (0..width * height).map(|i| (i * 7 % 251) as u8).collect();
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();
let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let params = KeyframeParams::default();

// The default: `Balanced` search, token probabilities fitted inside the
// one search, serial.
let vp8_bytes = encode_keyframe_with_options(&frame, &params, &KeyframeEncodeOptions::new())?;

// Every knob.
let options = KeyframeEncodeOptions::new()
    .with_effort(EncodeEffort::Fast)     // Fastest | Fast | Balanced (default) | Thorough
    .with_threads(4)                     // 0 / 1 = serial (default)
    .with_sampled_loop_filter(true);     // pick the §9.4 level / sharpness on a sample
let encoded = encode_keyframe_detailed(&frame, &params, &options)?;
// What the encoder settled on, next to the bytes and the reconstruction:
let _ = (encoded.loop_filter_level, encoded.sharpness_level);
// ... and the fitted table, to seed the search of the next similar picture.
let next = options.with_search_token_probs(encoded.token_prob_updates);
# let _ = (vp8_bytes, next);
# Ok::<(), Box<dyn std::error::Error>>(())
```

* **`effort`** — how much mode search each macroblock gets.
  `Balanced` (level 2, the default) compares every candidate mode with a
  closed-form cost model and quantises only the winners. `Fast` (1)
  additionally shortlists the sub-block modes by squared prediction
  error and skips the per-sub-block search where the whole-block
  prediction leaves no luma AC. `Fastest` (0) shortlists harder, picks
  the whole-block modes by prediction error and drops the trellis.
  `Thorough` (3) carries every candidate through transform, trellis,
  reconstruction and exact token pricing — the search of the option-less
  entry points.
* **`token_prob_fit`** (on by default) — the search records its §13.3
  token decisions on a tape instead of arithmetic-coding them; when the
  frame is done the §13.4 probability updates are fitted to the tape and
  the tape is coded against the fitted table. One search, no re-encode,
  and never a larger stream. Before the search a pilot pass over a few
  hundred macroblocks (cheapest search, predicting from the source)
  gives the search a table close to the one that will ship, which is
  worth about what a full second search against the fitted table used
  to be. `fitted_token_prob_updates` still adds that second search for
  callers who want it.
* **`search_token_probs`** — a table to search against instead (the
  `token_prob_updates` a previous, similar picture came out with);
  `encode_keyframe_detailed` returns the table each frame carries.
  Inter frames have the same pair: `InterCodingOptions::with_search_token_probs`
  and `encode_p_frame_detailed`.
* **`sampled_loop_filter`** — chooses the §9.4 loop-filter level and
  sharpness on three to six bands of macroblock rows instead of
  filtering the whole frame for every candidate (`auto_loop_filter`): at
  both bench sizes it lands on the level the full search finds.
* **`threads`** — the macroblock analysis runs as a macroblock-row
  wavefront on up to that many scoped threads (no thread pool, no
  dependency, blocking waits); the loop filter and its search, the
  pilot and the token coding of separate partitions spread over the
  same budget. **The bytes are the same at every thread budget.** With
  the fit the tokens are coded after the search, one partition a serial
  job — give a threaded encode `nbr_of_dct_partitions` 4 or 8.
* `KeyframeEncodeOptions::exhaustive()` is `encode_keyframe` byte for
  byte (`Thorough`, no fit).

Every combination emits a conformant stream that this crate's decoder
and a black-box decoder reconstruct to exactly the picture the encoder
predicted from. Measured on the photo-like bench picture at the
quality-80 operating point (`y_ac_qi` 25 with the matching trellis
strength; Apple M-series, a machine shared with other builds, minimum of
3–15 runs, old and new binaries run back to back — `cargo run --release
--example keyframe_timing`; `BENCHMARKS.md` §Round 492 has the method):

**800×533** (the default moved from the first row to the bold one)

| | 1 thread | 4 threads | bytes | PSNR-Y | SSIM-Y |
|---|---:|---:|---:|---:|---:|
| before: `Thorough` (old default) | 60.8 ms | 17.3 ms | 61 633 | 38.917 dB | 0.96721 |
| before: `Thorough` + two-pass fit | 132 ms | — | 56 181 | 38.922 dB | 0.96714 |
| before: `Balanced` | 30.2 ms | 8.8 ms | 61 656 | 38.960 dB | 0.96789 |
| before: `Fast` | 24.5 ms | 7.4 ms | 62 102 | 38.912 dB | 0.96770 |
| before: `Fastest` | 12.8 ms | 4.2 ms | 64 047 | 38.790 dB | 0.96717 |
| `Thorough` | 59.2 ms | 19.9 ms | 55 947 | 38.913 dB | 0.96723 |
| **`Balanced` (default)** | **31.0 ms** | 11.3 ms | **56 066** | **38.955 dB** | **0.96798** |
| `Fast` | 26.3 ms | 9.4 ms | 56 434 | 38.908 dB | 0.96765 |
| `Fastest` | 14.0 ms | 6.1 ms | 58 245 | 38.790 dB | 0.96717 |
| default + `sampled_loop_filter` | 36.6 ms | 12.0 ms | 56 066 | 39.360 dB | 0.97179 |
| default + `auto_loop_filter` (full search) | 60.7 ms | — | 56 066 | 39.360 dB | 0.97179 |
| `exhaustive()` | 57.3 ms | 17.4 ms | 61 633 | 38.917 dB | 0.96721 |

**4000×3000 (12 MP)**

| | 1 thread | 8 threads | bytes | PSNR-Y | SSIM-Y |
|---|---:|---:|---:|---:|---:|
| before: `Thorough` (old default) | 999 ms | 158 ms | 563 081 | 41.814 dB | 0.96284 |
| before: `Thorough` + two-pass fit | 1 994 ms | — | 499 399 | 41.832 dB | — |
| before: `Balanced` | 579 ms | 103 ms | 560 573 | 41.85 dB | — |
| before: `Fast` | 394 ms | 76 ms | 563 796 | 41.82 dB | — |
| before: `Fastest` | 229 ms | 43 ms | 582 069 | 41.74 dB | — |
| `Thorough` | 946 ms | 160 ms | 497 551 | 41.827 dB | 0.96284 |
| **`Balanced` (default)** | **488 ms** | 98 ms (88 ms with 8 partitions) | **495 698** | **41.847 dB** | **0.96310** |
| `Fast` | 373 ms | 80 ms | 498 317 | 41.823 dB | 0.96300 |
| `Fastest` | 230 ms | 55 ms | 518 110 | 41.736 dB | 0.96256 |
| default + `sampled_loop_filter` | 566 ms | — | 495 698 | 42.332 dB | 0.96695 |
| default + `auto_loop_filter` (full search) | 1 348 ms | — | 495 698 | 42.332 dB | 0.96695 |

The default is 1.96× (800×533) and 2.05× (12 MP) faster than the old
default at 9 % and 12 % fewer bytes; against the old best-compression
path (two full passes) it is 4.3× / 4.1× faster and still smaller. At a
thread budget the single-partition token coding is a serial tail of
about a tenth of the single-thread time, which is why the 4- and
8-thread columns gain less than the 1-thread column.

`Vp8EncoderConfig` and `AltrefStreamConfig` carry the same `effort` /
`threads` pair for the key frames of a stream. For multi-frame inter encoding use `Vp8Encoder` +
`Vp8EncoderConfig` (both reachable standalone) and feed successive
`I420Frame`s; see `tests/encoder_inter_stream.rs` for a worked example.

#### Inter-frame effort, threads and rate control

Inter frames take the same `EncodeEffort`. On the stream encoders it is
`inter_effort` (default `Balanced`); on the per-frame entry points it is
`InterCodingOptions::with_effort(level)` — without it those keep the
crate's original exhaustive search and its bytes, which every level
beats on both speed and bits. `threads` covers the inter frames too
(macroblock-row wavefront, loop-filter search and loop filter, alt-ref
filter); the stream is identical at every budget.

```rust
use oxideav_vp8::{EncodeEffort, I420Frame, Vp8Encoder, Vp8EncoderConfig};

# let (width, height) = (64u32, 48u32);
# let y_plane = vec![128u8; (width * height) as usize];
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();
# let i420_frames = vec![I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane); 4];
let config = Vp8EncoderConfig::default()
    .with_inter_effort(EncodeEffort::Fast)  // Fastest | Fast | Balanced (default) | Thorough
    .with_threads(4)                         // caller-granted budget; serial by default
    .with_golden_interval(120)               // key-frame interval
    .with_target_bitrate_bps(1_500_000)      // one-pass rate control; 0 = constant qindex
    .with_frame_rate(30.0);
let packets = Vp8Encoder::new(config).encode_sequence(&i420_frames)?;
# let _ = packets;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Measured on the moving test clip at `qindex` 50, one key frame then
inter frames, one thread unless noted (shared machine, medians — see
`BENCHMARKS.md` §Round 490 and
`cargo run --release --example inter_timing`):

| Inter effort | 1280×720 fps | 4 threads | kbit/s | PSNR-Y | 1920×1080 fps | 4 threads | kbit/s | PSNR-Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before this round (one search, no options) | 1.58 | — | 7 482 | 35.83 dB | 0.77 | — | 10 534 | 37.33 dB |
| `Fastest` | 50.6 | 141 | 2 693 | 34.28 dB | 23.5 | 66 | 3 877 | 36.34 dB |
| `Fast` | 33.4 | 100 | 2 652 | 34.66 dB | 16.4 | 47 | 4 010 | 36.54 dB |
| `Balanced` (default) | 25.0 | 51–76 | 2 523 | 34.69 dB | 13.6 | 54 | 3 799 | 36.62 dB |
| `Thorough` | 18.0 | 72 | 2 400 | 34.60 dB | 8.6 | 27 | 3 520 | 36.56 dB |

(Single-thread columns: 10 frames; 4-thread columns: 30 frames of the
same clip.) The levels spend about a third of the bits of the old search
at the same quantiser; at matched quality `Balanced` reaches 36.5 dB at
3 477 kbit/s (`qindex` 35) where the old search needed 7 482 kbit/s for
35.8 dB. The one-pass rate control lands within 7 % of 0.5 / 1.5 /
4 Mbit/s targets on a 2-second 720p clip at every level tried.

### Two-pass rate-control encode

```rust
use oxideav_vp8::encoder::{two_pass_qindices, Vp8TwoPassConfig, Vp8TwoPassEncoder};
# use oxideav_vp8::I420Frame;
# let (width, height) = (64u32, 48u32);
# let y_plane = vec![128u8; (width * height) as usize];
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();
# let i420_frames = vec![I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane); 4];

let config = Vp8TwoPassConfig::default();              // wraps a base Vp8EncoderConfig
let mut encoder = Vp8TwoPassEncoder::new(config);

// First pass: cheap per-frame complexity stats (cached on the encoder).
let stats = encoder.first_pass_analyze(&i420_frames)?;
// (Optional) inspect the planned schedule without encoding:
let qindices = two_pass_qindices(&config, &stats)?;

// Second pass: encode each frame against its complexity record.
let mut packets = Vec::new();
for (frame, stat) in i420_frames.iter().zip(stats.iter()) {
    packets.push(encoder.encode_frame(frame, *stat)?);   // first frame is a keyframe
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

In **constant-quality** mode (the default — no bitrate target) the
algorithm distributes per-frame qindex around `config.base.qindex` so
heavier-than-mean frames get lower qindex (better quality) and lighter
frames get higher qindex (smaller bytes), with scene-cut detection
forcing extra-quality keyframes.

#### Bitrate-targeting mode (closed loop)

Set `target_bitrate_bps` **and** a frame rate (`fps_num` / `fps_den`) and
the second pass becomes a closed-loop bitrate controller:

```rust
# use oxideav_vp8::encoder::Vp8TwoPassConfig;
let config = Vp8TwoPassConfig {
    target_bitrate_bps: 600_000,   // 600 kbps
    fps_num: 30, fps_den: 1,
    ..Vp8TwoPassConfig::default()
};
```

`first_pass_analyze` converts the bits/sec target into a per-GOP byte
budget (`gop_byte_budget`) and bisects the §9.6 qindex range for the
lowest base qindex whose predicted byte total — via the in-tree bit-cost
model (`estimate_bits_per_mb` / `estimate_frame_bytes`) — fits
`target_bitrate_bps × overshoot_ratio` (`solve_base_qindex_for_budget`).
Each `encode_frame` then re-centres the complexity delta on that solved
base and adds a feedback correction proportional to the running rate
debt (`rate_debt_bytes()`), so an over-budget stream coarsens later
frames and an under-budget one spends quality. The whole rate-control
family is clean-room — RFC 6386 is silent on rate control by design.

### Auto-altref (lagged) stream encode

`Vp8AltrefStreamEncoder` buffers source frames into lookahead groups;
each completed group emits one **invisible** ARNR anchor (a §9.1
`show_frame = 0` frame that installs a temporally-filtered picture into
the §9.7 ALTREF slot) followed by the group's visible multi-reference
P-frames:

```rust
use oxideav_vp8::{AltrefStreamConfig, I420Frame, Vp8AltrefStreamEncoder, Vp8DecoderState};

# let (width, height) = (64u32, 48u32);
# let y_plane = vec![128u8; (width * height) as usize];
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();
# let i420_frames = vec![I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane); 4];
let config = AltrefStreamConfig::default();   // window 8, ARNR 3, scene-cut on
let mut enc = Vp8AltrefStreamEncoder::new(config).unwrap();

let mut packets = Vec::new();
for frame in &i420_frames {
    packets.extend(enc.push_frame(frame)?);   // lagged: 0..N packets per push
}
packets.extend(enc.finish()?);                // drain the tail group

// The stream has MORE packets than source frames (one invisible anchor
// per group). Decode in order; display only the visible ones.
let mut dec = Vp8DecoderState::new();
for p in &packets {
    let picture = dec.decode_frame(&p.bytes)?;
    if p.is_visible() {
        // present `picture`
        let _ = picture;
    } // else: reference-slot side effects only — drop the picture
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`AltrefStreamConfig` knobs: `altref_window` (group size), `arnr`
(temporal-filter strength; `0` anchors on the raw frame),
`keyframe_interval`, `golden_promotion` (§9.7 `copy_buffer_to_golden=2`
on group-final P-frames so GOLDEN carries the previous anchor), and
`scene_cut_mad_threshold` (close the group + force a key at hard content
changes; `0.0` disables). The same pipeline is reachable through the
historical `Vp8Encoder` via `encode_sequence(&frames)`, which maps
`Vp8EncoderConfig::alt_ref_interval` / `lookahead_window` /
`golden_interval` onto the stream driver.

### Segment-based adaptive-quant keyframe encode

```rust
use oxideav_vp8::{encode_keyframe_adaptive_quant, AdaptiveQuantConfig, I420Frame};
# let (width, height) = (64u32, 48u32);
# let y_plane = vec![128u8; (width * height) as usize];
# let u_plane = vec![128u8; ((width / 2) * (height / 2)) as usize];
# let v_plane = u_plane.clone();

let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let config = AdaptiveQuantConfig::default();   // base qi 32, flat→coarser / busy→finer
let vp8_bytes = encode_keyframe_adaptive_quant(&frame, &config)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`encode_keyframe_adaptive_quant` sorts each macroblock into one of four
§10 segments by luma variance and quantises it at that segment's effective
qindex (`clamp(base_y_ac_qi + quant_delta[seg], 0, 127)`), emitting the
§9.3 `update_segmentation()` block + per-MB §10 `segment_id`. The default
gradient raises the quantiser on flat regions and lowers it on detailed
ones (activity masking). All knobs — base qindex, variance boundaries,
per-segment deltas, loop-filter — are on `AdaptiveQuantConfig`.

### Container helpers

The `ivf` module is a standalone IVF reader / writer for the common
case of `*.ivf` fixtures:

```rust
use oxideav_vp8::ivf::{IvfHeader, parse_header, write_header, write_frame};
# let (width, height, fps_num, fps_den) = (64u32, 48u32, 30u32, 1u32);
# let timed_packets: Vec<(u64, Vec<u8>)> = Vec::new();

let mut out = Vec::new();
let hdr = IvfHeader::vp8(width, height, fps_num, fps_den);
out.extend_from_slice(&write_header(&hdr));
for (pts, frame_bytes) in &timed_packets {
    write_frame(&mut out, *pts, frame_bytes);
}
```

### Quality / rate-control knobs

A `0.0..=100.0` quality scalar maps to a VP8 qindex via
`encoder::quality_to_qindex` (`100.0` → `0`, best; `NAN` → `127`,
worst-but-safe). `KeyframeParams::y_ac_qi` (range `0..=127`, default
`32`, §9.6) is the principal direct rate-control dial — every other
§9.6 quantiser delta defaults to 0, so a single value moves the DC +
AC luma / chroma quantiser bank in lockstep. Lower = higher quality /
larger output; higher = lower quality / smaller output. See
`BENCHMARKS.md` for the rate / throughput trade-off sweep.

`KeyframeParams::trellis_strength` (and `Vp8EncoderConfig::trellis_strength`)
is a second, orthogonal size dial — a `TrellisStrength` multiplier on the
§13 coefficient-RDO trellis aggressiveness. `TrellisStrength::DEFAULT`
reproduces the historically-calibrated trade (hold the PSNR floor, trim
~12–13 % of bytes); a higher strength spends more PSNR for fewer bytes
(`4.0` ≈ −24–40 % vs the no-trellis wire at mid/high quantisers for
~0.3 dB); `TrellisStrength::OFF` reverts to plain round-quantisation.
`TrellisStrength::new()` clamps into `0.0..=8.0` and maps `NaN` to the
default. The dial re-prices only the coefficient search, not the mode
decision, and the reconstruction always uses the trellis-chosen levels, so
encoder↔decoder pixel lockstep holds at every strength. It is honoured on
every encode front door — the keyframe / inter / SPLITMV residual, the
single-pass `Vp8Encoder`, the registry factories, and the two-pass
encoder via `Vp8TwoPassConfig::base.trellis_strength`.

`encoder::quality_to_trellis_strength` maps the same `0.0..=100.0` quality
scalar that drives `quality_to_qindex` onto a coherent `TrellisStrength`
(≥ 90 → `DEFAULT`, ramping to `4.0` at `0`), so a single quality number
tunes the quantiser and the trellis together; the registry
`make_encoder_with_quality` factory applies both.

## With the OxideAV runtime (`registry` feature, the default)

```rust
use oxideav_core::RuntimeContext;
use oxideav_vp8::CODEC_ID_STR;                  // = "vp8"

let mut ctx = RuntimeContext::new();
oxideav_vp8::register(&mut ctx);
// ctx.codecs now has a "vp8" Decoder + Encoder factory.

// Or directly call the factories:
use oxideav_vp8::encoder::{make_encoder_with_quality, make_encoder_with_qindex};
# let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new(CODEC_ID_STR));
# params.width = Some(64);
# params.height = Some(48);
let enc = make_encoder_with_quality(&params, 75.0)?;   // Box<dyn Encoder>
let enc = make_encoder_with_qindex(&params, 32)?;
# Ok::<(), oxideav_core::Error>(())
```

Both factories return `Box<dyn oxideav_core::Encoder>` and integrate
with the OxideAV pipeline through `send_frame` / `receive_packet`.

The registered encoder factory (`ctx.codecs.first_encoder(&params)`,
directly `make_encoder_from_param_options`) reads its settings from
`CodecParameters::options`; `make_encoder_with_options` takes the same
settings typed (`Vp8EncoderOptions`):

| Option | Kind | Default | Meaning |
|---|---|---|---|
| `effort` | u32 | `3` | mode-search effort, `0` fastest ..= `3` thorough |
| `threads` | u32 | `0` | thread budget; `0` follows `set_execution_context` (serial when none is announced) |
| `qindex` | u32 | `32` | §9.6 quantiser index `0..=127` |
| `quality` | f32 | unset | `0..=100`; replaces `qindex` and tunes the trellis |
| `loop_filter_level` | u32 | `0` | §9.4 level `0..=63` |
| `auto_loop_filter` | bool | `false` | choose the §9.4 level / sharpness per frame |
| `partitions` | u32 | `1` | §9.5 token partitions (1 / 2 / 4 / 8) |

```rust
# use oxideav_core::{CodecId, CodecParameters, ExecutionContext, RuntimeContext};
# let mut ctx = RuntimeContext::new();
# oxideav_vp8::register(&mut ctx);
let mut params = CodecParameters::video(CodecId::new(oxideav_vp8::CODEC_ID_STR));
params.width = Some(800);
params.height = Some(533);
params.options = params.options.set("quality", "80").set("effort", "1");
let mut encoder = ctx.codecs.first_encoder(&params)?;
encoder.set_execution_context(&ExecutionContext::with_threads(4));
# Ok::<(), oxideav_core::Error>(())
```

Both framework encoders take their thread budget from
`set_execution_context` (an explicit `threads` option wins); the output
never depends on it.

### Stream parameters

The framework `Vp8Decoder` answers `output_params()` with the §9.1
visible `width` / `height` of the pictures it returns, `Yuv420P`, and
the §9.2 `color_space` as the colour signal's matrix (value 0 → the
BT.601 matrix; §9.2 states no range, primaries or transfer, so those
stay unspecified). Before the first picture it answers from the key
frame already decoded, else from the first queued key frame's headers.

Only a key frame can change that description. When the first **shown**
picture of a new size is about to come out, the decoder bumps
`output_params_generation()` and returns one `Frame::StreamParams`
event carrying the new `output_params()`, then the picture. No event
for the first shape, for a key frame repeating the current one, or for
an invisible (§9.1 `show_frame = 0`) key frame — the event waits for
the first shown picture of the new size. `reset()` keeps the
description of the last picture returned, so a seek into a
different-size segment is announced the same way. The §9.1 upscale
codes and §9.2 `clamping_type` do not change the emitted planes;
`Vp8Decoder::upscale()` / `clamping_type()` expose them. The decoder
runs serially and ignores `set_execution_context`.

Both framework encoders accept a `Frame::StreamParams` event that
changes the input size: the keyframe-only adapter encodes the next
picture at the new size, and the lagged adapter closes its pending
lookahead group at the old size and restarts with a key frame. An
event naming a non-`Yuv420P` format, a non-video stream, or a size
outside the §9.1 14-bit range is `Error::Unsupported`.

## Clean-room sources

The implementation is derived entirely from the public format spec:

* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`).

The two-pass rate-control algorithm is intentionally outside RFC scope
(the spec is silent on rate control by design); its design is
clean-room, sourced from the in-tree per-MB activity primitives plus
log-MAD + log-variance first-pass statistics. Fixture reference
pictures are produced by black-box invocations of a validator binary;
no third-party codec library source is consulted at any stage.

## Test coverage

The crate ships per-stage unit tests, two interop suites, and a fleet
of `cargo-fuzz` panic-freedom / round-trip targets under
[`fuzz/`](./fuzz/):

* [`tests/standalone_e2e.rs`](./tests/standalone_e2e.rs) — keyframe,
  multi-frame inter, two-pass, IVF container, and `quality_to_qindex`
  end-to-end checks against the `default-features = false` standalone
  surface. Every imported symbol resolves without the `registry`
  feature; passes under both feature configurations.
* [`tests/blackbox_oracle.rs`](./tests/blackbox_oracle.rs) —
  bidirectional cross-validation against a black-box validator binary
  (our encode → external decode, PSNR-Y ≥ 30 dB; external encode → our
  decode via `ivf::parse_header`, PSNR-Y ≥ 25 dB). Skips cleanly when
  the validator binary isn't on `$PATH`.
* [`tests/stream_params_event.rs`](./tests/stream_params_event.rs) —
  the `output_params()` / generation / `Frame::StreamParams` contract
  (switches, flush drain, reset, invisible frames, the 13 staged
  vectors byte-identical with no event, the encoders' answer to the
  event); [`tests/blackbox_resolution_switch.rs`](./tests/blackbox_resolution_switch.rs)
  splices black-box-encoded segments of different sizes (including a
  two-pass auto-alt-ref leg with hidden frames) and checks every shown
  picture byte-identical to the validator's per-segment decode with
  exactly one change per switch.
* [`tests/encoder_keyframe_effort.rs`](./tests/encoder_keyframe_effort.rs) —
  the key-frame speed options: default options byte-identical to every
  option-less front door, the same bytes and reconstruction at every
  thread budget for every effort level, encoder↔decoder pixel lockstep
  and a black-box cross-decode at every level, a size / PSNR envelope
  for the reduced levels, and the registry option bag.
* [`tests/encoder_inter_effort.rs`](./tests/encoder_inter_effort.rs) —
  the inter-frame options: default options byte-identical to the
  option-less front door, the same bytes and reconstruction at every
  thread budget for every level and the original search,
  encoder↔decoder pixel lockstep (intra picker, token-probability fit,
  loop-filter search, segmentation, 1–4 partitions) and a black-box
  cross-decode, the ladder's size / PSNR ordering, and the one-pass rate
  control within tolerance at every level.
  [`tests/decoder_threads.rs`](./tests/decoder_threads.rs) — the
  vendored vectors, encoded streams with 1–8 token partitions and 160
  damaged streams decode identically at every thread budget.
* The 43 `fuzz/` targets cover the public encode / decode surface plus
  the per-stage primitive layers (bool coder, token coding, transforms,
  dequant, intra prediction, motion search / sub-pixel interpolation,
  inter-MB reconstruction, mode-info decode — key-frame *and*
  interframe intra variants, the §17 MV component codec under raw
  probability tables, the IVF demux loop *and* writer, the stream
  encoders including the caller-driven §9.7 refresh / §9.4 lf-deltas
  P-frame family, the silent-keyframe and per-segment loop-filter
  keyframe writers, the ARNR altref synthesiser, and the framework
  `Encoder` / `Decoder` trait adapters) for panic-freedom and, where a
  target carries an equivalence leg, byte-exact agreement between
  paired surfaces. The wire-gated decode targets run from committed
  fixture-derived seed corpora. A scheduled `Fuzz` workflow runs every
  target under ASan on a 60-minute daily budget. See
  [`fuzz/README.md`](./fuzz/README.md) for caps and run instructions.

## License

MIT. See [`LICENSE`](./LICENSE).
