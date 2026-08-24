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
  bit so players can suppress invisible altref-update frames.
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
byte-identical output, and the r451 pass added §18.3 identity-pass
elision on single-axis sub-pel fractions plus rectangle-row SPLITMV
group SAD — a further −4 % on the inter-encode bench), and bench / fuzz
coverage — see `BENCHMARKS.md`.

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
}
```

`Vp8DecodedFrame` (also re-exported as `Vp8Frame`) carries the visible
`width` / `height` plus three tightly-packed `Vec<u8>` planes.

### Encode a keyframe

```rust
use oxideav_vp8::{encode_keyframe, I420Frame, KeyframeParams};

let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let params = KeyframeParams::new(width, height);
let vp8_bytes: Vec<u8> = encode_keyframe(&frame, &params)?;
```

The output is a raw VP8 keyframe bitstream (3-byte tag + 7-byte start
code + partitions); wrap it in an IVF / RIFF-WEBP / Matroska container
as needed. For multi-frame inter encoding use `Vp8Encoder` +
`Vp8EncoderConfig` (both reachable standalone) and feed successive
`I420Frame`s; see `tests/encoder_inter_stream.rs` for a worked example.

### Two-pass rate-control encode

```rust
use oxideav_vp8::encoder::{two_pass_qindices, Vp8TwoPassConfig, Vp8TwoPassEncoder};

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
    } // else: reference-slot side effects only — drop the picture
}
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

let frame = I420Frame::packed(width, height, &y_plane, &u_plane, &v_plane);
let config = AdaptiveQuantConfig::default();   // base qi 32, flat→coarser / busy→finer
let vp8_bytes = encode_keyframe_adaptive_quant(&frame, &config)?;
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
let enc = make_encoder_with_quality(&params, 75.0)?;   // Box<dyn Encoder>
let enc = make_encoder_with_qindex(&params, 32)?;
```

Both factories return `Box<dyn oxideav_core::Encoder>` and integrate
with the OxideAV pipeline through `send_frame` / `receive_packet`.

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
* The 40 `fuzz/` targets cover the public encode / decode surface plus
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
