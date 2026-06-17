# oxideav-vp8

Pure-Rust VP8 video codec (RFC 6386) for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Decoder and encoder are both at production status.

## Status

* **Decode** — RFC 6386 key-frame and inter-frame decode, bit-exact
  against the reference output on a multi-frame fixture corpus
  (mid-GOP golden refresh, multi-frame auto-alt-ref, ARNR).
* **Encode** — full encoder with SPLITMV, GOLDEN / ALTREF,
  multi-partition output, refresh controls, loop-filter deltas, the
  §11 intra mode picker, the §13.4 token-probability fitter, and a
  complexity-aware two-pass rate-control family. The per-coefficient
  token emission hot path is allocation-free (§13.2 token bit paths
  precomputed into a static table).
* The full public surface is reachable both with the default
  `registry` build and under `--no-default-features`; a compile-only
  assertion suite lives in
  [`tests/api_compat_0_1_13.rs`](./tests/api_compat_0_1_13.rs).

VP8 has no remaining bitstream gaps in this crate; ongoing work is
performance tuning (SIMD primitives, profile-guided fast paths) and
bench / fuzz coverage — see `BENCHMARKS.md`.

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
| `simd` | — | **Nightly-only.** Routes selected §14 transform, §14.1 dequantize, §12 intra (TM_PRED row kernel + §12.2 DC edge-sum reduction), §15 loop-filter, and §18.3 sub-pixel-interpolation kernels through `core::simd` rewrites. Every SIMD primitive is byte-exact against its scalar partner on a stress matrix. Default (stable) builds use the scalar paths unchanged. See `BENCHMARKS.md` for the A/B numbers. |

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
use oxideav_vp8::encoder::{
    first_pass_analyze, two_pass_qindices,
    Vp8TwoPassConfig, Vp8TwoPassEncoder,
};

let stats = first_pass_analyze(&i420_frames);          // cheap per-frame stats
let config = Vp8TwoPassConfig::default();              // wraps a base Vp8EncoderConfig
let qindices = two_pass_qindices(&stats, &config)?;
let encoder = Vp8TwoPassEncoder::new(config);
let packets = encoder.encode(&i420_frames, &stats)?;
```

The algorithm distributes per-frame qindex around `config.base.qindex`
so heavier-than-mean frames get lower qindex (better quality) and
lighter frames get higher qindex (smaller bytes), with scene-cut
detection forcing extra-quality keyframes.

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
* The `fuzz/` targets cover the public encode / decode surface plus the
  per-stage primitive layers (bool coder, token coding, transforms,
  dequant, intra prediction, motion search / sub-pixel interpolation,
  inter-MB reconstruction, mode-info decode, the IVF demux loop, and
  the stream encoders) for panic-freedom and, where a target carries an
  equivalence leg, byte-exact agreement between paired surfaces. A
  scheduled `Fuzz` workflow runs every target under ASan. See
  [`fuzz/README.md`](./fuzz/README.md) for caps and run instructions.

## License

MIT. See [`LICENSE`](./LICENSE).
