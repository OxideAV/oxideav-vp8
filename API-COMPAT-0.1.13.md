# Public API compatibility target — crates.io `oxideav-vp8 0.1.13`

This file pins the **minimum** public surface that the current
`oxideav-vp8` master must keep exposing so that historical consumers
pinned to `oxideav-vp8 = "0.1"` can upgrade transparently. The current
master is on `0.2.0` (a major bump after the orphan rebuild line),
which by SemVer allows breaking changes — but the user-stated goal
(2026-05-27) is **upgrade-transparent migration for historical
consumers**, so we MUST keep the 0.1.13 surface reachable, even if its
internal implementation is now a thin adapter over the post-rebuild
`Vp8InterStreamEncoder` and friends.

> Source of every signature below: per-version rustdoc at
> `https://docs.rs/oxideav-vp8/0.1.13/`. Recovered from the crate-root
> index + per-module index pages. Not from the removed `src/`, not from
> libvpx, not from libwebp, not from FFmpeg.

## Cargo features (verbatim from the 0.1.13 manifest)

```toml
[features]
default = ["registry"]
registry = ["dep:oxideav-core"]
simd     = []                       # no-op flag — reserved
```

- `default-features = false` ⇒ `oxideav-core` is **not** linked. This
  is the no-oxideav-core standalone build. Historical users who set
  `default-features = false` to embed vp8 in a non-framework pipeline
  MUST keep building after the upgrade.
- The `simd` feature was a no-op in 0.1.13. Keep it declared (no-op)
  so consumers don't fail to set it; add a doc comment that it is
  reserved for future SIMD code-paths and currently has no effect.

## Crate-root constant

```rust
pub const CODEC_ID_STR: &str = "vp8";
```

Status today: **MISSING** at the crate root. Likely lives inside the
`encoder` or `decoder` registration code. Re-export it.

## Crate-root re-exports (must all be reachable at `oxideav_vp8::<name>`)

Group / item / status legend:
- **ALREADY** — currently re-exported under this exact name. No change.
- **RENAME** — exists under a different name; add a `pub use` alias.
- **MISSING** — does not exist in current master. Add a minimal stub
  type or thin-adapter function whose **shape** matches the documented
  signature, even if its body is "delegate to <new internal API>" or
  "wrap a value, no behaviour change."
- **WIDEN** — exists but with a narrower signature; widen it.

### From `decoder`

```rust
pub use decoder::{decode_vp8, decode_frame, Vp8Decoder};
```

- `decode_vp8` — **ALREADY** (currently re-exported).
- `decode_frame` — **MISSING**. Doc: "Legacy alias of `decode_vp8`
  yielding `oxideav_core::VideoFrame` instead of `Vp8Frame`. Gated on
  the `registry` feature; standalone callers should use `decode_vp8`."
  Add as a thin wrapper that calls `decode_vp8` and converts the
  result into the `oxideav-core` frame shape; gate on
  `#[cfg(feature = "registry")]`.
- `Vp8Decoder` — **MISSING** at this path. Currently the typed handle
  is `Vp8DecoderState` (re-exported from `state` module). Either
  rename `Vp8DecoderState` → `Vp8Decoder` (preferred) or add a
  `pub type Vp8Decoder = Vp8DecoderState;` alias at the `decoder`
  module level.

### From `error`

```rust
pub use error::{Result, Vp8Error};
```

- `Result` — **MISSING** at this path. Add `pub type Result<T> =
  core::result::Result<T, Vp8Error>;` to `error.rs` and re-export.
- `Vp8Error` — **ALREADY** (landed in commit `0ee6e65`).

### From `frame`

```rust
pub use frame::Vp8Frame;
```

- `Vp8Frame` — **MISSING** at this name. Currently the per-decoded-
  frame container is `Vp8DecodedFrame` (re-exported from `decoder`).
  Add `pub type Vp8Frame = Vp8DecodedFrame;` (preferred) or rename if
  the rename can be done without breaking too many internal
  call-sites. Re-export at both `frame::Vp8Frame` and crate root.

### From `frame_header`

```rust
pub use frame_header::{parse_keyframe_header, FrameHeader};
```

- `FrameHeader` — likely **ALREADY** present in `frame_header.rs`;
  verify and re-export at the crate root.
- `parse_keyframe_header` — verify name match; the current crate may
  use a different verb (`decode_*`, `read_*`). If different, add a
  thin alias `pub fn parse_keyframe_header(buf: &[u8]) ->
  Result<KeyframeHeader> { … }`.

### From `frame_tag` (new module path — may need creating)

```rust
pub use frame_tag::{parse_header, FrameTag, FrameType, KeyframeHeader, ParsedHeader};
```

This whole module was split out of frame_header in 0.1.13. The
current master likely keeps the types inside `frame_header.rs`. Two
options:
1. **Preferred** — add `pub mod frame_tag { /* re-exports */ }` that
   re-exports the existing types from their current home; consumers
   that wrote `oxideav_vp8::frame_tag::FrameType` keep working.
2. Move the type definitions out into a new `frame_tag.rs` (more
   invasive; only do this if (1) bumps into an orphan-rule corner).

Items to expose:
- `parse_header(buf: &[u8]) -> Result<ParsedHeader>` — top-level
  frame-tag parser. Doc: "Parses the 3-byte frame tag at the start of
  every VP8 frame. Returns a `ParsedHeader` that carries the
  `FrameTag` plus, for keyframes, the 7-byte start-code + width/height
  in a `KeyframeHeader`."
- `FrameTag` — struct holding the 3-byte tag bit-fields
  (`frame_type`, `version`, `show_frame`, `first_partition_size`).
- `FrameType` — enum `{ Key, Inter }`.
- `KeyframeHeader` — struct holding the 7-byte keyframe extension
  (start-code, scaled-width / scaled-height with their 2-bit upscale
  fields).
- `ParsedHeader` — struct combining `tag: FrameTag` + `keyframe:
  Option<KeyframeHeader>` (Some iff frame_type == Key).

If renaming would break internal call-sites, expose these as thin
wrapper types/functions whose body delegates to the existing
internal parsers.

### From `registry` (registry feature only)

```rust
#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};
```

- `register` — **ALREADY** (currently `decoder::register`, re-exported
  at crate root). Move/expose at `registry::register` (preferred —
  matches 0.1.13's module layout) or add a `pub mod registry` that
  re-exports the existing function.
- `register_codecs` — **MISSING**. Add: registers only the VP8 codec,
  no container.
- `register_containers` — **MISSING**. Add: registers the IVF
  container (see "Module `ivf`" below). For backward compat the body
  can be a no-op for now if IVF support isn't fully wired — but the
  symbol MUST exist.

## Modules that must exist (may add more, must not remove)

The current master has 21 modules. 0.1.13 had 18, several with
different names. Map the existing modules onto the 0.1.13 names by
adding **alias modules** (re-export shells) without renaming the
underlying files:

```rust
// In src/lib.rs:
pub mod fdct       { pub use crate::forward_transform::*; }
pub mod inter      { pub use crate::motion_comp::*;    pub use crate::motion_search::*;
                     pub use crate::motion_vector::*;  pub use crate::near_mv::*; }
pub mod intra      { pub use crate::intra_predict::*; }
pub mod loopfilter { pub use crate::loop_filter::*; }
pub mod mv         { pub use crate::motion_vector::*; }
pub mod tables     { /* re-export any in-tree default-prob / scan / dequant tables here */ }
pub mod tokens     { pub use crate::dct_tokens::*; }
pub mod transform  { pub use crate::inverse_transform::*; }
pub mod bool_encoder { /* see below — may need to create */ }
pub mod frame_tag    { /* see frame_tag section above */ }
pub mod ivf          { /* see ivf section below */ }
pub mod registry     { /* see registry section above */ }
```

Note: `pub use crate::<existing>::*;` is a low-risk alias that keeps
the file structure intact while restoring the historical module
discovery path for downstream consumers using
`oxideav_vp8::loopfilter::SomeType`.

### Module `bool_encoder`

The forward bool coder. The current master has `bool_decoder.rs`. The
encoder writes bool-coded streams inside `encoder.rs` / `dct_tokens.rs`
/ encoder-Phase code. If the entry-point type already exists (e.g.
`BoolEncoder`), expose it at `oxideav_vp8::bool_encoder::BoolEncoder`.
If it's deeply embedded, expose a minimal `pub mod bool_encoder {
/* re-exports of any bool-encoder types already in tree */ }` and
note the gap in the per-crate README.

### Module `ivf`

IVF container support (used for VP8 test fixtures + as the canonical
single-codec container). Historical 0.1.13 exposed reader + writer
helpers. If no IVF code exists in current master, add a minimal
`pub mod ivf { /* skeleton */ }` with:
- a `pub fn parse_header(buf: &[u8]) -> Result<IvfHeader>`
- a `pub struct IvfHeader { width: u32, height: u32, framerate_num:
  u32, framerate_den: u32, frame_count: u32, fourcc: [u8; 4] }`
- a `pub fn write_header(...) -> Vec<u8>`
- a `pub fn write_frame(buf: &mut Vec<u8>, pts: u64, frame: &[u8])`

These can have minimal "wraps a single frame" bodies; the test corpus
under `tests/` will tell us if real parser logic is needed.

## Encoder module surface

```rust
pub use encoder::{
    // structs
    Vp8Encoder, Vp8EncoderConfig, Vp8EncoderStats, FrameComplexity,
    Vp8TwoPassConfig, Vp8TwoPassEncoder,
    // enum
    LoopFilterMode,
    // factories
    make_encoder, make_encoder_with_config, make_encoder_with_qindex,
    make_encoder_with_quality, make_encoder_typed_with_config,
    make_two_pass_encoder,
    // free fns
    encode_keyframe, encode_vp8_keyframe, first_pass_analyze,
    two_pass_qindex_for_frame, two_pass_qindices,
    // constants — verbatim names below
    AQ_QINDEX_RANGE_MAX, DEFAULT_ADAPTIVE_SEGMENT_THRESHOLDS,
    DEFAULT_ALT_REF_INTERVAL, DEFAULT_AQ_QINDEX_RANGE,
    DEFAULT_CHROMA_AWARE_SPATIAL_CHROMA_WEIGHT_X256,
    DEFAULT_CHROMA_AWARE_SPATIAL_LUMA_WEIGHT_X256,
    DEFAULT_GOLDEN_INTERVAL, DEFAULT_JOINT_R44R49_PICKER_MAX_ITERS,
    DEFAULT_KMEANS_CONVERGENCE_THRESHOLD, DEFAULT_KMEANS_SPATIAL_ALPHA_X256,
    DEFAULT_LAMBDA_LONG_REF_SCALE_X256, DEFAULT_LOOKAHEAD_WINDOW,
    DEFAULT_NLM_H2, DEFAULT_PSY_RD_STRENGTH, DEFAULT_QINDEX,
    DEFAULT_SCENE_CUT_BOOST_FRAMES, DEFAULT_SCENE_CUT_QUANT_BOOST,
    DEFAULT_SCENE_CUT_THRESHOLD, DEFAULT_SEGMENT_LF_DELTAS,
    DEFAULT_SEGMENT_QUANT_DELTAS, DEFAULT_SIMPLE_LF_MAX_LEVEL,
    DEFAULT_SPATIAL_LF_N_COL_BANDS, DEFAULT_SPATIAL_LF_N_ROW_BANDS,
    DEFAULT_SPLIT_MV_JOINT_REFINE_PASSES,
    INTRA_IN_P_BPRED_VARIANCE_THRESHOLD,
    JOINT_R44R49_PICKER_MAX_ITERS_MAX, KMEANS_SPATIAL_MAX_ITERS,
    LAMBDA_SCALE_DEFAULT, QP_SENSITIVITY_X8, SCENE_CUT_ABS_FLOOR,
    SEGMENT_VARIANCE_THRESHOLDS, SPLIT_MV_JOINT_REFINE_PASSES_MAX,
};
```

Status:
- `make_encoder`, `make_encoder_with_qindex`, `make_encoder_with_quality` —
  **ALREADY** (commit `0ee6e65`).
- `encode_vp8_keyframe` — **ALREADY** at crate root (lib.rs:446).
- All other items — **MISSING**. Add type shapes for the structs +
  constants (literal `pub const = ...;` lines for each `DEFAULT_*`,
  matching the documented type / ballpark value); add stub bodies for
  the new factory functions that delegate to the existing `Phase 18`
  encoder behind a thin adapter that ignores the extra config knobs
  it doesn't yet implement.
- `LoopFilterMode { Auto, Normal, Simple }` — add the enum; the
  encoder's existing loop-filter routing can map all three onto its
  current single behaviour (`Auto` selects `Simple` for high
  `lf_level`, `Normal` otherwise — matches the documented selector).

**Tier-3 caveat:** the two-pass encoder (`Vp8TwoPassEncoder`,
`Vp8TwoPassConfig`, `first_pass_analyze`, `two_pass_qindex_for_frame`,
`two_pass_qindices`) is a substantial body of work. For this dispatch,
expose the **type shapes** (struct definitions, enum, constants,
function signatures) but make the bodies return a clean
`Vp8Error::Unsupported("two-pass encoder not yet implemented in this
release")`. Re-implementing the actual two-pass algorithm is a
follow-up — the goal of THIS round is the surface, not the algorithm.

## Standalone build is NON-NEGOTIABLE

```bash
cargo build -p oxideav-vp8 --no-default-features
cargo test  -p oxideav-vp8 --no-default-features
```

Both must pass. Items gated on `registry` (anything that returns
`Box<dyn oxideav_core::Encoder>` / `Box<dyn oxideav_core::Decoder>`
/ `oxideav_core::VideoFrame`) must be `#[cfg(feature = "registry")]`.
Everything else — `Vp8Error`, `Result`, `Vp8Frame`, `Vp8Decoder`,
`Vp8Encoder`, `Vp8EncoderConfig`, all `DEFAULT_*` constants,
`quality_to_qindex`, `encode_vp8_keyframe`, `parse_keyframe_header`,
`FrameHeader`, `FrameTag`, `FrameType`, `KeyframeHeader`,
`ParsedHeader`, `LoopFilterMode`, the standalone IVF helpers — must
be reachable under `--no-default-features`.

## Verification checklist

- [ ] `cargo build -p oxideav-vp8` (default registry build)
- [ ] `cargo build -p oxideav-vp8 --no-default-features` (standalone)
- [ ] `cargo test  -p oxideav-vp8`
- [ ] `cargo test  -p oxideav-vp8 --no-default-features`
- [ ] `cargo tree -p oxideav-vp8 --no-default-features --edges normal`
      shows no `oxideav-core`
- [ ] `cargo doc  -p oxideav-vp8 --no-default-features` shows every
      symbol listed in "Crate-root re-exports" section above
- [ ] Add `tests/api_compat_0_1_13.rs` compile-only assertion suite
      that imports every Crate-root re-export by fully-qualified
      name (`use oxideav_vp8::Vp8Decoder;` etc.) and binds it to a
      `let _: T = ...;` or `let _: fn(...) -> ... = name;`.
- [ ] No symbols **removed** versus the post-`0ee6e65` master; only
      added/widened.

## Out-of-scope follow-ups

- Real two-pass rate-control algorithm (the surface shapes land
  here; bodies stub to `Vp8Error::Unsupported`).
- Real SIMD code paths under the `simd` feature flag (still no-op).
- Closing the IVF parser to ffmpeg-bit-exact (skeleton only).
- Migrating internal call-sites from `Vp8DecoderState` →
  `Vp8Decoder` (use a `type` alias for now).
