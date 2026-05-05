# oxideav-vp8

Pure-Rust **VP8** video codec (RFC 6386) and IVF container for oxideav.
Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.1"
oxideav-vp8 = "0.1"
```

## Standalone use (no `oxideav-core`)

Image-library consumers that just want to turn a VP8 frame buffer
into pixels — no framework, no codec registry, no trait objects —
can depend on this crate with the default `registry` feature off:

```toml
[dependencies]
oxideav-vp8 = { version = "0.1", default-features = false }
```

That drops the `oxideav-core` dependency entirely and exposes a
free-standing decode/encode API:

```rust
use oxideav_vp8::{decode_vp8, encode_vp8_keyframe, Vp8Frame, Vp8Error};

let frame: Vp8Frame = decode_vp8(&buf)?;
let encoded: Vec<u8> = encode_vp8_keyframe(frame.width, frame.height, 50, &frame)?;
# Ok::<_, Vp8Error>(())
```

`Vp8Frame` carries the cropped Yuv420P planes (`y`, `u`, `v` as
`Vec<u8>`, `width` / `height` / `y_stride` / `uv_stride` as `u32`).
`Vp8Error` covers `InvalidData` / `Unsupported` / `Eof` / `NeedMore`
— no `Io` variant; standalone callers handle their own buffer
shuffling. Turning the `registry` feature back on adds the
`Decoder` / `Encoder` trait implementations + the IVF
`Demuxer` / `Muxer` + the `register` helpers and a
`From<Vp8Error> for oxideav_core::Error` conversion so the legacy
`decode_frame` / `encode_keyframe` entry points still return the
`oxideav_core` shapes (`VideoFrame`, `Result<T, oxideav_core::Error>`).

## Status

### Decoder

I-frame and P-frame decode against LAST / GOLDEN / ALTREF references,
with the 6-tap sub-pel filter (luma AND chroma — profile 0 sets
`use_bilinear_mc_filter = 0`, see libvpx `vp8_setup_version`),
reference sign-bias and refresh / copy-buffer flags, and the in-loop
deblocking filter (simple + normal modes). Parses all inter MB modes
(NEAREST / NEAR / ZERO / NEW / SPLIT) and decodes MV diffs via the
19-entry per-component probability tree.

### Encoder

| Path     | What's emitted                                                  |
| -------- | --------------------------------------------------------------- |
| I-frame  | Per-MB choice among all 5 intra modes: DC / V / H / TM 16x16 plus B_PRED (10 per-4x4 sub-modes, greedy SSE-per-block selection), 8x8 chroma in DC / V / H / TM, fixed qindex, default coef probs, single partition. |
| P-frame  | REF_LAST with per-MB choice of SKIP, ZERO_MV, NEAREST_MV, NEAR_MV, NEW_MV, SPLIT_MV (16x8 / 8x16 / 8x8 / 4x4 partitionings with per-partition integer + quarter-pel motion search), or an intra fallback (DC/V/H/TM picked by SSE) for MBs the inter candidates cannot reconstruct. NEW_MV runs a full integer-pel SAD search over +-8 luma pixels then a quarter-pel 3x3 refinement using the same 6-tap filter the decoder applies. NEAREST / NEAR are preferred over NEW_MV when their SAD is within a small margin to save the MV-delta bits. |

**Loop filter on the write side**: enabled, level derived from `qindex`
via libvpx's heuristic `clamp(15 + qindex/8, 1, 63)`, sharpness 0,
mode/ref deltas disabled. The encoder applies the same deblocking to
its own reconstruction so subsequent P-frames reference post-filter
samples — bit-exact with what the decoder produces.

**Mode selection** is SSE-on-source for every candidate: each intra
mode is predicted from the running reconstruction and the 16x16 / 8x8
/ 4x4 SSE is summed; the lowest wins. B_PRED pays a fixed extra-bit
margin before it can beat the best 16x16 mode. SPLIT_MV pays a
per-partition margin over NEW_MV so it only wins when the per-partition
motion reduces SAD substantially.

**Lagrangian rate-distortion** P-frame mode decision (default on):
each candidate inter mode is scored as `D + λ·R` where `D` is the
luma-plane SSE of the prediction, `R` is the approximate bool-coded
cost in eighth-of-a-bit units of the per-MB mode-info (skip /
intra-vs-inter / ref-frame / MV-tree / MV deltas), and
`λ = lambda_scale · QP² / 256` with a default scale of 218 (≈
0.85·256). The rate term lets neighbour-derived MVs (NEAREST / NEAR)
beat NEWMV when the SSE is comparable, and lets shorter MV deltas
win on flat content. The per-MB reference-frame selector below uses
the same RD cost for cross-reference picking. Setting
`enable_rdo = false` falls back to the SAD-only picker.

**Alt-ref / golden-ref planning** keeps three reconstructions live
(LAST / GOLDEN / ALTREF) and refreshes GOLDEN every
`golden_interval` P-frames (default 8) and ALTREF every
`alt_ref_interval` (default 13). The per-MB picker evaluates each
populated reference and picks the one with the lowest RD cost,
emitting the appropriate `prob_last` / `prob_gf` bits in the
ref-frame tree. The per-frame inter header signals
`refresh_golden_frame` and `refresh_alternate_frame` exactly when the
schedule fires, with `copy_buffer_to_*` set to "no copy" otherwise.
Disable with `enable_multi_ref = false` to recover the legacy
single-reference behaviour.

**Context-adaptive ref-frame probabilities**: each P-frame is encoded
in two passes — pass 1 makes per-MB mode/ref decisions and tallies
the actual ref-frame distribution (intra / LAST / GOLDEN / ALT
counts); pass 2 picks the entropy-matched `prob_intra` / `prob_last`
/ `prob_gf` triple from those counts (`round(256 · n_zero / total)`,
clamped to 1..=255) and emits the frame with the optimised probs
(RFC 6386 §9.10 field J). On the existing real-content fixtures this
saves ~13% (SMPTE bars), ~23% (gray pan), and ~1% (Mandelbrot, where
residual dominates) of total bytes vs the previous fixed
`prob_intra=200`, `prob_last=128`/`1`, `prob_gf=128` literals.

**Per-MB segment maps** (RFC 6386 §10) classify each MB by source-luma
variance into one of four segments and apply a per-segment quantiser
delta. The encoder pre-computes the per-MB segment ids, picks the
entropy-matched `tree_probs` triple from the actual distribution, then
emits the segmentation block (`update_map = 1`, `update_data = 1`,
`abs_delta = 0`) at the top of the frame header. The default deltas
`[-8, -4, 0, +4]` give smooth regions a finer quant (more bits where
the eye notices banding) and high-variance regions a coarser quant
(saves bits where texture masks the loss). On a synthetic mixed
smooth+pseudo-noise clip the bit-saving variant `[0, +2, +6, +12]`
shrinks the bitstream by ~14% with sub-1 dB PSNR cost. The decoder
looks up the per-MB qi via the segment id when dequantising, so the
in-tree decode stays bit-exact with what `ffmpeg` produces. Disable
with `enable_segments = false` to recover the legacy single-segment
encoding bit-for-bit.

**Per-frame scene-cut adaptation** watches each incoming source
frame's per-pixel luma mean-absolute-difference (MAD) versus the
previous source frame, then compares it against the running
`mean(MAD) + threshold · stddev(MAD)` over the last 16 frames (with
an absolute floor of 12 luma units to suppress spurious cuts on
quiet content). When the MAD crosses both thresholds the next frame
is forced to a keyframe, the LAST / GOLDEN / ALTREF reference slots
are dropped so the keyframe rebuilds the GOP from scratch, and the
post-cut N frames receive a linearly-tapered qindex boost
(`scene_cut_quant_boost`, default 8 over `scene_cut_boost_frames=4`
frames) so the rebuild GOP starts at a higher quality floor than the
plain frame qindex would give. Disable with
`enable_scene_cut = false` to recover the legacy
single-keyframe-at-frame-0 cadence bit-for-bit.

Pass a custom `Vp8EncoderConfig` to `make_encoder_with_config` for
fine-grained control over `qindex`, `golden_interval`,
`alt_ref_interval`, `lambda_scale`, `enable_rdo`, `enable_multi_ref`,
`enable_segments`, `segment_quant_deltas`, `enable_scene_cut`,
`scene_cut_threshold`, `scene_cut_quant_boost`, and
`scene_cut_boost_frames`.

### Container

IVF read *and* write:

- **Demuxer** probes the `DKIF` / `VP80` magic, parses the 32-byte file
  header and the 12-byte per-frame length + pts prefix.
- **Muxer** emits the same file shape, patching in the final frame
  count in `write_trailer`.

### Conformance corpus

`tests/docs_corpus.rs` runs every fixture in
[`docs/video/vp8/fixtures/`](https://github.com/OxideAV/oxideav-docs/tree/master/video/vp8/fixtures)
through the in-tree decoder and scores per-plane pixel-match against
each fixture's `expected.yuv` ground truth. Three tiers:

| Tier | Fixtures | Behaviour |
| --- | --- | --- |
| `BitExact` (CI gate) | `tiny-i-only-16x16`, `partition-padding-16x16-4parts`, `q-low`, `segment-4-partitions`, `i-only-loopfilter-off`, `i-only-64x64`, `webm-mux-vs-ivf-ivf`, `q-high`, `i-only-loopfilter-high`, `gradient-and-noise-128x128`, `vp8-with-loopfilter-mode-simple` (every keyframe-only fixture in the corpus) | Test fails on any divergence |
| `ReportOnly` | 4 multi-frame inter-decode fixtures (`i-frame-then-p-frame-64x64`, `golden-update-cycle`, `altref-arnr-on`, `small-roi-segmentation`) | Logs match% + max diff; does not gate CI |
| `Ignored` | `webm-mux-vs-ivf-webm` | Disabled until oxideav-mkv is wired in for WebM demux (paired IVF version is still scored) |

Plus a two-part check for the `yuv422-not-supported` negative case:
the decoder accepts libvpx's auto-converted yuv420 stream, and the
encoder does not panic on a 4:2:2-shaped frame. The remaining
ReportOnly fixtures track residual loopfilter rounding ±1 / inter-
frame chain accumulation; each is tagged `TODO(vp8-corpus)` in the
test source.

Round-29 delta (this round, see CHANGELOG): RFC 6386 §17.1
`vp8_default_mv_context` had its trailing three long-bit
probabilities (entries `[16]`/`[17]`/`[18]`, controlling decoded
high bits 7/8/9 of long-magnitude MV components) transcribed wrong
— `145/162/163` for row and `166/172/182` for col instead of the
spec's `239/254/254` and `236/254/254`. Inter MV components in the
long-magnitude path (any |component| ≥ 1 pixel) had their high
bits decoded against near-50/50 probabilities instead of the
near-deterministic spec values, so any non-trivial encoder-written
MV decoded with wildly wrong top bits and almost always saturated
on the §16.3 `vp8_clamp_mv` lower bound. Reproduced byte-by-byte
on `small-roi-segmentation` frame 1 MB(4,0) NEW_MV decoding as
`mv=(0, -640)` (exactly the `mb_to_left_edge - MV_BORDER` clamp).
Net pixel-match improvement:

| Fixture | Was | Now |
| --- | --- | --- |
| `small-roi-segmentation` | 41.92% | **78.92%** (Y max diff 209 → 158) |
| `altref-arnr-on` | 90.36% | **90.75%** (Y max diff 99 → 21) |

Every per-MB `mode/ref/seg/skip` field in `small-roi-segmentation`
now matches the trace bit-exactly across all three frames (192
MBs); the residual ~21% pixel divergence is in inter-MB motion
compensation / sub-pel filter / dequant for non-zero MVs and is
the next natural target. The encoder side was symmetrically wrong
(`encode_mv_component` and `mv_component_cost_x256` use the same
table), so encoder round-trips on our own bitstream stayed green
even with the bad probs — the bug only surfaced against bitstreams
written by a spec-compliant encoder.

Round-28 delta (previous round, see CHANGELOG): RFC 6386 §16.3
`split_mv_tree` had three of its four leaves transcribed in the wrong
slots — the bit-`10` branch decoded as 16x8 instead of 8x8 quarters,
bit-`110` as 8x16 instead of 16x8, bit-`111` as quarters instead of
8x16. Brought `i-frame-then-p-frame-64x64` 88.57% → 96.98%,
`golden-update-cycle` 93.23% → 96.59%, `altref-arnr-on` 82.31% →
90.36%.

Round-24 deltas (previous round, see CHANGELOG): IDCT pass-order, Y2
DC-step uncap, loopfilter formula + per-MB iteration, encoder TL-
pixel defaults swap.

## Quick use

Decode a raw VP8 frame out of an IVF file:

```rust
use oxideav_core::{Frame, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_vp8::register(&mut ctx);
let codecs = &ctx.codecs;
let containers = &ctx.containers;

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("clip.ivf")?),
);
let mut dmx = containers.open("ivf", input)?;
let stream = &dmx.streams()[0];
let mut dec = codecs.make_decoder(&stream.params)?;

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            dec.send_packet(&pkt)?;
            while let Ok(Frame::Video(vf)) = dec.receive_frame() {
                // vf.format == PixelFormat::Yuv420P
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Encode into an IVF file:

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational};

let mut params = CodecParameters::video(CodecId::new("vp8"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P);
params.frame_rate = Some(Rational::new(30, 1));

let mut enc = codecs.make_encoder(&params)?;
let stream = oxideav_core::StreamInfo {
    index: 0,
    time_base: oxideav_core::TimeBase::new(1, 30),
    duration: None,
    start_time: Some(0),
    params: enc.output_params().clone(),
};
let out: Box<dyn oxideav_container::WriteSeek> =
    Box::new(std::io::Cursor::new(Vec::new()));
let mut mux = containers.open_muxer("ivf", out, std::slice::from_ref(&stream))?;
mux.write_header()?;
enc.send_frame(&Frame::Video(frame_yuv))?;
let pkt = enc.receive_packet()?;
mux.write_packet(&pkt)?;
mux.write_trailer()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Codec / container IDs

- Codec: `"vp8"`, accepted pixel format `Yuv420P`.
- Container: `"ivf"`, matches `.ivf` by extension and the `DKIF` / `VP80`
  magic bytes.

## License

MIT — see [LICENSE](LICENSE).
