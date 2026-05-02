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
oxideav-vp8 = "0.0"
```

## Status

### Decoder

I-frame and P-frame decode against LAST / GOLDEN / ALTREF references,
with the 6-tap luma sub-pel filter, the 2-tap bilinear chroma filter,
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

## Quick use

Decode a raw VP8 frame out of an IVF file:

```rust
use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::Frame;

let mut codecs = CodecRegistry::new();
let mut containers = ContainerRegistry::new();
oxideav_vp8::register(&mut codecs, &mut containers);

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
