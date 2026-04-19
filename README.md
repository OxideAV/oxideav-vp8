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
