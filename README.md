# oxideav-vp8

Pure-Rust **VP8** video codec (RFC 6386) and IVF container for oxideav.
Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-container = "0.0"
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
| I-frame  | DC_PRED 16x16 luma + 8x8 chroma, fixed qindex, default coef probs, single partition. |
| P-frame  | REF_LAST with per-MB choice of SKIP, ZERO_MV, or NEW_MV. NEW_MV runs a full integer-pel SAD search over +-8 luma pixels and emits the MV via the bit-exact companion encoders of the decoder's MV-component reader. |

Loop filter stays off on the write side (`filter_level = 0`).

Not yet covered (planned follow-ups):

- Sub-pel refinement after the integer search.
- NEAREST / NEAR / SPLIT MV modes.
- Intra-as-fallback inside P-frames.

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
