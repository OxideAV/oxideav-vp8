# oxideav-vp8

Pure-Rust VP8 video codec for oxideav (RFC 6386) + IVF container.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Status

- **Decoder**: I-frame + P-frame decode (LAST/GOLDEN/ALTREF refs, 6-tap luma +
  bilinear chroma sub-pel, sign-bias & refresh/copy-buffer flags, loop filter).
- **Encoder**:
  - I-frame (keyframe): DC_PRED 16×16 luma + 8×8 chroma, fixed qindex,
    default coefficient probabilities, single token partition.
  - **P-frame**: ZERO_MV with REF_LAST on every MB, per-MB `mb_skip` flag
    for MBs that match the reference well (SAD-based heuristic). Produces
    dramatically smaller bitstreams than all-keyframe on static / slow
    content. No NEWMV / NEAREST / NEAR / SPLIT motion modes yet — NEWMV
    with motion search is a planned follow-up.
- **Container**: IVF read-side demuxer with FourCC `VP80` probe.

## Usage

```toml
[dependencies]
oxideav-vp8 = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
