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

**Per-MB QP refinement (round-31 / #522)** — the segment classifier
optionally derives its three variance breakpoints from the actual
per-frame variance distribution (population quartiles) instead of the
static `SEGMENT_VARIANCE_THRESHOLDS` ladder. This is the per-MB QP
refinement webp's lossy encoder was waiting on (#479): on diverse
content where one half of the frame is smooth and the other textured
the static thresholds lump every MB into segment 0 (or 3) and the
per-segment quant deltas have nothing to bite on; the adaptive variant
keeps every segment slot well-populated. Opt-in via
`adaptive_segment_thresholds = true`. On a 128×128 mixed-content clip
at qindex=50 it shaves ~4% bytes for +0.11 dB Y vs the static-threshold
baseline; combined with the long-ref lambda boost (below) it lands
~14% smaller for +0.77 dB Y. Falls back to the static table on
flat-or-tiny frames so the legacy single-segment behaviour is preserved
bit-for-bit.

**SPLIT_MV joint refinement (round-31 / #522)** — after the initial
per-partition motion search picks each partition's MV independently,
optional joint-refinement passes walk every partition again,
hill-climbing each MV in a 3×3 quarter-pel neighbourhood while holding
the others fixed and accepting moves that strictly reduce the
partition's sub-pel-filtered SAD. Catches boundary cases where the
independent search lands one quarter-pel off the joint optimum. Opt-in
via `enable_split_mv_joint_refine = true` + `split_mv_joint_refine_passes`
(0..=`SPLIT_MV_JOINT_REFINE_PASSES_MAX`). Same commit straightens out
a latent bug where the initial pass wrote MVs to `part_mvs[sub_block_idx]`
instead of `part_mvs[partition_id]`, silently aliasing the per-partition
MVs for SPLIT_16X8 / QUARTERS — the mode-0/2 reconstructions were
taking partition 0's MV for both halves until #522. Joint refine off
(default) so legacy callers see the prior search behaviour.

**Long-reference lambda tilt (round-31 / #522)** — the per-MB picker's
Lagrangian cost gets a per-reference scale: GOLDEN / ALTREF candidates
have lambda multiplied by `lambda_long_ref_scale_x256 / 256` (default
`256` = no change; `320` ≈ +25% is the libvpx ballpark). Drift across
the GOP makes long-term references disproportionately expensive in the
residual coding tail; boosting their lambda makes the rate term weigh
more on those candidates so the picker only takes them when the
distortion improvement is large enough to justify the higher amortised
cost. On the mixed-content clip with this knob set to 320, the
encoder shaves ~8% bytes for +0.60 dB Y. Set to `256` to recover the
uniform-lambda behaviour bit-for-bit.

**Trellis quantisation (round-32)** — after the per-MB quantiser
produces the initial coefficient array, an optional backward dynamic
programme sweeps from the last non-zero coefficient toward the start,
trying to move the EOB earlier by zeroing trailing coefficients whose
rate cost exceeds their distortion contribution. Distortion is
`(q·step)²` (dequant-domain squared error, consistent with the
`vp8_optimize_b` formulation in libvpx); rate is from the
`PROB_COST_BITS_X256` bool-coder cost table in 1/256-bit units; lambda
is `step² / 16` per block so the aggressiveness scales with QP without
extra configuration. Covers all planes (Y, Y2, U, V) on keyframes and
P-frames. Only the quantised coefficient array is modified — the
in-loop reconstruction is unchanged, keeping the decoder's reference
frames bit-identical to the non-trellis path. Opt-in via
`Vp8EncoderConfig::enable_trellis_quant = true`; off by default.
On the mixed-content test clip at qindex=50 this trims ~5–7% bytes.

**Rate-aware sub-pel ME (round-32)** — the quarter-pel 3×3
hill-climb in `subpel_refine_luma` optionally adds
`λ × mv_component_cost_x256 / 256` to the effective cost of each
fractional-pel candidate (using the same `DEFAULT_MV_CONTEXT` and
`mv_component_cost_x256` table the Lagrangian mode picker uses), so
the hill-climb biases toward entropy-cheaper MVs when multiple
neighbours have similar pixel distortion. Opt-in via
`Vp8EncoderConfig::enable_subpel_mv_cost = true`; off by default.
Effective only when `enable_rdo = true`; no measurable PSNR
regression on the mixed-content test clip.

**Two-pass ABR rate control (round-36)** — a functional two-pass
interface allows the caller to scan the raw luma frames cheaply
before any encoding, then distribute QP across the clip optimally:

```rust
use oxideav_vp8::{first_pass_analyze, two_pass_qindices, Vp8TwoPassConfig,
                  QP_SENSITIVITY_X8, make_encoder_with_config};

// Phase 1: cheap per-frame complexity scan (no ME, no transforms)
let complexity = first_pass_analyze(&frames); // frames: &[(&[u8], stride, w, h)]

// Phase 2: derive per-frame qindex
let cfg = Vp8TwoPassConfig {
    target_bitrate_bps: 1_000_000,
    fps_num: 30, fps_den: 1,
    base_qindex: 60,
    min_qindex: 20, max_qindex: 127,
    qp_sensitivity_x8: QP_SENSITIVITY_X8,
    enc_config: Vp8EncoderConfig::default(),
};
let qindices = two_pass_qindices(&complexity, &cfg);

// Phase 3: encode each frame with its assigned qindex
for (i, frame) in frames.iter().enumerate() {
    let mut enc_cfg = cfg.enc_config.clone();
    enc_cfg.qindex = qindices[i];
    let encoder = make_encoder_with_config(&codec_params, enc_cfg)?;
    // ... encode frame
}
```

The first-pass complexity score for each frame is
`score = mean_MB_variance + inter_MAD × 256`, combining spatial
texture (intra variance across all 16×16 MBs) with temporal motion
(per-pixel MAD vs. the previous frame's luma). QP assignment uses a
log-linear formula: `qindex = base_qindex − round(sensitivity × log₂(score / mean_score))`
clamped to `[min_qindex, max_qindex]`, so complex/busy frames get a
lower (higher-quality) qindex and simple/static frames get a higher
qindex. `QP_SENSITIVITY_X8 = 48` (6 QP steps per complexity
doubling) is the recommended default. A `Vp8TwoPassEncoder` wrapper
that implements the `Encoder` trait is also provided via
`make_two_pass_encoder` for callers that want to drive encoding
through the standard codec trait without managing per-frame state
manually.

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

**Perceptual RDO activity mask (round-38)** — when
`enable_psy_rdo = true`, the Lagrangian lambda is scaled per-MB by an
activity mask: `activity = MB_luma_variance + 16 × Laplacian_edge_energy`.
Flat MBs (activity < frame_mean) receive a higher lambda (fewer bits
allocated); textured/edge-rich MBs (activity > frame_mean) receive a
lower lambda (distortion-penalised). The scale is clamped to [64, 512]/256
so outlier MBs cannot dominate. Strength is set by `psy_rd_strength`
(0 = neutral / no change; default 64 ≙ ≈ ±75 % swing per 2× activity
ratio). Effective only when `enable_rdo = true`; off by default.

**NLM ARNR temporal denoising (round-38)** — when `enable_arnr_nlm = true`
and `enable_lookahead_altref = true`, the alt-ref synthesis appends an NLM
(non-local means) denoising pass over the Y plane after the Gaussian filter.
For each non-centre MC-aligned frame in the lookahead window, a sliding 5×5
patch MSE drives per-pixel weights `w = exp(−mse / h²)`; pixels from the
MC-aligned frames are blended into the composite in proportion to their
weight. Strength is set by `nlm_h2` (higher = accept noisier patches;
default 225.0 ≈ σ=15 noise tolerance). Off by default; harmless (no-op) if
lookahead is disabled.

**libvpx-shape per-coefficient Trellis (round-39)** — when
`enable_trellis_full = true` (and `enable_trellis_quant = true`),
the EOB-trim Trellis pass is preceded by a per-coefficient
forward DP analogous to libvpx `vp8_optimize_b`. For each kept
non-zero coefficient at position `n`, the DP evaluates two
candidates — the original quantised magnitude `q` and the
toward-zero magnitude `q-1` — and walks the bool-coder ctx
transitions (1 if `|c|=1`, 2 if `|c|≥2`, 0 if zero) to pick the
trajectory that minimises the block's total `D + λ·R`.
Distortion uses `(q-mag)² · step² / 2`, calibrated against the
existing trellis-lambda so the DP only accepts moves with real
rate savings. Strictly tighter than the EOB-only path: any block
that benefits from EOB-trim also benefits from at least one
position's magnitude reduction. On the mixed smooth+noise clip
this trims an additional ~1.4 % bytes (15160 → 14942) for
~0.02 dB PSNR-Y loss. Opt-in; default `false`.

**Activity-driven AQ (round-39)** — when `enable_aq = true` and
`enable_segments = true`, the per-MB segment classifier uses
population quartiles of the per-MB *activity* (variance + 16 ×
Laplacian edge energy) instead of raw variance. Smooth MBs
(low activity) land in the low-qindex segments (finer quant →
fewer banding artefacts) and textured / edge-rich MBs (high
activity) land in the high-qindex segments (coarser quant where
the eye masks the loss). Reuses the existing 4-segment bitstream
signalling — no new header bits. Falls back to the variance path
when activity is degenerate (uniform-noise frames). Strength
controlled by `aq_qindex_range` (default `8`). Opt-in; default
`false`.

Pass a custom `Vp8EncoderConfig` to `make_encoder_with_config` for
fine-grained control over `qindex`, `golden_interval`,
`alt_ref_interval`, `lambda_scale`, `enable_rdo`, `enable_multi_ref`,
`enable_segments`, `segment_quant_deltas`, `enable_scene_cut`,
`scene_cut_threshold`, `scene_cut_quant_boost`,
`scene_cut_boost_frames`, `adaptive_segment_thresholds`,
`enable_split_mv_joint_refine`, `split_mv_joint_refine_passes`,
`lambda_long_ref_scale_x256`, `enable_trellis_quant`,
`enable_trellis_full`, `enable_subpel_mv_cost`, `enable_psy_rdo`,
`psy_rd_strength`, `enable_arnr_nlm`, `nlm_h2`, `enable_aq`,
`aq_qindex_range`, and `enable_joint_lf_rdo`.

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
| `BitExact` (CI gate) | every active fixture in the corpus — keyframe + inter (`tiny-i-only-16x16`, `partition-padding-16x16-4parts`, `q-low`, `segment-4-partitions`, `i-only-loopfilter-off`, `i-only-64x64`, `webm-mux-vs-ivf-ivf`, `q-high`, `i-only-loopfilter-high`, `gradient-and-noise-128x128`, `vp8-with-loopfilter-mode-simple`, `i-frame-then-p-frame-64x64`, `golden-update-cycle`, `altref-arnr-on`, `small-roi-segmentation`) | Test fails on any divergence |
| `Ignored` | `webm-mux-vs-ivf-webm` | Disabled until oxideav-mkv is wired in for WebM demux (paired IVF version is still scored) |

Plus a two-part check for the `yuv422-not-supported` negative case:
the decoder accepts libvpx's auto-converted yuv420 stream, and the
encoder does not panic on a 4:2:2-shaped frame.

Round-30 delta (this round, see CHANGELOG): the §18.1 luma-MV
doubling step was missing from `decode_mv_component`. RFC 6386
§17.1 encodes the bitstream MV component as a quarter-pel value
V; §18.1 mandates that "the stored luma motion vectors are all
doubled, each component of each luma vector becoming an even
integer in the range -2046 to +2046, inclusive" — i.e. the
decoded value is shifted left by 1 to land at 1/8-pel resolution
(matching chroma + the 8-phase sub-pel filters). The dixie
reference decoder shipped with RFC 6386 §20.11 ends
`read_mv_component` with `return x << 1;`. Without that shift
every inter-MB MV is half its encoder-intended value, so motion-
compensated predictions point at a different reference area and
every inter or inter-adjacent intra MB drifts by a few luma units.
The encoder side (`encode_mv_component`, `mv_component_cost_x256`)
was symmetrically wrong, so encoder round-trips on our own
bitstream stayed bit-exact even pre-fix — the bug only surfaced
against bitstreams written by a spec-compliant encoder. Net pixel-
match improvement, every previously-divergent inter-frame fixture
to bit-exact:

| Fixture | Was | Now |
| --- | --- | --- |
| `small-roi-segmentation` | 78.92% | **100.00%** |
| `altref-arnr-on` | 90.75% | **100.00%** |
| `golden-update-cycle` | 96.59% | **100.00%** |
| `i-frame-then-p-frame-64x64` | 96.98% | **100.00%** |

All four ReportOnly inter-frame fixtures promote to `Tier::BitExact`;
the corpus is now uniformly bit-exact except for the `Ignored`
WebM-container fixture (paired IVF version still scored).

Round-29 delta (previous round, see CHANGELOG): RFC 6386 §17.1
`vp8_default_mv_context` trailing three long-bit probabilities
(entries `[16]`/`[17]`/`[18]`, controlling decoded high bits 7/8/9
of long-magnitude MV components) were `145/162/163` for row and
`166/172/182` for col instead of the spec's `239/254/254` and
`236/254/254`. Brought `small-roi-segmentation` 41.92% → 78.92%
(setting up the round-30 fix above to land cleanly).

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
