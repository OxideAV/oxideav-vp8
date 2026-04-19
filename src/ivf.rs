//! IVF container — demuxer and muxer.
//!
//! IVF is a tiny, length-prefixed container used by libvpx and ffmpeg
//! for raw VP8/VP9/AV1 streams.
//!
//! 32-byte file header:
//!   0..4   "DKIF" magic
//!   4..6   version (u16-le, 0)
//!   6..8   header length (u16-le, 32)
//!   8..12  FourCC ("VP80" for VP8)
//!   12..14 width (u16-le)
//!   14..16 height (u16-le)
//!   16..20 frame rate numerator (u32-le)
//!   20..24 frame rate denominator (u32-le)
//!   24..28 frame count (u32-le, may be 0 = unknown)
//!   28..32 unused
//!
//! Each frame is preceded by:
//!   0..4 frame size in bytes (u32-le)
//!   4..12 pts in time-base units (u64-le)

use std::io::{Seek, SeekFrom, Write};

use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, PixelFormat, Rational,
    Result, StreamInfo, TimeBase,
};

const IVF_HEADER_LEN: usize = 32;
const FRAME_HEADER_LEN: usize = 12;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("ivf", open);
    reg.register_muxer("ivf", open_muxer);
    reg.register_extension("ivf", "ivf");
    reg.register_probe("ivf", probe);
}

fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() < 12 {
        return 0;
    }
    if &p.buf[0..4] != b"DKIF" {
        return 0;
    }
    if &p.buf[8..12] == b"VP80" {
        return 100;
    }
    // Other FourCCs (VP90, AV01) are still IVF — but oxideav-vp8 only
    // claims VP8.
    0
}

fn open(mut input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    let mut hdr = [0u8; IVF_HEADER_LEN];
    read_exact(&mut input, &mut hdr)?;
    if &hdr[0..4] != b"DKIF" {
        return Err(Error::invalid("IVF: bad magic"));
    }
    let version = u16::from_le_bytes([hdr[4], hdr[5]]);
    if version != 0 {
        return Err(Error::invalid(format!(
            "IVF: unsupported version {version}"
        )));
    }
    let header_len = u16::from_le_bytes([hdr[6], hdr[7]]) as u64;
    if header_len < IVF_HEADER_LEN as u64 {
        return Err(Error::invalid("IVF: header length too small"));
    }
    let fourcc = &hdr[8..12];
    if fourcc != b"VP80" {
        return Err(Error::invalid(format!(
            "IVF: unsupported FourCC {:?}",
            std::str::from_utf8(fourcc).unwrap_or("???")
        )));
    }
    let width = u16::from_le_bytes([hdr[12], hdr[13]]) as u32;
    let height = u16::from_le_bytes([hdr[14], hdr[15]]) as u32;
    let fr_num = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
    let fr_den = u32::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]);
    let _frame_count = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);

    // Skip extra header bytes if any.
    if header_len > IVF_HEADER_LEN as u64 {
        input.seek(SeekFrom::Current(
            (header_len - IVF_HEADER_LEN as u64) as i64,
        ))?;
    }

    // IVF time_base = den / num seconds per tick.
    let (tb_num, tb_den) = if fr_num == 0 || fr_den == 0 {
        (1, 1000)
    } else {
        (fr_den as i64, fr_num as i64)
    };
    let time_base = TimeBase::new(tb_num, tb_den);

    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    if fr_num != 0 && fr_den != 0 {
        params.frame_rate = Some(Rational::new(fr_num as i64, fr_den as i64));
    }

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: None,
        start_time: Some(0),
        params,
    };

    Ok(Box::new(IvfDemuxer {
        input,
        stream,
        time_base,
    }))
}

struct IvfDemuxer {
    input: Box<dyn ReadSeek>,
    stream: StreamInfo,
    time_base: TimeBase,
}

impl Demuxer for IvfDemuxer {
    fn format_name(&self) -> &str {
        "ivf"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        let mut hdr = [0u8; FRAME_HEADER_LEN];
        match read_full(&mut self.input, &mut hdr) {
            Ok(true) => {}
            Ok(false) => return Err(Error::Eof),
            Err(e) => return Err(e),
        }
        let size = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let pts = u64::from_le_bytes([
            hdr[4], hdr[5], hdr[6], hdr[7], hdr[8], hdr[9], hdr[10], hdr[11],
        ]) as i64;
        let mut data = vec![0u8; size];
        read_exact(&mut self.input, &mut data)?;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        // VP8 frame type lives in bit 0 of the first byte: 0 = key.
        pkt.flags.keyframe = !pkt.data.is_empty() && (pkt.data[0] & 1) == 0;
        Ok(pkt)
    }
}

// ---------------------------------------------------------------------------
// Muxer
// ---------------------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::invalid("IVF: exactly one video stream required"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Video {
        return Err(Error::invalid("IVF: stream must be video"));
    }
    if s.params.codec_id.as_str() != crate::CODEC_ID_STR {
        return Err(Error::invalid(format!(
            "IVF: requires codec_id={} (got {})",
            crate::CODEC_ID_STR,
            s.params.codec_id
        )));
    }
    let width = s
        .params
        .width
        .ok_or_else(|| Error::invalid("IVF: missing width"))?;
    let height = s
        .params
        .height
        .ok_or_else(|| Error::invalid("IVF: missing height"))?;
    if width == 0 || height == 0 || width > 0xffff || height > 0xffff {
        return Err(Error::invalid(format!(
            "IVF: dimensions {width}x{height} out of range (1..=65535)"
        )));
    }
    // Frame-rate hint: prefer the stream's frame_rate, fall back to the
    // time-base if unavailable. IVF stores num/den as u32 — clamp.
    let (fr_num, fr_den) = if let Some(fr) = s.params.frame_rate {
        (fr.num.max(0) as u32, fr.den.max(1) as u32)
    } else {
        let tb = s.time_base.as_rational();
        (tb.den.max(0) as u32, tb.num.max(1) as u32)
    };

    Ok(Box::new(IvfMuxer {
        output,
        width: width as u16,
        height: height as u16,
        fr_num,
        fr_den,
        frame_count: 0,
        header_written: false,
        trailer_written: false,
    }))
}

struct IvfMuxer {
    output: Box<dyn WriteSeek>,
    width: u16,
    height: u16,
    fr_num: u32,
    fr_den: u32,
    frame_count: u32,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for IvfMuxer {
    fn format_name(&self) -> &str {
        "ivf"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("IVF muxer: write_header called twice"));
        }
        let mut hdr = [0u8; IVF_HEADER_LEN];
        hdr[0..4].copy_from_slice(b"DKIF");
        // version = 0, header length = 32 (little-endian).
        hdr[4..6].copy_from_slice(&0u16.to_le_bytes());
        hdr[6..8].copy_from_slice(&(IVF_HEADER_LEN as u16).to_le_bytes());
        hdr[8..12].copy_from_slice(b"VP80");
        hdr[12..14].copy_from_slice(&self.width.to_le_bytes());
        hdr[14..16].copy_from_slice(&self.height.to_le_bytes());
        hdr[16..20].copy_from_slice(&self.fr_num.to_le_bytes());
        hdr[20..24].copy_from_slice(&self.fr_den.to_le_bytes());
        // Frame count (patched in `write_trailer`) + 4 reserved bytes.
        hdr[24..28].copy_from_slice(&0u32.to_le_bytes());
        hdr[28..32].copy_from_slice(&0u32.to_le_bytes());
        self.output.write_all(&hdr)?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("IVF muxer: write_header not called"));
        }
        if packet.data.len() > u32::MAX as usize {
            return Err(Error::invalid("IVF: packet size exceeds 4 GiB"));
        }
        let pts = packet.pts.unwrap_or(self.frame_count as i64) as u64;
        let mut frame_hdr = [0u8; FRAME_HEADER_LEN];
        frame_hdr[0..4].copy_from_slice(&(packet.data.len() as u32).to_le_bytes());
        frame_hdr[4..12].copy_from_slice(&pts.to_le_bytes());
        self.output.write_all(&frame_hdr)?;
        self.output.write_all(&packet.data)?;
        self.frame_count = self.frame_count.saturating_add(1);
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Seek back and patch the frame_count field (bytes 24..28) in the
        // file header now that we know the total.
        let end = self.output.stream_position()?;
        self.output.seek(SeekFrom::Start(24))?;
        self.output.write_all(&self.frame_count.to_le_bytes())?;
        self.output.seek(SeekFrom::Start(end))?;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

fn read_exact(input: &mut Box<dyn ReadSeek>, buf: &mut [u8]) -> Result<()> {
    let mut got = 0;
    while got < buf.len() {
        match input.read(&mut buf[got..]) {
            Ok(0) => {
                return Err(Error::invalid(format!(
                    "IVF: unexpected EOF after {got} bytes"
                )))
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Read up to `buf.len()` bytes. Returns `Ok(true)` if the buffer was
/// completely filled, `Ok(false)` if EOF was hit before any byte was
/// read, and `Err` if EOF was hit mid-buffer.
fn read_full(input: &mut Box<dyn ReadSeek>, buf: &mut [u8]) -> Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        match input.read(&mut buf[got..]) {
            Ok(0) => {
                if got == 0 {
                    return Ok(false);
                } else {
                    return Err(Error::invalid("IVF: truncated frame header"));
                }
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn probe_recognises_dkif_vp80() {
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"DKIF");
        buf[8..12].copy_from_slice(b"VP80");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 100);
    }

    #[test]
    fn probe_rejects_other_fourcc() {
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"DKIF");
        buf[8..12].copy_from_slice(b"VP90");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }

    #[test]
    fn probe_rejects_non_dkif() {
        let p = ProbeData {
            buf: b"RIFF................VP80",
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }

    fn make_stream(width: u32, height: u32, fr_num: i64, fr_den: i64) -> StreamInfo {
        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        params.width = Some(width);
        params.height = Some(height);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(fr_num, fr_den));
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(fr_den, fr_num),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    #[test]
    fn mux_then_demux_roundtrips_headers_and_payloads() {
        // Use an `Arc<Mutex<Vec<u8>>>` as the backing store so we can
        // recover the written bytes after the muxer is dropped.
        let shared: std::sync::Arc<std::sync::Mutex<Vec<u8>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        struct SharedSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>, u64);
        impl Write for SharedSink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                let mut v = self.0.lock().unwrap();
                let pos = self.1 as usize;
                if pos + b.len() > v.len() {
                    v.resize(pos + b.len(), 0);
                }
                v[pos..pos + b.len()].copy_from_slice(b);
                self.1 += b.len() as u64;
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl Seek for SharedSink {
            fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
                let len = self.0.lock().unwrap().len() as u64;
                let new = match pos {
                    SeekFrom::Start(o) => o,
                    SeekFrom::Current(o) => (self.1 as i64 + o) as u64,
                    SeekFrom::End(o) => (len as i64 + o) as u64,
                };
                self.1 = new;
                Ok(new)
            }
        }

        let stream = make_stream(320, 240, 30, 1);
        let payload0 = vec![0xaau8; 37];
        let payload1 = vec![0x5au8; 19];
        let mut p0 = Packet::new(0, stream.time_base, payload0.clone());
        p0.pts = Some(0);
        let mut p1 = Packet::new(0, stream.time_base, payload1.clone());
        p1.pts = Some(1);

        let sink = SharedSink(shared.clone(), 0);
        let mut mux = open_muxer(Box::new(sink), std::slice::from_ref(&stream)).expect("mux");
        mux.write_header().unwrap();
        mux.write_packet(&p0).unwrap();
        mux.write_packet(&p1).unwrap();
        mux.write_trailer().unwrap();
        drop(mux);

        let bytes = shared.lock().unwrap().clone();
        assert_eq!(&bytes[0..4], b"DKIF");
        assert_eq!(&bytes[8..12], b"VP80");
        assert_eq!(u16::from_le_bytes([bytes[12], bytes[13]]), 320);
        assert_eq!(u16::from_le_bytes([bytes[14], bytes[15]]), 240);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            2,
            "trailer must patch frame_count"
        );

        let mut dmx = open(
            Box::new(Cursor::new(bytes)),
            &oxideav_core::NullCodecResolver,
        )
        .expect("demux");
        let streams = dmx.streams().to_vec();
        assert_eq!(streams[0].params.width, Some(320));
        assert_eq!(streams[0].params.height, Some(240));
        let r0 = dmx.next_packet().expect("pkt 0");
        assert_eq!(r0.data, payload0);
        assert_eq!(r0.pts, Some(0));
        let r1 = dmx.next_packet().expect("pkt 1");
        assert_eq!(r1.data, payload1);
        assert_eq!(r1.pts, Some(1));
        assert!(matches!(
            dmx.next_packet().err(),
            Some(oxideav_core::Error::Eof)
        ));
    }
}
