//! Timed decode of an IVF file.
//!
//! `cargo run --release --example ivf_decode -- <file.ivf> [RUNS [THREADS...]]`
//!
//! Decodes the whole file `RUNS` times at each thread budget (default
//! `1`) and prints the median time per packet and packets per second. A
//! checksum over the decoded luma must not change with the budget; the
//! example asserts it.

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: ivf_decode <file.ivf> [RUNS [THREADS...]]");
    let ivf = std::fs::read(path).expect("readable IVF file");
    let runs: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let budgets: Vec<usize> = if args.len() > 3 {
        args[3..]
            .iter()
            .map(|v| v.parse().expect("thread count"))
            .collect()
    } else {
        vec![1]
    };
    let mut packets = Vec::new();
    let mut at = 32usize;
    while at + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[at..at + 4].try_into().expect("4 bytes")) as usize;
        packets.push(&ivf[at + 12..at + 12 + size]);
        at += 12 + size;
    }
    println!("{} packets, {runs} runs per budget (median)", packets.len());
    println!("threads  ms/packet  packets/s  checksum");
    let mut reference = None;
    for &threads in &budgets {
        let mut times = Vec::with_capacity(runs);
        let mut sum = 0u64;
        for _ in 0..runs {
            let mut dec = oxideav_vp8::Vp8DecoderState::new().with_threads(threads);
            let t = Instant::now();
            sum = 0;
            for p in &packets {
                let d = dec.decode_frame(p).expect("decode");
                sum = sum
                    .wrapping_mul(31)
                    .wrapping_add(d.y.iter().step_by(97).map(|&b| b as u64).sum::<u64>());
            }
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        assert_eq!(
            *reference.get_or_insert(sum),
            sum,
            "decoded pictures changed with the thread budget"
        );
        let ms = times[runs / 2] / packets.len() as f64;
        println!("{threads:>7} {ms:>10.3} {:>10.1}  {sum:016x}", 1e3 / ms);
    }
}
