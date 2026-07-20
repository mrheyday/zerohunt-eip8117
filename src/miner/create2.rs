//! CREATE2 salt-mining driver: dispatch keccak-only Metal batches, verify every
//! hit vs host keccak, report strictly-increasing bests. The found value is a
//! **salt** (public), so — unlike the EOA miner — nothing is encrypted; bests
//! are appended in plaintext to `scanned_salts.txt`.
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::types::Address;
use ethers::utils::hex;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::erc8117;
use crate::gpu::MetalContext;

/// GPU threads per batch (one salt-stream each). Matches the EOA driver's grid.
pub const C2_N_THREADS: usize = 65536;
/// Salts each thread scans per batch.
pub const C2_ITERS: u32 = 256;
/// Dispatch-threshold floor (keeps a full ~16.7M-candidate batch under the
/// 1024-hit buffer cap).
const C2_GPU_FLOOR: usize = 4;
/// GPU duty-cycle target (fraction of wall-clock spent computing).
const UTILIZATION: f64 = 0.80;

/// Threshold for the next batch: at least the floor, rising with the best; if
/// the target is below the floor, clamp to it so low targets still surface.
fn c2_threshold(best_zeros: usize, target: usize) -> u32 {
    let t = best_zeros.max(C2_GPU_FLOOR);
    let t = if target < C2_GPU_FLOOR { target } else { t };
    t as u32
}

/// Mine CREATE2 salts for `deployer` + `initcodehash` until `stop` is set or a
/// salt yielding `target` leading-zero nibbles is found. Verifies every GPU hit
/// against host keccak (a mismatch hard-aborts) and appends each new best to
/// `scanned_salts.txt` as `<total>\t<address(erc8117)>\t<zeros>\t<salt_hex>`.
pub fn run_create2(
    ctx: &MetalContext,
    deployer: &[u8; 20],
    initcodehash: &[u8; 32],
    target: usize,
    stop: Arc<AtomicBool>,
) {
    // Random per-thread base salts: the high 24 bytes give each thread a unique
    // salt space; the low 8 bytes are overwritten by the per-iter/batch counter.
    let mut rng = StdRng::from_entropy();
    let mut base_salts = vec![[0u8; 32]; C2_N_THREADS];
    for s in base_salts.iter_mut() {
        rng.fill_bytes(s);
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open("scanned_salts.txt")
        .expect("Unable to open scanned_salts.txt");
    let mut file = file;

    let mut best = 0usize;
    let mut base: u64 = 0;
    let mut total: u64 = 0;
    let start = Instant::now();
    let mut last_report = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let threshold = c2_threshold(best, target);
        let base_counters = vec![base; C2_N_THREADS];

        let t = Instant::now();
        let hits = ctx.dispatch_create2(
            deployer,
            initcodehash,
            &base_salts,
            &base_counters,
            C2_ITERS,
            threshold,
        );
        let batch_time = t.elapsed();

        base = base.wrapping_add(C2_ITERS as u64);
        total += C2_N_THREADS as u64 * C2_ITERS as u64;

        for h in &hits {
            if !ctx.verify_create2(deployer, initcodehash, &h.salt, &h.address) {
                eprintln!(
                    "FATAL: GPU CREATE2 hit failed host re-derivation: salt={} address={}",
                    hex::encode(h.salt),
                    hex::encode(h.address),
                );
                std::process::exit(1);
            }
            let zeros = h.zeros as usize;
            if zeros > best {
                best = zeros;
                let addr_str = format!("{:?}", Address::from_slice(&h.address));
                let notated = erc8117::format_address(&addr_str, erc8117::Mode::Subscript, false);
                let _ = writeln!(
                    file,
                    "{}\t{}\t{}\t0x{}",
                    total,
                    notated,
                    zeros,
                    hex::encode(h.salt)
                );
                println!(
                    "New best {} leading zeros: {}  salt=0x{}",
                    zeros,
                    erc8117::format_both(&addr_str, true),
                    hex::encode(h.salt),
                );
                if zeros >= target {
                    stop.store(true, Ordering::SeqCst);
                }
            }
        }

        if last_report.elapsed() >= Duration::from_secs(5) {
            let secs = start.elapsed().as_secs_f64().max(1.0);
            println!(
                "GPU: {:.2} Mkeys/s | best: {}",
                total as f64 / secs / 1e6,
                best
            );
            last_report = Instant::now();
        }

        // Hold ~UTILIZATION duty cycle (avoid pegging the GPU at 100%).
        std::thread::sleep(batch_time.mul_f64((1.0 - UTILIZATION) / UTILIZATION));
    }

    if best > 0 {
        println!("Done. Best: {best} leading-zero nibbles. Salts recorded in scanned_salts.txt.");
    } else {
        println!("Stopped before any qualifying salt was found.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_floors_at_gpu_floor_and_rises_with_best() {
        assert_eq!(c2_threshold(0, 12), C2_GPU_FLOOR as u32); // start at floor
        assert_eq!(c2_threshold(3, 12), C2_GPU_FLOOR as u32); // best below floor -> floor
        assert_eq!(c2_threshold(6, 12), 6); // best above floor -> best
    }

    #[test]
    fn threshold_clamps_to_target_when_target_below_floor() {
        // e.g. `nullforge-gpu 2 --create2 ...`: we must still surface >=2 hits
        // instead of flooring at C2_GPU_FLOOR (which would never match target 2).
        assert_eq!(c2_threshold(0, 2), 2);
        assert_eq!(c2_threshold(0, 1), 1);
    }

    #[test]
    fn threshold_never_drops_below_current_best() {
        // Monotonic: once the best has risen above the floor, the threshold
        // tracks it exactly (never regresses), so later batches don't re-report
        // hits the caller has already surfaced.
        assert_eq!(c2_threshold(5, 20), 5);
        assert_eq!(c2_threshold(10, 20), 10);
    }
}
