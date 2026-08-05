//! Tagged-CREATE2 salt-mining driver: dispatch keccak-only Metal batches
//! (three keccaks per candidate — `saltFor`-style tag wrap, `abi.encode` salt
//! mix, then CREATE2), verify every hit vs host keccak, report
//! strictly-increasing bests. The found value is a **tag** (public), so —
//! like the other salt miners and unlike the EOA miner — nothing is
//! encrypted; bests are appended in plaintext to `scanned_salts.txt`.
//!
//! This mines the caller-chosen "tag" input for factories that wrap it
//! through a stable per-deployer salt before CREATE2 (e.g. mev-arbitrum's
//! `MevSafeFactory.saltFor`), rather than a raw CREATE2 salt directly —
//! `create2.rs` cannot mine this shape because the effective salt is a
//! keccak digest of the tag, not the tag itself.
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

/// GPU threads per batch (one tag-stream each). Matches the other salt drivers.
pub const C2T_N_THREADS: usize = 65536;
/// Tags each thread scans per batch.
pub const C2T_ITERS: u32 = 256;
/// Dispatch-threshold floor (keeps a full ~16.7M-candidate batch under the
/// 1024-hit buffer cap).
const C2T_GPU_FLOOR: usize = 4;
/// GPU duty-cycle target (fraction of wall-clock spent computing).
const UTILIZATION: f64 = 0.80;

/// Threshold for the next batch: at least the floor, rising with the best; if
/// the target is below the floor, clamp to it so low targets still surface.
fn c2t_threshold(best_zeros: usize, target: usize) -> u32 {
    let t = best_zeros.max(C2T_GPU_FLOOR);
    let t = if target < C2T_GPU_FLOOR { target } else { t };
    t as u32
}

/// Mine tagged-CREATE2 tags for a fixed `prefix`/`deployer`/`owner`/
/// `permissions`/`factory`/`initcodehash` until `stop` is set or a tag
/// yielding `target` leading-zero nibbles is found. Verifies every GPU hit
/// against the three-step host keccak (a mismatch hard-aborts) and appends
/// each new best to `scanned_salts.txt` as
/// `<total>\t<address(erc8117)>\t<zeros>\t<tag_hex>`.
#[allow(clippy::too_many_arguments)]
pub fn run_create2tag(
    ctx: &MetalContext,
    prefix: &[u8],
    deployer: &[u8; 20],
    owner: &[u8; 20],
    permissions: &[u8; 20],
    factory: &[u8; 20],
    initcodehash: &[u8; 32],
    target: usize,
    stop: Arc<AtomicBool>,
) {
    // Random per-thread base tags: the high 24 bytes give each thread a
    // unique tag space; the low 8 bytes are overwritten by the per-iter/batch
    // counter.
    let mut rng = StdRng::from_entropy();
    let mut base_tags = vec![[0u8; 32]; C2T_N_THREADS];
    for s in base_tags.iter_mut() {
        rng.fill_bytes(s);
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open("scanned_salts.txt")
        .expect("Unable to open scanned_salts.txt");

    let mut best = 0usize;
    let mut base: u64 = 0;
    let mut total: u64 = 0;
    let start = Instant::now();
    let mut last_report = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let threshold = c2t_threshold(best, target);
        let base_counters = vec![base; C2T_N_THREADS];

        let t = Instant::now();
        let hits = ctx.dispatch_create2tag(
            prefix,
            deployer,
            owner,
            permissions,
            factory,
            initcodehash,
            &base_tags,
            &base_counters,
            C2T_ITERS,
            threshold,
        );
        let batch_time = t.elapsed();

        base = base.wrapping_add(C2T_ITERS as u64);
        total += C2T_N_THREADS as u64 * C2T_ITERS as u64;

        for h in &hits {
            if !ctx.verify_create2tag(
                prefix,
                deployer,
                owner,
                permissions,
                factory,
                initcodehash,
                &h.tag,
                &h.address,
            ) {
                eprintln!(
                    "FATAL: GPU tagged-CREATE2 hit failed host re-derivation: tag={} address={}",
                    hex::encode(h.tag),
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
                    hex::encode(h.tag)
                );
                println!(
                    "New best {} leading zeros: {}  tag=0x{}",
                    zeros,
                    erc8117::format_both(&addr_str, true),
                    hex::encode(h.tag),
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
        println!(
            "Done. Best: {best} leading-zero nibbles. Pass the recorded tag into the factory's \
             existing salt-tag input (e.g. `MEVSAFE_SALT_TAG`). Tags in scanned_salts.txt."
        );
    } else {
        println!("Stopped before any qualifying tag was found.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_floors_at_gpu_floor_and_rises_with_best() {
        assert_eq!(c2t_threshold(0, 12), C2T_GPU_FLOOR as u32);
        assert_eq!(c2t_threshold(3, 12), C2T_GPU_FLOOR as u32);
        assert_eq!(c2t_threshold(6, 12), 6);
    }

    #[test]
    fn threshold_clamps_to_target_when_target_below_floor() {
        assert_eq!(c2t_threshold(0, 2), 2);
        assert_eq!(c2t_threshold(0, 1), 1);
    }

    #[test]
    fn threshold_never_drops_below_current_best() {
        assert_eq!(c2t_threshold(5, 20), 5);
        assert_eq!(c2t_threshold(10, 20), 10);
    }
}
