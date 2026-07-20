//! CREATE3 salt-mining driver: dispatch keccak-only Metal batches, verify every
//! hit vs the host two-hop derivation (CREATE2 of a fixed proxy -> proxy CREATE
//! at nonce 1), report strictly-increasing bests. The found value is a **salt**
//! (public), so — unlike the EOA miner — nothing is encrypted; bests are appended
//! in plaintext to `scanned_salts.txt`.
//!
//! Unlike CREATE2 mining, the mined salt is **init-code-independent**: the same
//! salt lands any contract deployed through the given factory on the vanity
//! address, because CREATE3 derives the address from `(factory, salt)` alone.
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

/// 0xSequence/Solady CREATE3 proxy init code: the constant bytecode the factory
/// CREATE2-deploys at the first hop. Its keccak256 is `DEFAULT_CREATE3_PROXY_HASH`.
pub const CREATE3_PROXY_INIT_CODE: [u8; 16] = [
    0x67, 0x36, 0x3d, 0x3d, 0x37, 0x36, 0x3d, 0x34, 0xf0, 0x3d, 0x52, 0x60, 0x08, 0x60, 0x18, 0xf3,
];

/// Default CREATE3 proxy-init-code hash (`keccak256(CREATE3_PROXY_INIT_CODE)`),
/// used by the 0xSequence/Solady CREATE3 factory family. Override via `--proxy-hash`
/// for a factory that deploys a different proxy — a wrong hash silently mines
/// salts for the wrong address.
pub const DEFAULT_CREATE3_PROXY_HASH: [u8; 32] = [
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f,
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f,
];

/// GPU threads per batch (one salt-stream each). Matches the CREATE2 driver's grid.
pub const C3_N_THREADS: usize = 65536;
/// Salts each thread scans per batch.
pub const C3_ITERS: u32 = 256;
/// Dispatch-threshold floor (keeps a full ~16.7M-candidate batch under the
/// 1024-hit buffer cap).
const C3_GPU_FLOOR: usize = 4;
/// GPU duty-cycle target (fraction of wall-clock spent computing).
const UTILIZATION: f64 = 0.80;

/// Threshold for the next batch: at least the floor, rising with the best; if the
/// target is below the floor, clamp to it so low targets still surface.
fn c3_threshold(best_zeros: usize, target: usize) -> u32 {
    let t = best_zeros.max(C3_GPU_FLOOR);
    let t = if target < C3_GPU_FLOOR { target } else { t };
    t as u32
}

/// Mine CREATE3 salts for `factory` + `proxy_hash` until `stop` is set or a salt
/// yielding `target` leading-zero nibbles is found. Verifies every GPU hit against
/// the host two-hop keccak (a mismatch hard-aborts) and appends each new best to
/// `scanned_salts.txt` as `<total>\t<address(erc8117)>\t<zeros>\t<salt_hex>`.
pub fn run_create3(
    ctx: &MetalContext,
    factory: &[u8; 20],
    proxy_hash: &[u8; 32],
    target: usize,
    stop: Arc<AtomicBool>,
) {
    // Random per-thread base salts: the high 24 bytes give each thread a unique
    // salt space; the low 8 bytes are overwritten by the per-iter/batch counter.
    let mut rng = StdRng::from_entropy();
    let mut base_salts = vec![[0u8; 32]; C3_N_THREADS];
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
        let threshold = c3_threshold(best, target);
        let base_counters = vec![base; C3_N_THREADS];

        let t = Instant::now();
        let hits = ctx.dispatch_create3(
            factory,
            proxy_hash,
            &base_salts,
            &base_counters,
            C3_ITERS,
            threshold,
        );
        let batch_time = t.elapsed();

        base = base.wrapping_add(C3_ITERS as u64);
        total += C3_N_THREADS as u64 * C3_ITERS as u64;

        for h in &hits {
            if !ctx.verify_create3(factory, proxy_hash, &h.salt, &h.address) {
                eprintln!(
                    "FATAL: GPU CREATE3 hit failed host re-derivation: salt={} address={}",
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
    use crate::gpu::create3_deployed_address;
    use ethers::utils::keccak256;

    // Independently-derived CREATE3 vector (cross-checked with sha3::Keccak256):
    // factory = 0x00..0042, salt = [0u8; 32], default (0xSequence/Solady) proxy hash
    //   -> proxy    = 0xeb6de9107e8fa627ad24c24adc66ebf795d0c183
    //   -> deployed = 0xb284f24ec1f008f26fb34d81207c7a8d68751645
    const V_FACTORY: [u8; 20] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x42,
    ];
    const V_DEPLOYED: [u8; 20] = [
        0xb2, 0x84, 0xf2, 0x4e, 0xc1, 0xf0, 0x08, 0xf2, 0x6f, 0xb3, 0x4d, 0x81, 0x20, 0x7c, 0x7a,
        0x8d, 0x68, 0x75, 0x16, 0x45,
    ];

    #[test]
    fn default_proxy_hash_is_keccak_of_proxy_init_code() {
        // Guards the hardcoded constant: it must be keccak256 of the proxy bytecode.
        assert_eq!(keccak256(CREATE3_PROXY_INIT_CODE), DEFAULT_CREATE3_PROXY_HASH);
    }

    #[test]
    fn deployed_address_matches_pinned_vector() {
        let salt = [0u8; 32];
        let got = create3_deployed_address(&V_FACTORY, &DEFAULT_CREATE3_PROXY_HASH, &salt);
        assert_eq!(got, V_DEPLOYED, "CREATE3 two-hop derivation regressed");
    }

    #[test]
    fn wrong_proxy_hash_changes_the_address() {
        // A different proxy hash must resolve to a different deployed address —
        // regression guard against the fixed proxy constant being dropped/ignored.
        let salt = [0u8; 32];
        let mut wrong = DEFAULT_CREATE3_PROXY_HASH;
        wrong[0] ^= 0x01;
        let got = create3_deployed_address(&V_FACTORY, &wrong, &salt);
        assert_ne!(got, V_DEPLOYED);
    }

    #[test]
    fn c3_threshold_floors_at_gpu_floor_and_rises_with_best() {
        assert_eq!(c3_threshold(0, 12), C3_GPU_FLOOR as u32); // start at floor
        assert_eq!(c3_threshold(3, 12), C3_GPU_FLOOR as u32); // best below floor -> floor
        assert_eq!(c3_threshold(6, 12), 6); // best above floor -> best
    }

    #[test]
    fn c3_threshold_clamps_to_target_when_target_below_floor() {
        // e.g. `nullforge-gpu 2 --create3 ...`: still surface >=2 hits.
        assert_eq!(c3_threshold(0, 2), 2);
        assert_eq!(c3_threshold(0, 1), 1);
    }
}
