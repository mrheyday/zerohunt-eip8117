//! CreateX permissionless CREATE3 salt-mining driver.
//!
//! Algorithm locked to upstream CreateX:
//! <https://github.com/pcaversaccio/createx> (`src/CreateX.sol`).
//!
//! Mirrors `create3`, but the deployer is a CreateX factory, the salt is
//! guarded (`guardedSalt = keccak256(salt)` for the permissionless
//! `SenderBytes.Random` branch — see `CreateX._guard`) inside the kernel, and
//! the emitted value is the ORIGINAL salt you pass to
//! `CreateX.deployCreate3(salt, initCode)`.
//!
//! Default factory = Nick's-method canonical
//! `0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed` (identical on every chain that
//! ran the pre-signed deploy). Override with `CREATEX_FACTORY` or
//! `CREATEX_ADDRESS` (20-byte hex) for a project-local CreateX redeploy — the
//! CREATE3 math is the same; only the factory identity changes.
//!
//! Salt is public → plaintext `scanned_salts.txt`.
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
use crate::gpu::{MetalContext, CREATEX_ADDRESS, STANDARD_CREATE3_PROXY_HASH};

pub const CX_N_THREADS: usize = 65536;
pub const CX_ITERS: u32 = 256;
const CX_GPU_FLOOR: usize = 4;
const UTILIZATION: f64 = 0.80;

/// Upstream CreateX repository (algorithm + proxyChildBytecode source of truth).
pub const CREATEX_UPSTREAM: &str = "https://github.com/pcaversaccio/createx";

fn cx_threshold(best_zeros: usize, target: usize) -> u32 {
    let t = best_zeros.max(CX_GPU_FLOOR);
    let t = if target < CX_GPU_FLOOR { target } else { t };
    t as u32
}

/// Resolve factory: `CREATEX_FACTORY` → `CREATEX_ADDRESS` → canonical constant.
fn resolve_createx_factory() -> [u8; 20] {
    let raw = std::env::var("CREATEX_FACTORY")
        .or_else(|_| std::env::var("CREATEX_ADDRESS"))
        .ok();
    match raw {
        Some(s) => {
            let hexstr = s.trim().trim_start_matches("0x");
            assert_eq!(
                hexstr.len(),
                40,
                "CREATEX_FACTORY/CREATEX_ADDRESS must be 20 bytes hex"
            );
            let mut out = [0u8; 20];
            for i in 0..20 {
                out[i] = u8::from_str_radix(&hexstr[i * 2..i * 2 + 2], 16)
                    .expect("CREATEX_FACTORY/CREATEX_ADDRESS hex");
            }
            eprintln!(
                "nullforge createx: factory override 0x{} (upstream {})",
                hex::encode(out),
                CREATEX_UPSTREAM
            );
            out
        }
        None => CREATEX_ADDRESS,
    }
}

/// Mine CreateX permissionless CREATE3 salts until `stop` is set or a salt
/// yielding `target` leading-zero nibbles is found. Proxy hash is the fixed
/// CreateX `proxyChildBytecode` hash (Solmate/0xSequence-compatible). Factory
/// defaults to the canonical address; override via env. Verifies every GPU hit
/// against the host guard + two-step keccak (mismatch hard-aborts); appends
/// each new best to `scanned_salts.txt` as
/// `<total>\t<address(erc8117)>\t<zeros>\t<salt_hex>`.
pub fn run_createx(ctx: &MetalContext, target: usize, stop: Arc<AtomicBool>) {
    let createx = resolve_createx_factory();
    let proxyhash = STANDARD_CREATE3_PROXY_HASH;

    let mut rng = StdRng::from_entropy();
    let mut base_salts = vec![[0u8; 32]; CX_N_THREADS];
    for s in base_salts.iter_mut() {
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
        let threshold = cx_threshold(best, target);
        let base_counters = vec![base; CX_N_THREADS];

        let t = Instant::now();
        let hits = ctx.dispatch_createx(
            &createx,
            &proxyhash,
            &base_salts,
            &base_counters,
            CX_ITERS,
            threshold,
        );
        let batch_time = t.elapsed();

        base = base.wrapping_add(CX_ITERS as u64);
        total += CX_N_THREADS as u64 * CX_ITERS as u64;

        for h in &hits {
            if !ctx.verify_createx(&createx, &proxyhash, &h.salt, &h.address) {
                eprintln!(
                    "FATAL: GPU CreateX hit failed host re-derivation: salt={} address={}",
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
            println!("GPU: {:.2} Mkeys/s | best: {}", total as f64 / secs / 1e6, best);
            last_report = Instant::now();
        }

        std::thread::sleep(batch_time.mul_f64((1.0 - UTILIZATION) / UTILIZATION));
    }

    if best > 0 {
        println!(
            "Done. Best: {best} leading-zero nibbles. Pass the recorded salt to \
             CreateX.deployCreate3(salt, initCode). Salts in scanned_salts.txt."
        );
    } else {
        println!("Stopped before any qualifying salt was found.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_behaves_like_create3() {
        assert_eq!(cx_threshold(0, 12), CX_GPU_FLOOR as u32);
        assert_eq!(cx_threshold(0, 2), 2);
        assert_eq!(cx_threshold(7, 20), 7);
    }
}
