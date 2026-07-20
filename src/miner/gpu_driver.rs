//! GPU driver: dispatch Metal incremental-EC mine batches ("Approach B" --
//! docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md), verify every
//! hit vs k256, report.
use std::sync::Arc;
use std::time::Instant;

use ethers::types::Address;
use ethers::utils::hex;

use crate::gpu::{Hit, MetalContext};
use crate::miner::cpu::leading_zero_nibbles;
use crate::miner::shared::{Engine, MinerShared};

/// One thread per seed; each thread scans `ITERS` consecutive counters.
pub const N_THREADS: usize = 65536;
pub const ITERS: u32 = 256;
/// Dispatch-threshold floor: keeps a full ~16.7M-candidate batch under the
/// kernel's 1024-hit buffer cap (16.7M / 16^4 ≈ 256 << 1024).
pub const GPU_FLOOR: usize = 4;
/// GPU duty-cycle target (fraction of wall-clock spent computing).
const UTILIZATION: f64 = 0.80;

/// Threshold for the next batch: at least `GPU_FLOOR`, rising with the current
/// best so higher-zero targets don't flood the buffer. If `target` is below the
/// floor (e.g. `target=2`), clamp down so we still surface qualifying hits.
pub fn gpu_threshold(best_zeros: usize, target: usize) -> u32 {
    let t = best_zeros.max(GPU_FLOOR);
    let t = if target < GPU_FLOOR { target } else { t };
    t as u32
}

/// Re-derive the hit's address from its private key via k256. `Ok(zeros)` with
/// the host-recomputed leading-zero count on a match; `Err(msg)` on mismatch —
/// the caller MUST treat `Err` as fatal (a GPU bug must never emit a bad key).
pub fn verify_hit_or_err(ctx: &MetalContext, hit: &Hit) -> Result<usize, String> {
    if !ctx.verify_hit(hit.privkey, hit.address) {
        return Err(format!(
            "GPU hit failed k256 re-derivation: privkey={} claimed_address={}",
            hex::encode(hit.privkey),
            hex::encode(hit.address),
        ));
    }
    Ok(leading_zero_nibbles(&hit.address))
}

/// Dispatch mine batches until `shared.should_stop()`. Verifies every hit; on a
/// verified hit that strictly beats the best, reports it through the funnel.
/// Sleeps after each batch to hold the GPU at ~`UTILIZATION` duty cycle.
pub fn run_batches(ctx: &MetalContext, shared: Arc<MinerShared>, seeds: &[[u8; 32]]) {
    let n = seeds.len();
    let mut base: u64 = 0;

    while !shared.should_stop() {
        let threshold = gpu_threshold(shared.best_zeros(), shared.target);
        let base_counters = vec![base; n];

        let t = Instant::now();
        let hits = ctx.dispatch_mine_incremental(seeds, &base_counters, ITERS, threshold);
        let batch_time = t.elapsed();

        base = base.wrapping_add(ITERS as u64);
        shared.add_keys(Engine::Gpu, n as u64 * ITERS as u64);

        for hit in &hits {
            match verify_hit_or_err(ctx, hit) {
                Ok(zeros) => {
                    if zeros > shared.best_zeros() {
                        let address_str = format!("{:?}", Address::from_slice(&hit.address));
                        shared.report_hit(Engine::Gpu, hit.privkey, &address_str, zeros);
                    }
                }
                Err(msg) => {
                    eprintln!("FATAL: {msg}");
                    std::process::exit(1);
                }
            }
        }

        // Hold ~UTILIZATION duty cycle: idle (1-U)/U of the compute time.
        let idle = batch_time.mul_f64((1.0 - UTILIZATION) / UTILIZATION);
        std::thread::sleep(idle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_floors_at_gpu_floor_and_rises_with_best() {
        assert_eq!(gpu_threshold(0, 8), GPU_FLOOR as u32); // start at floor
        assert_eq!(gpu_threshold(3, 8), GPU_FLOOR as u32); // best below floor -> floor
        assert_eq!(gpu_threshold(6, 8), 6); // best above floor -> best
    }

    #[test]
    fn threshold_clamps_to_target_when_target_below_floor() {
        // e.g. `nullforge-gpu 2`: we must still surface >=2 hits
        assert_eq!(gpu_threshold(0, 2), 2);
    }

    #[test]
    fn threshold_at_target_exactly_equal_to_floor_stays_at_floor() {
        // Boundary: `target < GPU_FLOOR` clamps down, but `target == GPU_FLOOR`
        // must NOT clamp (the `<` comparison is strict) -- the floor itself is
        // still a valid, un-clamped threshold.
        assert_eq!(gpu_threshold(0, GPU_FLOOR), GPU_FLOOR as u32);
    }
}
