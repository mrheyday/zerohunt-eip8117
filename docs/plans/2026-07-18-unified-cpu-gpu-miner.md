# Unified CPU+GPU Miner (Stage 1: Integration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One process (`nullforge-gpu`) that mines Ethereum addresses on the CPU and GPU together against a single shared best-tracker, streaming strictly-increasing "new best" leading-zero records in ERC-8117 notation, stopping at `target_zeros`.

**Architecture:** A `nullforge::miner` library module holds the shared state + reporting funnel (`report_hit`), the CPU worker loop, and the GPU driver loop; each is unit-testable in isolation. `src/bin/gpu.rs` is thin wiring: parse args, build shared state, spawn `~80%·num_cpus` CPU workers + one GPU-driver thread + a rate reporter + a Ctrl-C handler, join, print the final summary. Every GPU hit is re-verified against `k256` (hard-abort on mismatch). This is Stage 1 of the spec; it runs on the *current* `mine` kernel and does not change any `.metal` code.

**Tech Stack:** Rust, `tokio` (rt-multi-thread, signal), `metal` (Apple Metal compute), `ethers`/`k256` (secp256k1 + address + verification), existing `nullforge::erc8117` and `nullforge::gpu` modules.

## Global Constraints

- Crate name is `nullforge`; the library exposes modules as `nullforge::<mod>`. Binary targets: `nullforge` (CPU-only, unchanged) and `nullforge-gpu` (this tool).
- `rand` is pinned to `0.8` (k256 0.13 via ethers 2.0.14 bounds `SigningKey::random` on rand_core 0.6). Use `rand::rngs::StdRng` + `SeedableRng::from_entropy` (matches `src/main.rs`).
- ERC-8117 rendering MUST go through `nullforge::erc8117`: console = `format_both(addr, true)` (both modes, truncated); file column = `format_address(addr, Mode::Subscript, false)` (subscript, non-truncated, lossless).
- Address string rendering MUST be `format!("{:?}", address)` on an `ethers::types::Address` (lowercase 0x-hex, 42 chars) — identical to `src/main.rs`, so the two tools' leading-zero counts and output match.
- Leading-zero-nibble count uses the existing byte semantics: full-zero byte → +2, first non-zero byte → `+ (byte.leading_zeros()/4)`, then stop.
- Every GPU-reported hit MUST be re-derived on the host via `MetalContext::verify_hit` and matched before it is trusted; any mismatch is a hard abort (`std::process::exit(1)`). A GPU bug may waste time but must never emit a bad key.
- CLI mirrors the CPU tool: `nullforge-gpu [target_zeros]`, default `8`; non-numeric arg → usage message + `exit(2)`.
- `UTILIZATION = 0.80`: CPU workers = `max(1, round(num_cpus as f64 * 0.80))`; GPU duty cycle ≈ 80% via post-batch sleep of `(1-0.80)/0.80 · batch_time`.

## Prerequisites (MUST be satisfied before Task 1)

The `mine` kernel (`kernels/miner.metal`), `MetalContext::dispatch_mine`, and `struct Hit { privkey:[u8;32], address:[u8;20], zeros:u8 }` (`src/gpu/mod.rs`) are **uncommitted WIP** in the main checkout and are NOT on the branch this plan was drafted against. Before starting:

- [ ] Confirm `src/gpu/mod.rs` exposes `pub fn dispatch_mine(&self, seeds:&[[u8;32]], base_counters:&[u64], iters:u32, threshold:u32) -> Vec<Hit>` and `pub struct Hit`, and `kernels/miner.metal` contains `kernel void mine(...)`. If absent, land that WIP (commit it) and base this work on it. Verify with: `cargo test --test gpu_stages` (the WIP includes a `dispatch_mine` e2e test that must pass on the target machine's Metal device).

## File Structure

- Create `src/miner/mod.rs` — declares `pub mod shared; pub mod cpu; pub mod gpu_driver;`
- Create `src/miner/shared.rs` — `MinerShared`, `Engine`, `FoundKey`, `report_hit` (the funnel). No threads, no GPU. Fully unit-tested.
- Create `src/miner/cpu.rs` — `cpu_worker(Arc<MinerShared>)` loop + `leading_zero_nibbles(&[u8]) -> usize` (pure, tested).
- Create `src/miner/gpu_driver.rs` — `run_batches(&MetalContext, Arc<MinerShared>, &[[u8;32]])` + `verify_hit_or_err(&MetalContext, &Hit) -> Result<usize,String>` (returns so the abort path is testable) + `N_THREADS`, `ITERS`, `GPU_FLOOR`.
- Modify `src/lib.rs` — add `pub mod miner;`
- Modify `src/bin/gpu.rs` — replace the placeholder with the full host wiring.
- Modify `Cargo.toml` — add `tempfile` as a `[dev-dependencies]` entry (for file-output unit tests).
- Test: unit tests live inline (`#[cfg(test)] mod tests`) in `shared.rs`, `cpu.rs`, `gpu_driver.rs`; a GPU-gated e2e test is added to `tests/gpu_stages.rs`.

---

### Task 1: `miner::shared` — state + `report_hit` funnel

**Files:**
- Create: `src/miner/mod.rs`
- Create: `src/miner/shared.rs`
- Modify: `src/lib.rs` (add `pub mod miner;`)
- Modify: `Cargo.toml` (add `tempfile` dev-dependency)
- Test: inline `#[cfg(test)] mod tests` in `src/miner/shared.rs`

**Interfaces:**
- Consumes: `nullforge::erc8117::{format_address, format_both, Mode}`.
- Produces:
  - `enum Engine { Cpu, Gpu }` with `fn label(self) -> &'static str`
  - `struct FoundKey { pub privkey:[u8;32], pub address_str:String, pub zeros:usize }` (derives `Clone`)
  - `struct MinerShared` with `pub target: usize`, and:
    - `fn new(target: usize, file: std::fs::File, start: std::time::Instant) -> Self`
    - `fn should_stop(&self) -> bool` / `fn request_stop(&self)`
    - `fn best_zeros(&self) -> usize`
    - `fn add_keys(&self, engine: Engine, n: u64)` / `fn cpu_keys(&self) -> u64` / `fn gpu_keys(&self) -> u64`
    - `fn report_hit(&self, engine: Engine, privkey:[u8;32], address_str:&str, zeros:usize) -> bool`
    - `fn take_best(&self) -> Option<FoundKey>`

- [ ] **Step 1: Add the `tempfile` dev-dependency**

In `Cargo.toml`, add after the `[dependencies]` block:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Create the module tree**

Create `src/miner/mod.rs`:

```rust
pub mod shared;
pub mod cpu;
pub mod gpu_driver;
```

Add to `src/lib.rs` (keep existing lines):

```rust
pub mod erc8117;
pub mod gpu;
pub mod miner;
```

Note: `cpu` and `gpu_driver` are created in later tasks; to compile Task 1 alone, temporarily comment out `pub mod cpu;` and `pub mod gpu_driver;` in `mod.rs` and re-enable them in Tasks 2 and 3. (State this in the commit messages.)

- [ ] **Step 3: Write the failing tests**

Create `src/miner/shared.rs` with only the test module first (the types don't exist yet, so it fails to compile — that is the failing state):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::time::Instant;

    fn ctx(target: usize) -> (MinerShared, tempfile::NamedTempFile) {
        let f = tempfile::NamedTempFile::new().unwrap();
        let shared = MinerShared::new(target, f.reopen().unwrap(), Instant::now());
        (shared, f)
    }

    fn read_file(f: &tempfile::NamedTempFile) -> String {
        let mut s = String::new();
        f.reopen().unwrap().read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn first_qualifying_hit_becomes_best_and_writes_file() {
        let (shared, f) = ctx(8);
        let addr = "0x00000000abcd0123456789012345678901234567"; // 8 zeros
        let became = shared.report_hit(Engine::Gpu, [0u8; 32], addr, 8);
        assert!(became);
        assert_eq!(shared.best_zeros(), 8);
        // file column is subscript non-truncated (0x0₈ + full remainder)
        let line = read_file(&f);
        assert!(line.contains("0x0\u{2088}abcd0123456789012345678901234567"), "got: {line}");
        assert!(line.contains("\t8\t"), "zeros column, got: {line}");
    }

    #[test]
    fn strictly_greater_gating_ignores_equal_or_lower() {
        let (shared, _f) = ctx(8);
        let a = "0x00000000abcd0123456789012345678901234567"; // 8
        assert!(shared.report_hit(Engine::Cpu, [0u8; 32], a, 8));
        // equal zeros -> not a new best
        assert!(!shared.report_hit(Engine::Gpu, [1u8; 32], a, 8));
        // lower zeros -> not a new best
        let b = "0x0000abcd012345678901234567890123456789ab"; // 4
        assert!(!shared.report_hit(Engine::Cpu, [2u8; 32], b, 4));
        assert_eq!(shared.best_zeros(), 8);
    }

    #[test]
    fn stop_trips_exactly_at_target() {
        let (shared, _f) = ctx(6);
        assert!(shared.report_hit(Engine::Gpu, [0u8; 32], "0x00000abc0123456789012345678901234567890a", 5));
        assert!(!shared.should_stop(), "5 < target 6");
        assert!(shared.report_hit(Engine::Gpu, [0u8; 32], "0x000000abc123456789012345678901234567890a", 6));
        assert!(shared.should_stop(), "6 >= target 6");
    }

    #[test]
    fn take_best_returns_latest() {
        let (shared, _f) = ctx(8);
        shared.report_hit(Engine::Cpu, [7u8; 32], "0x0000abc0123456789012345678901234567890ab", 4);
        let best = shared.take_best().unwrap();
        assert_eq!(best.zeros, 4);
        assert_eq!(best.privkey, [7u8; 32]);
    }

    #[test]
    fn key_counters_are_per_engine() {
        let (shared, _f) = ctx(8);
        shared.add_keys(Engine::Cpu, 100);
        shared.add_keys(Engine::Gpu, 250);
        shared.add_keys(Engine::Cpu, 5);
        assert_eq!(shared.cpu_keys(), 105);
        assert_eq!(shared.gpu_keys(), 250);
    }
}
```

- [ ] **Step 4: Run the tests to verify they fail (don't compile)**

Run: `cargo test --lib miner::shared`
Expected: FAIL — `cannot find type MinerShared`/`Engine` in this scope.

- [ ] **Step 5: Implement the module (above the test module)**

Prepend to `src/miner/shared.rs`:

```rust
//! Shared state + the single reporting funnel both mining engines call.
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ethers::utils::hex;

use crate::erc8117;

/// Which engine found a candidate (for attribution + rate accounting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Cpu,
    Gpu,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Cpu => "CPU",
            Engine::Gpu => "GPU",
        }
    }
}

/// The current best find: a private key whose address has `zeros` leading zero
/// nibbles, plus the rendered address string used for display.
#[derive(Debug, Clone)]
pub struct FoundKey {
    pub privkey: [u8; 32],
    pub address_str: String,
    pub zeros: usize,
}

/// State shared by every CPU worker, the GPU driver, and the rate reporter.
pub struct MinerShared {
    pub target: usize,
    best_zeros: AtomicUsize,
    best: Mutex<Option<FoundKey>>,
    file: Mutex<File>,
    cpu_keys: AtomicU64,
    gpu_keys: AtomicU64,
    stop: AtomicBool,
    #[allow(dead_code)] // used by the rate reporter in the binary
    start: Instant,
}

impl MinerShared {
    pub fn new(target: usize, file: File, start: Instant) -> Self {
        Self {
            target,
            best_zeros: AtomicUsize::new(0),
            best: Mutex::new(None),
            file: Mutex::new(file),
            cpu_keys: AtomicU64::new(0),
            gpu_keys: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            start,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn best_zeros(&self) -> usize {
        self.best_zeros.load(Ordering::Relaxed)
    }

    pub fn add_keys(&self, engine: Engine, n: u64) {
        match engine {
            Engine::Cpu => self.cpu_keys.fetch_add(n, Ordering::Relaxed),
            Engine::Gpu => self.gpu_keys.fetch_add(n, Ordering::Relaxed),
        };
    }

    pub fn cpu_keys(&self) -> u64 {
        self.cpu_keys.load(Ordering::Relaxed)
    }

    pub fn gpu_keys(&self) -> u64 {
        self.gpu_keys.load(Ordering::Relaxed)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// The single reporting funnel. Returns `true` iff this became a new best
    /// (strictly more leading zeros than any prior). On a new best it writes the
    /// `scanned_keys.txt` line (ERC-8117 subscript, non-truncated), prints the
    /// "new best" line (ERC-8117 both modes, truncated), and requests stop when
    /// `zeros >= target`.
    pub fn report_hit(
        &self,
        engine: Engine,
        privkey: [u8; 32],
        address_str: &str,
        zeros: usize,
    ) -> bool {
        // Fast path: no lock unless this strictly beats the current best.
        if zeros <= self.best_zeros.load(Ordering::Relaxed) {
            return false;
        }
        let mut best = self.best.lock().unwrap();
        if zeros <= self.best_zeros.load(Ordering::Relaxed) {
            return false; // lost the race to another thread
        }
        self.best_zeros.store(zeros, Ordering::SeqCst);
        *best = Some(FoundKey {
            privkey,
            address_str: address_str.to_string(),
            zeros,
        });

        let total = self.cpu_keys() + self.gpu_keys();
        let notated = erc8117::format_address(address_str, erc8117::Mode::Subscript, false);
        let privhex = hex::encode(privkey);
        {
            let mut file = self.file.lock().unwrap();
            let _ = writeln!(file, "{}\t{}\t{}\t{}", total, notated, zeros, privhex);
        }
        println!(
            "New best [{}] {} leading zeros: {}",
            engine.label(),
            zeros,
            erc8117::format_both(address_str, true)
        );
        if zeros >= self.target {
            self.request_stop();
        }
        true
    }

    pub fn take_best(&self) -> Option<FoundKey> {
        self.best.lock().unwrap().clone()
    }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib miner::shared`
Expected: PASS — 5 tests.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/lib.rs src/miner/mod.rs src/miner/shared.rs
git commit -m "feat(miner): shared state + report_hit funnel (ERC-8117, strictly-increasing best)"
```

---

### Task 2: `miner::cpu` — worker loop + zero-count helper

**Files:**
- Create: `src/miner/cpu.rs`
- Modify: `src/miner/mod.rs` (uncomment `pub mod cpu;`)
- Test: inline `#[cfg(test)] mod tests` in `src/miner/cpu.rs`

**Interfaces:**
- Consumes: `MinerShared`, `Engine` from `crate::miner::shared`; `ethers::utils::secret_key_to_address`; `ethers::core::k256::ecdsa::SigningKey`.
- Produces:
  - `fn leading_zero_nibbles(addr: &[u8]) -> usize`
  - `fn cpu_worker(shared: std::sync::Arc<MinerShared>)`

- [ ] **Step 1: Write the failing test** (for the pure helper — the loop is exercised by the e2e test in Task 5)

Create `src/miner/cpu.rs` with just the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_leading_zero_nibbles_like_main() {
        // full zero bytes -> +2 each; first non-zero byte -> +(leading_zeros/4)
        assert_eq!(leading_zero_nibbles(&[0x00, 0x00, 0x0a, 0xff]), 5); // 2+2+1
        assert_eq!(leading_zero_nibbles(&[0x0a, 0xff]), 1);
        assert_eq!(leading_zero_nibbles(&[0xff, 0x00]), 0);
        assert_eq!(leading_zero_nibbles(&[0x00, 0x00]), 4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib miner::cpu`
Expected: FAIL — `cannot find function leading_zero_nibbles`.

- [ ] **Step 3: Implement** (prepend to `src/miner/cpu.rs`)

```rust
//! CPU mining worker: random full-entropy keys -> address -> report.
use std::sync::Arc;

use ethers::core::k256::ecdsa::SigningKey;
use ethers::utils::secret_key_to_address;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::miner::shared::{Engine, MinerShared};

/// Flush the local key counter to the shared atomic every this many keys, to
/// keep the hot loop off the shared cache line (mirrors `src/main.rs`).
const COUNTER_FLUSH: u64 = 4096;

/// Count leading zero nibbles of an address by raw bytes: a fully-zero byte
/// contributes 2, the first non-zero byte contributes `leading_zeros()/4`
/// (1 if its top nibble is zero, else 0), then stop.
pub fn leading_zero_nibbles(addr: &[u8]) -> usize {
    let mut zeros = 0usize;
    for &byte in addr {
        if byte == 0 {
            zeros += 2;
        } else {
            zeros += (byte.leading_zeros() / 4) as usize;
            break;
        }
    }
    zeros
}

/// Mine random keys until `shared.should_stop()`. Reports any candidate that
/// strictly beats the current best through the shared funnel.
pub fn cpu_worker(shared: Arc<MinerShared>) {
    let mut rng = StdRng::from_entropy();
    let mut local: u64 = 0;

    loop {
        if shared.should_stop() {
            break;
        }
        let signer = SigningKey::random(&mut rng);
        let address = secret_key_to_address(&signer);
        let zeros = leading_zero_nibbles(address.as_bytes());

        local += 1;
        if local >= COUNTER_FLUSH {
            shared.add_keys(Engine::Cpu, local);
            local = 0;
        }

        if zeros > shared.best_zeros() {
            let address_str = format!("{:?}", address);
            let privkey: [u8; 32] = signer.to_bytes().into();
            shared.report_hit(Engine::Cpu, privkey, &address_str, zeros);
        }
    }
    shared.add_keys(Engine::Cpu, local);
}
```

In `src/miner/mod.rs`, ensure `pub mod cpu;` is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib miner::cpu`
Expected: PASS — 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/miner/cpu.rs src/miner/mod.rs
git commit -m "feat(miner): CPU worker loop + leading-zero-nibble helper"
```

---

### Task 3: `miner::gpu_driver` — batch loop + verify/abort gate

**Files:**
- Create: `src/miner/gpu_driver.rs`
- Modify: `src/miner/mod.rs` (uncomment `pub mod gpu_driver;`)
- Test: inline `#[cfg(test)] mod tests` in `src/miner/gpu_driver.rs` (no GPU needed — tests the verify decision logic only)

**Interfaces:**
- Consumes: `crate::gpu::{MetalContext, Hit}`; `crate::miner::shared::{Engine, MinerShared}`; `crate::miner::cpu::leading_zero_nibbles`; `ethers::types::Address`.
- Produces:
  - `const N_THREADS: usize = 65536; const ITERS: u32 = 256; const GPU_FLOOR: usize = 4;`
  - `fn verify_hit_or_err(ctx: &MetalContext, hit: &Hit) -> Result<usize, String>`
  - `fn gpu_threshold(best_zeros: usize, target: usize) -> u32`
  - `fn run_batches(ctx: &MetalContext, shared: std::sync::Arc<MinerShared>, seeds: &[[u8; 32]])`

- [ ] **Step 1: Write the failing tests** (verify decision + threshold policy — both pure)

Create `src/miner/gpu_driver.rs` with just the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_floors_at_gpu_floor_and_rises_with_best() {
        assert_eq!(gpu_threshold(0, 8), GPU_FLOOR as u32); // start at floor
        assert_eq!(gpu_threshold(3, 8), GPU_FLOOR as u32); // best below floor -> floor
        assert_eq!(gpu_threshold(6, 8), 6);                // best above floor -> best
    }

    #[test]
    fn threshold_clamps_to_target_when_target_below_floor() {
        // e.g. `nullforge-gpu 2`: we must still surface >=2 hits
        assert_eq!(gpu_threshold(0, 2), 2);
    }
}
```

Note: the `verify_hit_or_err` Err path and `run_batches` are covered by the GPU-gated e2e test in Task 5 (they need a real `MetalContext`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib miner::gpu_driver`
Expected: FAIL — `cannot find function gpu_threshold`.

- [ ] **Step 3: Implement** (prepend to `src/miner/gpu_driver.rs`)

```rust
//! GPU driver: dispatch Metal mine batches, verify every hit vs k256, report.
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
        let hits = ctx.dispatch_mine(seeds, &base_counters, ITERS, threshold);
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
```

In `src/miner/mod.rs`, ensure `pub mod gpu_driver;` is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib miner::gpu_driver`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/miner/gpu_driver.rs src/miner/mod.rs
git commit -m "feat(miner): GPU driver batch loop + verify-or-abort gate + threshold policy"
```

---

### Task 4: `src/bin/gpu.rs` — host wiring (the working tool)

**Files:**
- Modify: `src/bin/gpu.rs` (replace the placeholder entirely)

**Interfaces:**
- Consumes: `nullforge::miner::shared::{Engine, MinerShared}`, `nullforge::miner::cpu::cpu_worker`, `nullforge::miner::gpu_driver::{run_batches, N_THREADS}`, `nullforge::gpu::MetalContext`, `nullforge::erc8117`.
- Produces: the `nullforge-gpu` binary. No new library symbols.

- [ ] **Step 1: Replace `src/bin/gpu.rs`**

```rust
//! nullforge-gpu: unified CPU+GPU vanity miner. Mines addresses with the most
//! leading zero nibbles across CPU workers and the Metal GPU against one shared
//! best-tracker, streaming strictly-increasing "new best" records in ERC-8117
//! notation. Stops at `target_zeros` (default 8) or Ctrl-C.
use std::env;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::utils::hex;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tokio::task;

use nullforge::erc8117;
use nullforge::gpu::MetalContext;
use nullforge::miner::cpu::cpu_worker;
use nullforge::miner::gpu_driver::{run_batches, N_THREADS};
use nullforge::miner::shared::MinerShared;

/// CPU workers = round(num_cpus * UTILIZATION); ~20% of cores left for the
/// system + the (I/O-bound) GPU driver thread.
const UTILIZATION: f64 = 0.80;

#[tokio::main]
async fn main() {
    // CLI: `nullforge-gpu [target_zeros]`, default 8 (mirrors the CPU tool).
    let target: usize = match env::args().nth(1) {
        None => 8,
        Some(arg) => match arg.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!(
                    "Invalid leading-zero count: {arg:?}\nUsage: nullforge-gpu [target_zeros]   (positive integer, default 8)"
                );
                std::process::exit(2);
            }
        },
    };

    let cpu_workers = ((num_cpus::get() as f64) * UTILIZATION).round().max(1.0) as usize;
    println!(
        "nullforge-gpu: {cpu_workers} CPU workers + GPU, finding an address with {target} leading zeros"
    );

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("scanned_keys.txt")
        .expect("Unable to open scanned_keys.txt");

    let start = Instant::now();
    let shared = Arc::new(MinerShared::new(target, file, start));

    // Build the GPU context up front so a missing device fails fast and clearly.
    let ctx = MetalContext::new();

    // Full-entropy per-thread seeds (generated once; the GPU driver advances the
    // per-thread counter across batches, so keys never repeat).
    let mut rng = StdRng::from_entropy();
    let mut seeds = vec![[0u8; 32]; N_THREADS];
    for s in seeds.iter_mut() {
        rng.fill_bytes(s);
    }

    // Ctrl-C -> request stop.
    {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            println!("Received Ctrl+C. Stopping...");
            shared.request_stop();
        });
    }

    // Rate reporter (per-engine attribution) every 5s until stop.
    let rate_handle = {
        let shared = Arc::clone(&shared);
        task::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                if shared.should_stop() {
                    break;
                }
                let secs = shared.elapsed_secs().max(1.0);
                let cpu = shared.cpu_keys() as f64 / secs;
                let gpu = shared.gpu_keys() as f64 / secs;
                println!(
                    "CPU: {:.2} Mkeys/s | GPU: {:.2} Mkeys/s | total {:.2} Mkeys/s | best: {}",
                    cpu / 1e6,
                    gpu / 1e6,
                    (cpu + gpu) / 1e6,
                    shared.best_zeros()
                );
            }
        })
    };

    // GPU driver on a dedicated blocking thread (owns `ctx`, `seeds`).
    let gpu_handle = {
        let shared = Arc::clone(&shared);
        task::spawn_blocking(move || run_batches(&ctx, shared, &seeds))
    };

    // CPU workers.
    let mut cpu_handles = Vec::new();
    for _ in 0..cpu_workers {
        let shared = Arc::clone(&shared);
        cpu_handles.push(task::spawn_blocking(move || cpu_worker(shared)));
    }

    let _ = gpu_handle.await;
    for h in cpu_handles {
        let _ = h.await;
    }
    rate_handle.abort();

    match shared.take_best() {
        Some(best) => {
            println!("Found wallet with the most leading zeros:");
            println!("Address (raw):      {}", best.address_str);
            println!(
                "Address (ERC-8117): {}",
                erc8117::format_both(&best.address_str, false)
            );
            println!("Private Key: {}", hex::encode(best.privkey));
        }
        None => println!("No wallet found."),
    }
}
```

- [ ] **Step 2: Build the binary**

Run: `cargo build --bin nullforge-gpu`
Expected: compiles cleanly (warnings from transitive deps are fine).

- [ ] **Step 3: Smoke-run to a small target and confirm output shape**

Run (finds quickly, low target):
```bash
cd "$(git rev-parse --show-toplevel)" && rm -f scanned_keys.txt && timeout 60 ./target/debug/nullforge-gpu 4; echo "exit: $?"
```
Expected: at least one `New best [CPU|GPU] N leading zeros: 0x0₄…  (0x0(4)…)` line; a final `Found wallet …` block with raw + ERC-8117 + private key; and `scanned_keys.txt` containing a subscript, non-truncated address column. (Exit 0 on natural finish at 4; 124 if `timeout` fired — re-run with a lower target if so.)

- [ ] **Step 4: Commit**

```bash
git add src/bin/gpu.rs
git commit -m "feat(gpu-bin): unified CPU+GPU miner host loop + CLI + ERC-8117 output"
```

---

### Task 5: GPU-gated end-to-end integration test

**Files:**
- Modify: `tests/gpu_stages.rs` (append a new test)

**Interfaces:**
- Consumes: `nullforge::gpu::MetalContext`, `nullforge::miner::shared::MinerShared`, `nullforge::miner::gpu_driver::run_batches`.

- [ ] **Step 1: Write the failing (or device-skipped) test**

Append to `tests/gpu_stages.rs`:

```rust
/// End-to-end: the GPU driver, given a low target, finds a verified hit, writes
/// the file, and trips `stop`. Mirrors the device-gated style of the other GPU
/// tests in this file (runs on the Metal device present in CI/dev machines).
#[test]
fn gpu_driver_finds_and_reports_low_target() {
    use std::sync::Arc;
    use std::time::Instant;
    use nullforge::gpu::MetalContext;
    use nullforge::miner::gpu_driver::{run_batches, N_THREADS};
    use nullforge::miner::shared::MinerShared;

    let ctx = MetalContext::new();

    let file = tempfile::NamedTempFile::new().unwrap();
    // target 2 => the very first batch should surface >=2-zero hits fast.
    let shared = Arc::new(MinerShared::new(2, file.reopen().unwrap(), Instant::now()));

    // Deterministic distinct seeds (content is irrelevant to correctness).
    let mut seeds = vec![[0u8; 32]; N_THREADS];
    for (i, s) in seeds.iter_mut().enumerate() {
        s[0..8].copy_from_slice(&(i as u64).to_le_bytes());
    }

    run_batches(&ctx, Arc::clone(&shared), &seeds);

    assert!(shared.should_stop(), "should stop after reaching target 2");
    let best = shared.take_best().expect("should have a best");
    assert!(best.zeros >= 2, "best zeros {} >= target 2", best.zeros);

    use std::io::Read;
    let mut contents = String::new();
    file.reopen().unwrap().read_to_string(&mut contents).unwrap();
    assert!(!contents.trim().is_empty(), "file should have a hit line");
}
```

Add `tempfile` to `[dev-dependencies]` if Task 1 didn't already (it did).

- [ ] **Step 2: Run the test**

Run: `cargo test --test gpu_stages gpu_driver_finds_and_reports_low_target -- --nocapture`
Expected: PASS on a machine with a Metal device (the whole `gpu_stages.rs` suite is device-gated by nature; on a device-less CI it will fail at `MetalContext::new()` exactly like the existing tests, which is the established convention here).

- [ ] **Step 3: Commit**

```bash
git add tests/gpu_stages.rs
git commit -m "test(miner): GPU-gated e2e — driver finds, verifies, reports, stops"
```

---

## Self-Review

**Spec coverage:**
- Unified process, CPU workers + GPU driver + shared best-tracker → Tasks 1–4. ✓
- Strictly-increasing "new best" reporting (leading-zeros only) → `report_hit` (Task 1), tested. ✓
- ERC-8117 output (console both-truncated; file subscript-non-truncated; final raw+both) → Task 1 + Task 4. ✓
- Per-engine rate attribution → Task 1 counters + Task 4 rate task. ✓
- 80% CPU cores / 80% GPU duty → Task 4 (`UTILIZATION`) / Task 3 (`run_batches` sleep). ✓
- Verify-every-hit + hard abort → Task 3 (`verify_hit_or_err` + `exit(1)`), abort path is a returning `Err` for testability. ✓
- CLI (default 8, non-numeric → exit 2), no-Metal handling, file open → Task 4. ✓
- Stop at target or Ctrl-C + final summary → Task 4. ✓
- Batch params `N_THREADS`/`ITERS`/`GPU_FLOOR`, threshold policy, saturation floor → Task 3. ✓
- GPU crypto optimization (comb table, batched inverse) → **out of scope for Stage 1**; separate follow-on plans (see below). Spec §"Staging". ✓ (intentional deferral)

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `report_hit(engine, privkey, address_str, zeros) -> bool`, `Engine`, `FoundKey`, `MinerShared::{new, should_stop, request_stop, best_zeros, add_keys, cpu_keys, gpu_keys, elapsed_secs, take_best}`, `leading_zero_nibbles(&[u8])`, `gpu_threshold(usize,usize)->u32`, `verify_hit_or_err(&MetalContext,&Hit)->Result<usize,String>`, `run_batches(&MetalContext, Arc<MinerShared>, &[[u8;32]])`, `N_THREADS` — names/signatures are consistent across Tasks 1–5 and the binary. ✓

## Follow-on plans (not this plan)

- **Stage 2 — GPU comb table for `k·G`** (edits `kernels/ec.metal`/`field.metal` + a precomputed-table device buffer; verified bit-identical to k256 via the existing GPU-vs-k256 tests).
- **Stage 3 — GPU per-thread batched modular inverse** (Montgomery's trick across a thread's `ITERS` points; verified identical to k256).

Both require the `mine` kernel to be committed (see Prerequisites) and should be brainstormed/spec'd for the crypto-kernel details before planning.
