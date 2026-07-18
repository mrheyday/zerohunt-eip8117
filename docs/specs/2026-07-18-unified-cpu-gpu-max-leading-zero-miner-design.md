# Unified CPU+GPU max-leading-zero miner — design

**Date:** 2026-07-18
**Status:** Draft (design approved, pending spec review → implementation plan)
**Related:** `docs/specs/2026-07-17-gpu-vanity-miner-design.md`, `docs/plans/2026-07-17-gpu-vanity-miner.md`, ERC-8117 output (`src/erc8117.rs`)

## Goal

Maximize the rate at which we mine Ethereum addresses with the most leading zero
nibbles, and report results the way the CPU tool already does: a stream of
"new best" records, **each with strictly more leading zeros than the last**,
displayed in ERC-8117 notation. One process runs the CPU and GPU engines
together against a single shared best-tracker and output file.

Success = (a) higher sustained throughput than either engine alone, and
(b) correct, strictly-increasing "new best" reporting identical in spirit to
`src/main.rs`.

**Ranking metric:** the hybrid ranks candidates by **leading-zero nibble count
only**, matching the stated goal ("each higher leading zeros than the one
before"). It intentionally drops `main.rs`'s secondary "repeating characters in
order" tiebreak — the GPU `mine` kernel reports only the zero count, and the
goal is leading zeros. The pure-CPU `zerohunt` keeps its existing behavior.

## Measured baseline (Apple M1 Pro — 10 CPU cores, 16-core GPU, release build)

| Engine | Rate | How measured |
|---|---|---|
| CPU, all 10 cores | ~180,000 keys/s | `zerohunt 99` printed rate |
| GPU, current kernel | ~90,000 keys/s | timing `derive_address_gpu` |

Two findings drive this design:

1. **The GPU is currently ~half the CPU's speed** — the opposite of the usual
   "GPU dominates" assumption. So the CPU is today's workhorse; the design must
   not starve it.
2. **The GPU bottleneck is not SIMD-group width.** Widening the dispatch from
   1-thread/threadgroup to full 32-wide threadgroups left throughput unchanged
   (~0.08 vs 0.09 Mkeys/s). The limiter is **per-thread register pressure**
   (each thread holds many 256-bit temporaries → low occupancy) combined with a
   **naive scalar-multiply** (256-step double-and-add, no precomputed table).

### Explicitly ruled out: Metal 4 inline ML / tensor ops

Metal 4's `matmul2d` / cooperative-tensor / neural-accelerator path was
evaluated (Apple's "Running Inline ML Operations in a Shader" sample builds and
runs correctly on this machine: `✅ GPU == CPU`). It **cannot** accelerate this
workload: tensor cores do approximate low-precision (fp16/bf16/int8) matrix
MACs with no exact 32×32→64-bit integer products and no carry propagation, so
they cannot perform secp256k1 modular field arithmetic or Keccak-256 bit
permutation. Additionally the hardware neural accelerator requires GPU family
`apple10`; the M1 Pro is `apple7`. The throughput levers are the crypto-kernel
optimizations below, which are Metal-version-agnostic.

## Architecture

Single binary `zerohunt-gpu` (`src/bin/gpu.rs`), keeping the existing tokio
runtime. The pure-CPU `zerohunt` (`src/main.rs`) stays as-is for
GPU-less / quick runs.

Threads:
- **CPU workers ×`round(num_cpus * 0.80)`** — the existing random-key loop,
  reporting through the shared funnel. ~80% of cores; ~20% left for the system.
- **GPU driver ×1** — owns `MetalContext`; loops dispatch → verify → report.
  I/O-bound (mostly blocked on the GPU), so it lives in the ~20% system slack
  rather than taking a mining core.
- **Rate reporter ×1** (tokio, ~5 s) — reads per-engine atomics, prints
  attribution.
- **Ctrl-C ×1** — sets the shared `stop` flag.

```
                 ┌──────────── Arc<MinerShared> ────────────┐
                 │ stop · best_zeros · best · file · counters │
                 └───▲──────────▲───────────────▲────────────┘
   CPU worker ×~0.8n │          │ GPU driver ×1 │ rate task ×1
   random key→addr───┘   dispatch→verify→────────┘  reads counters
        → report_hit()         → report_hit()        every ~5s
```

`UTILIZATION = 0.80` is one named constant driving both the CPU core count and
the GPU duty cycle (below); it becomes a CLI flag later.

### GPU duty-cycle throttle

Metal exposes no "use 80% of the GPU" dial, so the GPU driver targets ~80% duty
by sleeping `((1-UTILIZATION)/UTILIZATION)·T ≈ 0.25·T` after a batch that took
`T` to compute. This leaves ~20% GPU headroom for the display/system (relevant
on Apple Silicon, where the GPU is shared) at a small throughput cost.

## Components (library-first, so each unit is testable in isolation)

- **`zerohunt::miner::shared`** — the unified core, GPU-free and thread-free to test:
  - `MinerShared` (`Arc`): `target: usize`, `best_zeros: AtomicUsize`,
    `best: Mutex<Option<FoundKey>>`, `file: Mutex<File>`,
    `cpu_keys/gpu_keys: AtomicU64`, `stop: AtomicBool`, `start: Instant`.
  - `enum Engine { Cpu, Gpu }`, `struct FoundKey { privkey:[u8;32], address_str:String, zeros:usize }`.
  - `report_hit(&self, engine, privkey, address_str, zeros)` — the single funnel
    both engines call. Fast-path `zeros <= best_zeros → return`; else lock,
    re-check, update, write file (ERC-8117), print (ERC-8117), trip `stop` at
    target.
- **`zerohunt::miner::cpu`** — `cpu_worker(shared)`: the random-key loop from
  `main.rs`, thin over `report_hit`.
- **`zerohunt::miner::gpu_driver`** — `run_batches(ctx, shared, seeds, params)`:
  the dispatch→verify→report loop. Verification decision is a **returning**
  function (`Result`), so the abort path is testable; the binary turns `Err`
  into `exit(1)`.
- **`src/bin/gpu.rs`** — thin wiring: parse args, build `MinerShared`, spawn
  threads + Ctrl-C + rate task, join, final summary.
- Reuses `src/erc8117.rs` for all address rendering.

Optional (out of scope, noted): `main.rs` could later be refactored to reuse
`cpu_worker`, removing the duplicated loop.

## Data flow

**CPU worker** (per key): random key → address → count leading-zero nibbles
(existing byte logic) → if `zeros > best_zeros`, render `address_str` and
`report_hit(Cpu, …)`. Batches its key counter (like `COUNTER_FLUSH`).

Initial batch parameters (tunable constants; the plan may adjust after
measuring): `N_THREADS = 65536`, `ITERS = 256` (~16.7M candidates/batch),
`GPU_FLOOR = 4`.

**GPU driver** (per batch):
- One-time: `N_THREADS` full-entropy seeds; global `base` counter.
- Batch: `threshold = max(GPU_FLOOR, best_zeros)`;
  `hits = dispatch_mine(seeds, [base;N], ITERS, threshold)`; `base += ITERS`;
  `gpu_keys += N*ITERS`.
- Each hit: **verify vs k256** → mismatch ⇒ `Err` ⇒ hard-abort `exit(1)`.
  Recompute zeros from the *verified* address (authoritative), render
  `address_str`, `report_hit(Gpu, …)`.
- Throttle `0.25·T`; check `stop` between batches.
- `GPU_FLOOR = 4` keeps a full batch under the kernel's 1024-hit cap
  (16.7M / 16⁴ ≈ 256 hits ≪ 1024); if the kernel ever reports more than the
  cap, warn (saturation). If `target_zeros < GPU_FLOOR`, clamp the threshold to
  `target_zeros`.

**`report_hit`** (the funnel):
1. `if zeros <= best_zeros { return }` — no lock. Strictly-greater gating gives
   the CPU-style "each new best beats the last" and prevents the two engines
   double-reporting the same level.
2. Lock `best`; re-check (lost race → return); store `best_zeros`, set `best`.
3. File line: `total \t {erc8117 subscript, non-truncated} \t zeros \t privkey-hex` + flush,
   where `total` = snapshot of combined keys tried (`cpu_keys + gpu_keys`).
4. Print: `New best [CPU|GPU] {zeros} leading zeros: {erc8117 both-modes, truncated}`.
5. `if zeros >= target { stop.store(true) }`.

**Rate task** (~5 s): `CPU: … /s | GPU: … /s | total … /s | best: N`.
**Final summary:** raw + ERC-8117 (both, non-truncated) + private key, or
"No wallet found."

## Maximize-output: GPU crypto-kernel optimization

This is where "maximize" is won (measured need: 0.08 → target multi-Mkeys/s).
All Metal-version-agnostic; none involve tensor ops.

1. **Fixed-base comb table for `k·G`.** Precompute multiples of the generator
   `G` into a read-only device buffer; replace the naive 256-step double-and-add
   with windowed table lookups + few adds. Every thread shares the table. Biggest
   single win, and it removes temporaries (helping occupancy too).
2. **Per-thread batched modular inverse.** Each thread already loops `ITERS`
   keys. Keep each result in Jacobian, store the `ITERS` `Z` values, then do one
   Montgomery batch-inversion over the thread's own `ITERS` points before the
   affine + address step. Amortizes the expensive `fe_inv` ~`ITERS`×, with **no**
   cross-thread coordination.
3. **Reduce register pressure → occupancy.** Fewer live 256-bit temporaries
   (the comb and Jacobian-batching both help). This is the measured limiter.

Non-goals for this optimization: Montgomery-form field mul (possible later),
multi-command-buffer pipelining, and any Metal 4 command-model rewrite (a clean
modernization but zero crypto speedup — explicitly deferred).

## Correctness & error handling

- **Every GPU hit is re-derived on the host via `k256` (`verify_hit`) before it
  is trusted.** Mismatch ⇒ hard-abort `exit(1)` with the privkey + expected/got
  address. A GPU bug can waste time but can never emit a bad key. This invariant
  is preserved through all three optimization stages.
- No Metal device → friendly message + `exit(1)` suggesting CPU-only `zerohunt`
  (no panic).
- Arg parse mirrors the CPU tool: `zerohunt-gpu [target_zeros]`, default 8,
  non-numeric → usage + `exit(2)`.
- File-open failure → error + exit.
- Batch saturation (hit_count > cap) → warn, continue with clamped hits.

## Testing (TDD, matching the repo's staged / verified-vs-k256 discipline)

- **Unit, no GPU:** `report_hit` — best update + ERC-8117 file line + `stop`
  trips exactly at target; strictly-greater gating; higher-wins race; exact
  file/console format.
- **Unit:** GPU `Hit` bytes → `address_str` → ERC-8117 for a known address; the
  verify-decision function returns `Err` on a corrupted address (abort path
  without calling `exit`).
- **GPU-gated e2e** (real Metal, like `tests/gpu_stages.rs`): drive
  `gpu_driver` at `target=2`, assert best set + file written + `stop` trips.
- **Optimization equivalence:** after each of the three GPU stages, the existing
  GPU-vs-`k256` derive/address tests must still pass bit-for-bit (comb table and
  batched inverse must produce identical addresses to the naive path).

## Dependencies & staging

**Dependency flag:** the `mine` kernel + `dispatch_mine` + `Hit` are currently
**uncommitted WIP** in the main checkout (`M kernels/miner.metal`,
`M src/gpu/mod.rs`, `M tests/gpu_stages.rs`), not on the committed base this
design was drafted against. The optimization edits `ec.metal` / `field.metal` /
the `mine` kernel, so implementation must first base on (or land) that WIP.

**Staging** (each a verified step; throughput climbs without ever risking a bad
key):
1. **Integration** — unified process + progressive strictly-increasing best
   reporting (ERC-8117) on the *current* kernel. Immediate value.
2. **Comb table** for `k·G` — verified identical to `k256`.
3. **Per-thread batched inverse** — verified identical to `k256`.

Each stage is independently testable and shippable.
