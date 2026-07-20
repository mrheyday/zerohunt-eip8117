# nullforge GPU vanity miner — design (Metal, Approach A)

**Date:** 2026-07-17
**Status:** approved (design), pending implementation plan
**Target hardware:** Apple M1 Pro (16 GPU cores, Metal 4, aarch64-apple-darwin). No CUDA.

## Goal

GPU-accelerate the leading-zero Ethereum vanity-address search that the existing
CPU tool (`src/main.rs`) performs, using Apple Metal compute. Must be **at least
as cryptographically sound** as the CPU path — every emitted private key must be
full-entropy and unpredictable.

Non-goal (this spec): the incremental point-addition speed optimization
("Approach B" / profanity-style). That is a possible follow-up, gated on Approach
A being proven correct and benchmarked.

## Background / why not just "use the GPU"

The crypto (secp256k1 key derivation + Keccak-256) runs on CPU via `k256`; those
libraries have no GPU backend. GPU acceleration means implementing secp256k1
scalar multiplication + Keccak-256 as Metal compute kernels. On Apple Silicon the
only viable GPU compute route is Metal (near-all Ethereum GPU vanity miners are
CUDA/NVIDIA and won't run here).

**Security precedent:** the `profanity` GPU vanity tool seeded GPU threads weakly,
collapsing private-key entropy to ~2^32, and attackers drained millions from
addresses it produced. Any GPU design here must seed each thread from full OS
entropy. This spec's security model (§2) is built around that requirement.

## 1. Architecture & integration

- New **independent binary target** `nullforge-gpu` (`src/bin/gpu.rs`). The working
  CPU tool (`src/main.rs`) stays untouched; the `metal` crate dependency is scoped
  to the GPU binary only.
- Rust host drives Apple Metal compute (via the `metal` crate — gfx-rs/metal-rs,
  native MSL, best Apple GPU support).
- The compute kernel is **MSL** (Metal Shading Language), compiled at runtime from
  an embedded source string (`device.new_library_with_source`) for fast iteration.
- Host-side crypto for the correctness gate reuses the existing `ethers`/`k256`
  path already in the crate.

## 2. Security model (must be right)

- Host generates **N full-entropy 256-bit thread seeds** from the OS CSPRNG
  (`rand::rngs::OsRng` / `getrandom`) at startup — one per concurrent GPU thread.
- Each GPU thread derives candidate keys as
  `privkey = Keccak256(thread_seed_32B ‖ counter_u64)`.
- Because `thread_seed` is full-entropy from the OS, each derived key is
  effectively a random-oracle output → full ~256-bit entropy and unpredictable.
  This structurally avoids the profanity flaw (base entropy ~2^32).
- Reuses the Keccak kernel (no separate, easy-to-get-wrong GPU RNG).
- The winning private key is written directly to the output buffer AND
  independently re-derivable by the host from `(thread_seed, counter)` — both are
  cross-checked by the correctness gate (§4).

## 3. GPU compute per thread

Each thread loops K iterations per dispatch (amortizes dispatch overhead):

1. Derive `privkey = Keccak256(thread_seed ‖ counter)`; `counter += 1`.
   - **Scalar-range guard:** if `privkey == 0` or `privkey >= n` (the secp256k1
     group order), skip this candidate (256-bit compare; negligibly rare,
     ~2^-128). This keeps the stored key canonical so the host `k256` gate (§4)
     re-derives an identical address instead of rejecting/reducing a raw scalar
     and false-aborting.
2. **secp256k1 scalar multiplication** `P = privkey · G`:
   - Field arithmetic: 256-bit values as **8 × u32 limbs**, mod the secp256k1
     prime `p`. Uses Metal-native u32 with `mulhi(a,b)` for 32×32→64 products.
   - Point mul: double-and-add over the 256-bit scalar in **Jacobian coordinates**
     (avoids per-step modular inversion).
   - **One modular inversion** at the end (Jacobian → affine `x, y`) via **Fermat**
     exponentiation (`a^(p-2) mod p`) — branch-free, one per key.
3. **Keccak-256** of the 64-byte `x ‖ y`; address = last 20 bytes of the digest.
4. Count leading-zero **nibbles** of the address (matches the CPU tool's metric).
5. On a hit (`zero_count >= threshold`): atomically claim a slot in the results
   buffer via an atomic counter and write `{ privkey[32], address[20], zero_count,
   thread_id, counter }`.

Field math correctness (mul, add/sub with carry/borrow, inversion) is the highest-
risk area and is gated by dedicated test vectors before the EC layer is trusted.

## 4. Host loop & correctness gate (non-negotiable)

- Host dispatches the kernel over a grid sized to the GPU (threadgroups ×
  threads-per-threadgroup), reads back the results buffer after each dispatch,
  advances per-thread counters, and repeats until stop (Ctrl-C) or the target
  zero count is reached.
- **Correctness gate:** for **every** hit the GPU reports, the host re-derives the
  address from the returned private key using the CPU `k256` + Keccak path and
  asserts it matches the GPU-reported address **and** the leading-zero count.
  - **Any mismatch → hard abort** with a diagnostic. A mismatch means the kernel is
    buggy; an unverified key is never saved or trusted.
  - Only verified hits update the best, write to `scanned_keys.txt` (same format as
    the CPU tool), and print.
- The cosmetic "repeating characters" ranking (from the CPU tool) is computed
  **host-side** on the few verified hits — kept out of the hot kernel.

## 5. Testing (staged gates — build in this order)

Each stage must pass before the next is built:

1. **Keccak kernel** vs known vectors: `Keccak256("")`,
   `Keccak256("abc")`, and a known pubkey→address derivation. Run on-GPU, compare
   to host reference bytes.
2. **Field arithmetic** vs reference vectors: modular `mul`, `add`, `sub`, and
   `inverse` on secp256k1 `p` against host-computed (or published) vectors.
3. **Full GPU `privkey → address`** vs CPU `k256` for a batch of known private keys
   (including edge cases: `1`, `n-1`, random) — **bit-exact** match required.
4. Only after 1–3 pass: wire the **miner loop** + correctness gate and run.

## Scope / YAGNI

- MVP = leading-zero search + full-entropy keys + verified key output.
- **Out:** Approach B (incremental), in-kernel repeating-char ranking, config
  beyond `target_zeros` (CLI arg, default 8), multi-GPU, checkpointing.
- CLI: `nullforge-gpu [target_zeros]` (mirrors the CPU tool's arg handling, incl.
  the clean invalid-arg exit).

## Risks

- **Field/EC correctness** — mitigated by staged test vectors + the mandatory
  per-hit CPU re-derivation gate (a bug can waste time but cannot emit a bad key
  that gets saved).
- **Metal u64 / `mulhi` semantics** — verified in stage 2 before EC is built.
- **Effort** — the field + EC kernel (~400–600 lines MSL) is a substantial,
  error-prone core; this is a multi-step staged build, not a quick edit.

## Deliverables

- `src/bin/gpu.rs` (Rust host + embedded MSL kernel source).
- `Cargo.toml` `[[bin]]` target + `metal` dependency (scoped).
- Test harness (stages 1–3) — as `#[test]`s or a `--self-test` mode on the binary.
- This spec.
