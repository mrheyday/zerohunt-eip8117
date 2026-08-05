# nullforge GPU batched modular inversion — design ("Approach B2")

**Date:** 2026-07-21
**Status:** proposed (design), pending implementation
**Related:** `docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` (Approach B / B1,
shipped), `kernels/miner.metal` (`mine_incremental`), `kernels/ec.metal`
(`scalarmul_jacobian`, `jpoint`), `kernels/field.metal` (`fe`, `fe_inv`, `fe_mul`).
README Roadmap "Stage 3: per-thread batched modular inverse (Montgomery's trick)".

## Goal

Approach B1 replaced Approach A's per-candidate scalar-multiply with one scalar-mult
per batch + a cheap incremental point-add walk (`P_it = P_{it-1} + G`). That made the
**per-candidate modular inversion the new bottleneck**: converting each Jacobian
`P_it` to affine needs `fe_inv(P.Z)`, a Fermat inverse (`z^(p-2) mod p`, ~256
squarings + ~a dozen mults ≈ **~260 field-mults each**), done once per candidate at
`kernels/miner.metal:361`.

B2 amortizes that away with **Montgomery's batch-inversion trick**: invert `K`
field elements with **one** `fe_inv` + `~3(K-1)` `fe_mul`, instead of `K` separate
`fe_inv` calls.

## Cost model (extends the B1 spec's table)

Field-multiplications per candidate:

| Path | scalar-mult | inversion | keccak/EC misc | total |
|---|---|---|---|---|
| A (per-candidate `k·G`) | ~4000 | ~260 | ~16 | ~4300 |
| **B1 (shipped)** | ~16 (`j_add`) | **~260 (Fermat)** | ~16 | **~290** |
| **B2 (this spec)** | ~16 | **~3 + 260/K (amortized)** | ~16 | **~35** (K≥16) |

B1 → B2 is **~8×** because the Fermat inverse — now the dominant term — is shared
across a whole sub-batch. (The fixed-base comb table for the once-per-batch `base·G`
is a *separate* lever, out of scope here — it's noise once the scalar-mult is already
amortized over the batch.)

## Montgomery batch inversion (the algorithm)

Given `z_0 … z_{K-1}` (all nonzero), compute all `z_i^{-1}` with one inverse:

```
# prefix products:  pre[i] = z_0 * z_1 * … * z_i
pre[0] = z_0
for i in 1..K:  pre[i] = fe_mul(pre[i-1], z_i)

acc = fe_inv(pre[K-1])          # the ONE inversion = (z_0·…·z_{K-1})^{-1}

# back-substitute, high → low:
for i in (K-1)..=1:
    zinv[i] = fe_mul(acc, pre[i-1])   # = (z_0·…·z_{K-1})^{-1} · (z_0·…·z_{i-1})
    acc     = fe_mul(acc, z_i)        # strip z_i → (z_0·…·z_{i-1})^{-1}
zinv[0] = acc
```

Cost: `(K-1)` mults (prefix) + `1` inv + `2(K-1)` mults (back-sub) = `3(K-1)` mults + 1 inv.

## Structure change to `mine_incremental`

Today (`kernels/miner.metal:353-361`, per candidate): `P = j_add(P, G)` → `zinv =
fe_inv(P.Z)` → affine `(x,y) = (X·zinv², Y·zinv³)` → keccak address → threshold check.

B2 walks in **sub-batches of `K`**:

1. Advance the walk `K` steps, storing each `jpoint` (`X,Y,Z`) into a thread-local
   array `pts[K]` (and the corresponding `(base + it) mod n` key, or recompute it).
2. Batch-invert the `K` `Z`-coordinates → `zinv[K]` (algorithm above).
3. For each `k`: affine-convert with the precomputed `zinv[k]`, keccak the address,
   threshold-check, emit hit (unchanged wire format / hit ring).

`ITERS` (currently 256) is processed as `ITERS / K` sub-batches. The emitted private
key `(base + it) mod n`, the range guard, and the host `k256` re-verification are all
**unchanged** — B2 only changes *how* the affine `Z^{-1}` is computed.

### Sub-batch size `K` (the register-pressure decision)

Each `jpoint` is `3 × fe = 3 × 8 × u32 = 96 bytes`; `pts[K]` costs `96·K` bytes of
thread-private memory, plus `pre[K]`/`zinv[K]` (`32·K` each). Holding the full
`ITERS=256` at once (~24 KB/thread) would spill and destroy occupancy — hence
sub-batching. Make `K` a `constant uint` (tunable); **start at `K = 16`** (≈1.5 KB
points + 1 KB scratch/thread) and let `forge`-style benchmarking on the M1 pick the
sweet spot (likely 8–32). `ITERS % K == 0` must hold (256 is divisible by 8/16/32).

## Correctness — the point-at-infinity / zero-`Z` edge (non-negotiable)

Batch inversion **fails if any `z_i == 0`** — one zero makes the whole product zero,
so the single `fe_inv` inverts 0 and every result in the sub-batch is garbage. In the
walk, `P.Z == 0` means `P` is the point at infinity, i.e. `base + it ≡ 0 (mod n)` — the
same range-guard-miss B1 already skips (natural probability `~2^-248`).

Handling (must be explicit and tested):
- While building `pre[]`, if `z_k == 0`, substitute `1` into the product for that slot
  **and** mark candidate `k` as `skip`.
- After back-substitution, `skip` candidates are **not** affine-converted / hashed /
  emitted (mirrors B1's `scalar_in_range` / infinity rejection).

This is the one genuinely new failure mode B2 introduces over B1; a naïve batch
inverse that ignores it would silently corrupt a whole sub-batch whenever the wrap
case occurs.

## Testing (staged, matching the repo's verified-vs-reference discipline)

1. **`fe_batch_inv` in isolation** — a small test kernel (or a `#[cfg]` entry) that
   batch-inverts `K` host-supplied `fe`s; assert each equals `fe_inv(that fe)`
   bit-exact. Include a batch that contains a **zero** slot: assert the non-zero
   inverses are still correct and the zero slot is flagged (not silently wrong). New
   case in `tests/gpu_stages.rs`.
2. **`mine_incremental` refactor is behavior-preserving** — the existing
   `incremental_walk_matches_k256` and `mine_incremental_finds_and_verifies_low_threshold`
   (`tests/gpu_stages.rs`) must still pass **bit-exact** after the switch.
3. **n-wrap-in-sub-batch boundary** — drive a base such that `base + it ≡ 0 (mod n)`
   for some `it` inside a sub-batch (`base = n - (K/2)`, small `iters`); assert the
   zero slot is skipped and the other slots' addresses are still correct vs `k256`.
4. Every GPU hit is still re-derived host-side via `k256` before it is trusted
   (`verify_hit`, unchanged; a mismatch remains a hard abort).

## Scope / YAGNI

- **In scope:** batched inversion inside `mine_incremental` (the EOA leading-zero hot
  path — the *only* miner with secp256k1 inversions).
- **Out of scope:** the fixed-base comb table for `base·G` (README Stage 2 — separate,
  lower-priority lever); the CREATE2/CREATE3/CreateX salt miners (keccak-only, no
  inversion — unaffected); tuning `K` beyond picking a sensible benchmarked default;
  keeping Approach A's `mine` / `mine_incremental_raw_test` kernels (they stay as
  reference / staged tests).

## Build note

Metal-only (macOS). The batch-inversion math is a pure MSL change to
`kernels/miner.metal` (+ a `field.metal`/`ec.metal` helper); it must be
`cargo test --test gpu_stages`-verified on an Apple device (bit-exact vs `fe_inv` and
`k256`) — it cannot be compiled or tested on a non-Apple target.
