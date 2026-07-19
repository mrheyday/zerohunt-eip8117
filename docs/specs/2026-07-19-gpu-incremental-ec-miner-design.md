# zerohunt GPU incremental-EC miner — design ("Approach B")

**Date:** 2026-07-19
**Status:** approved (design), pending implementation plan
**Related:** `docs/specs/2026-07-17-gpu-vanity-miner-design.md` (Approach A, ships
today as `kernels/miner.metal`'s `mine` kernel), `docs/specs/2026-07-18-unified-cpu-gpu-max-leading-zero-miner-design.md`

## Goal

Approach A's design doc explicitly deferred an optimization: "the incremental
point-addition speed optimization (Approach B / profanity-style)... a possible
follow-up, gated on Approach A being proven correct and benchmarked." Approach A
is shipped, tested, and integrated (unified CPU+GPU miner, `docs/specs/2026-07-18-...`).
This spec is that follow-up: replace the GPU `mine` kernel's per-candidate full
secp256k1 scalar multiplication with one scalar multiplication per **thread per
batch**, then a short walk of cheap Jacobian point additions for the rest of the
batch's candidates.

## Why this is the right lever (cost model)

Rough field-multiplication counts per candidate:

| Path | Scalar mult | Inversion | Total (approx) |
|---|---|---|---|
| Approach A (today) | ~4000+ (256-step double-and-add) | ~260 (Fermat) | ~4300+ |
| **B1 (this spec)**: incremental walk, per-candidate inversion | one `j_add` (~16) | ~260 (Fermat) | **~275 → ~15x over A** |
| B2 (future): batch-inverted B1 | ~16 | ~20 (amortized) | ~35 → another ~8-10x |

B1 alone is the big win, because dropping the scalar-mult makes the per-candidate
inversion the new bottleneck. B2 (Montgomery batch inversion across a sub-batch
of Jacobian points) is a **separate, later** optimization — it needs to hold a
sub-batch of Jacobian points for back-substitution, which is a real memory-layout
decision (sub-batch size, register pressure) independent of B1's correctness.
**Out of scope for this spec**, same for the fixed-base comb table (only speeds
the once-per-thread base multiply, which is noise once the scalar-mult is
already amortized over a whole batch).

## Structure: batch-scoped base, bounded walk

Per thread, per **batch** (not per candidate):

1. `base = keccak256(seed ‖ base_counter)`, reduced by rejection (not
   modular reduction) exactly like Approach A's `derive_privkey` +
   `scalar_in_range`: if `base == 0` or `base >= n` (secp256k1 order),
   skip the whole batch for this thread. Probability `~2^-128`; correctness-wise
   identical in spirit to Approach A's existing per-candidate guard, just scoped
   to the batch's base instead of every candidate.
2. One full scalar multiplication: `P_0 = base * G` (Jacobian, no final
   inversion — this reuses Approach A's double-and-add, just stops before the
   affine conversion).
3. Walk `P_it = P_{it-1} + G` for `it` in `[1, iters)` (one `j_add` per step).
4. For each `it`, convert `P_it` to affine (one inversion — B1 does this **every
   iteration**; B2 will batch it later) and hash to an address, exactly as
   Approach A does.
5. The emitted private key for candidate `it` is `(base + it) mod n` — computed
   as a 256-bit add followed by **at most one** conditional subtraction of `n`
   (valid because `base < n` and `it < ITERS_PER_BATCH ≪ n`, so `base + it < 2n`).

This reuses the *exact same* seed/counter derivation, buffer layout, and
per-hit host-side `k256` verification gate as Approach A — `gpu_driver.rs`
changes only in which kernel entry point it dispatches. The alternative
(carrying Jacobian state across dispatches, an unbounded walk) is rejected:
it would need new host-side per-thread state and gives no correctness benefit,
only a bigger correlated-key window (see below).

## Security model — how this differs from Approach A (be explicit)

Approach A: every candidate key is an independent random-oracle output
(`keccak256(seed‖counter)`), full 256-bit entropy, no relationship between any
two emitted keys. That is the strongest possible model and is why the profanity
incident (weak per-thread seed entropy, not the incremental-math technique
itself) doesn't apply to Approach A.

**Approach B is weaker, and this spec states the weakening precisely rather than
hand-waving it:**

- Within one batch (`ITERS` candidates, currently 256), all emitted keys are
  **affinely related**: `key_a - key_b = a - b`, both public offsets known to
  anyone who sees two keys from the same batch. Across different batches (and
  different threads), keys are unrelated (`base` is a fresh full-entropy oracle
  output every batch).
- This relation is **not** a weakness on its own: recovering `base` from
  `P_it = (base+it)*G` is the discrete log problem, exactly as hard as recovering
  any individual Approach-A key from its public point. Each emitted key is
  individually a uniform, unpredictable 256-bit scalar (mod the negligible
  batch-skip case above).
- The actual security invariant is **base-scalar secrecy + DLP hardness**, with
  the *correlation set* bounded to `ITERS` keys (currently 256) sharing one
  base. This is a materially different, and strictly weaker, model than "every
  key is an independent oracle output" — smaller `ITERS` shrinks the
  correlation set at the cost of amortizing the scalar-mult over fewer
  candidates (throughput). This spec keeps `ITERS = 256` (the existing
  Approach-A batch size) and does not tune it further.
- We are only mining our own vanity addresses and immediately hold the private
  keys ourselves (this is not a "generate addresses for others to trust" tool);
  the correlation only matters if a batch's keys leak partially and an attacker
  wants to derive the rest of that batch, which requires solving discrete log
  regardless. Documented here because this codebase exists specifically because
  a previous vanity miner (`profanity`) shipped a weak model without disclosing
  it.

## Correctness requirements (non-negotiable, matches Approach A's gate)

- **Every** GPU hit is still re-derived host-side via `k256` before being
  trusted (`verify_hit` / `verify_hit_or_err`, unchanged). A mismatch is still a
  hard abort. This invariant does not change with this spec.
- New arithmetic (`(base + it) mod n`) must be **bit-exact** against a
  host-computed reference (`ethers::types::U256` big-integer mod-n arithmetic),
  including the `n`-wrap boundary: `base` within `ITERS` of `n` is the only case
  where the conditional subtraction actually fires, and it must be tested
  explicitly (natural occurrence probability is `~ITERS / n ≈ 2^-248`, so a
  random test batch will never exercise it).
- The point-at-infinity case (`base + it ≡ 0 mod n` for some `it` in the walk,
  same probability) must be treated as a scalar-range-guard miss (skip that
  candidate), not hashed as if it were a real address — mirrors Approach A's
  `scalar_in_range` rejection of `privkey == 0`.

## Scope / YAGNI

- **In scope:** B1 (incremental walk, per-candidate inversion) as the new
  default GPU-mining kernel, replacing Approach A's `mine` kernel in
  `gpu_driver::run_batches`. Approach A's kernel and code stay in the tree
  (used by the existing `ec_test`/`derive_test`/`mine` staged tests and as a
  fallback reference), just no longer the hot path.
- **Out of scope (future work, separately gated):** B2 Montgomery batch
  inversion, fixed-base comb table for the once-per-thread base multiply,
  tuning `ITERS`/correlation-set size, multi-GPU, checkpointing.

## Testing (staged, matching the repo's verified-vs-k256 discipline)

1. Refactor `ec.metal`'s `scalarmul` into `scalarmul_jacobian` (returns the
   Jacobian point, no final inversion) + a thin `scalarmul` wrapper — pure
   refactor, existing `ec_test`-based tests must still pass bit-exact.
2. `scalar_add_small` (the `(base + it) mod n` arithmetic) as an isolated,
   directly-testable kernel entry point vs `U256` host reference, including the
   explicit `n`-wrap case.
3. A raw-base incremental-walk test kernel (input: explicit 256-bit bases, not
   keccak-derived) producing `(privkey, address)` pairs for a small `iters`
   window per base — bit-exact vs `k256` for random bases **and** the explicit
   `n`-wrap boundary (`base = n - 2`, `iters = 4`).
4. The production `mine_incremental` kernel (keccak-derived base, threshold-gated
   hit buffer, same wire format as `mine`) — GPU-gated e2e test mirroring
   `tests/gpu_stages.rs`, and `gpu_driver::run_batches` wired to it.
