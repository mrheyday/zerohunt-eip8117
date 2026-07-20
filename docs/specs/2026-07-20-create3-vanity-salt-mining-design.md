# nullforge CREATE3 vanity salt-mining — design

**Date:** 2026-07-20
**Status:** proposed (design), pending implementation plan
**Related:** `docs/specs/2026-07-17-gpu-vanity-miner-design.md` (EOA miner, Approach A),
the CREATE2 salt-mining mode (`src/miner/create2.rs` + `kernels/create2.metal`, shipped),
`docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` (Approach B).

## Goal

Add a **CREATE3** vanity salt-mining mode: mine a `salt` such that the address a
CREATE3 factory deploys a contract to has the maximum number of leading-zero
nibbles. This is the sibling of the existing CREATE2 mode, with one defining
property: **the deployed address is independent of the contract's init code** —
it depends only on `(factory, salt)`. That means a salt mined once is reusable
for *any* contract deployed through the same factory, which the CREATE2 mode
(whose address is bound to a specific `init_code_hash`) cannot offer.

Like CREATE2 mining, this is **keccak-only** (no secp256k1), so it is far faster
than the EOA miner, and the result is a **public salt** — nothing is encrypted;
bests append in plaintext to `scanned_salts.txt`.

## Background: what CREATE3 computes

CREATE3 (0xSequence / Solady `CREATE3`) deploys in two hops:

1. The factory **CREATE2-deploys a fixed proxy** with constant init code
   `PROXY_INIT_CODE`, at
   `proxy = keccak256(0xff ‖ factory ‖ salt ‖ keccak256(PROXY_INIT_CODE))[12:]`.
2. The freshly-deployed proxy does a plain **CREATE at nonce 1** of the user's
   contract, landing it at
   `deployed = keccak256(rlp([proxy, 1]))[12:]`.

Because the proxy's init code is constant, `deployed` is a pure function of
`(factory, salt)` — the user's init code never enters the derivation.

## Derivation (the exact math to mine on)

```
PROXY_HASH = keccak256(PROXY_INIT_CODE)            # 32 bytes, factory-implementation constant
proxy      = keccak256(0xff ‖ factory ‖ salt ‖ PROXY_HASH)[12:]     # 20 bytes  (CREATE2 step)
deployed   = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:]            # 20 bytes  (CREATE, nonce 1)
score      = leading_zero_nibbles(deployed)
```

- `factory` — 20 bytes, the CREATE3 factory address (a CLI parameter, replacing
  CREATE2's `--deployer`).
- `PROXY_HASH` — 32 bytes, **fixed** for a given factory implementation, replacing
  CREATE2's user-supplied `--init-code-hash`. Default to the canonical
  0xSequence/Solady proxy: `PROXY_INIT_CODE = 0x67363d3d37363d34f03d5260086018f3`,
  whose `keccak256` is
  `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`.
  Expose it as `--proxy-hash` (with that default) because other factories use a
  different proxy. **A wrong proxy hash silently mines salts that resolve to the
  wrong address**, so the value must be confirmed against the operator's actual
  factory (see Correctness).
- The second-hop RLP is **fixed-shape**: a fresh proxy's first CREATE is always
  nonce 1, and a 20-byte address always RLP-encodes as `0x94 ‖ addr`, so the
  preimage is the constant-length 23 bytes `0xd6 0x94 ‖ proxy(20) ‖ 0x01`. There
  is no variable-length nonce/RLP branch to handle (unlike general CREATE, where
  nonce ≥ 0x80 changes the encoding).

## Why this is a small, low-risk delta over CREATE2

The CREATE2 mode already gives us the entire scaffold: keccak-only GPU batches,
strictly-increasing best reporting, the mandatory **host-side re-derivation of
every GPU hit** before it is trusted, the duty-cycle throttle, the threshold
logic (`c2_threshold`), and plaintext `scanned_salts.txt` output. CREATE3 changes
exactly two things:

1. The 4th keccak input is the **fixed `PROXY_HASH`** instead of a user
   `init_code_hash` (a parameter swap — no new kernel shape for the first hop).
2. **One additional keccak** over the constant 23-byte RLP preimage turns the
   CREATE2 result (`proxy`) into the scored `deployed` address.

Everything else — grid size, iters, buffer caps, verify gate, reporting — is
reused unchanged.

## Implementation sketch

- **Kernel** (`kernels/create3.metal`, or a `mine_create3` entry appended to
  `create2.metal`): reuse the existing CREATE2 derivation to get `proxy`, then a
  second `keccak256` over the 23-byte RLP buffer to get `deployed`; count leading
  zeros of `deployed` and threshold-gate into the hit buffer with the same wire
  format as `mine_create2` (`salt`, `address`, `zeros`). keccak-only.
- **Host** (`src/miner/create3.rs`, mirroring `create2.rs`):
  - `run_create3(ctx, factory: &[u8;20], proxy_hash: &[u8;32], target, stop)` —
    a near-copy of `run_create2` (same threshold/report/throttle loop, writing
    `scanned_salts.txt`).
  - `MetalContext::dispatch_create3(...)` and `verify_create3(factory,
    proxy_hash, salt, address)` — the host reference re-derives
    `deployed` via the two-keccak formula above and compares all 20 bytes; a
    mismatch **hard-aborts** (identical discipline to `verify_create2`).
- **CLI** (`src/bin/gpu.rs`): `nullforge-gpu N --create3 --factory 0x.. [--proxy-hash 0x..]`,
  mutually exclusive with `--create2`. Validate `factory` = 20 bytes and
  `proxy-hash` = 32 bytes; default `proxy-hash` to the constant above and print
  which proxy it resolved to so a misconfiguration is visible.

## Correctness (non-negotiable, matches the CREATE2/EOA gate)

- **Every GPU hit re-derived host-side** via `verify_create3` before it is
  trusted or written; mismatch is a hard abort. Unchanged discipline.
- The host `verify_create3` must be **bit-exact** against an independent
  reference. Test vectors:
  1. A known CREATE3 deployment: pin one `(factory, salt) → deployed` triple
     computed independently (e.g. Solady `CREATE3.predictDeterministicAddress`
     or an on-chain 0xSequence deployment) and assert `verify_create3` reproduces
     it. This also pins `PROXY_HASH`.
  2. The two-hop composition: assert `deployed` equals
     `create(create2(factory, salt, PROXY_HASH), 1)` computed from the primitive
     CREATE2 + CREATE helpers, for random salts.
  3. `PROXY_HASH` guard: a deliberately-wrong proxy hash must produce a different
     `deployed` (regression guard against the fixed constant being dropped or the
     hop being skipped).

## Build & verification note (environment constraint)

The pure address-derivation math (`verify_create3`, the CPU reference, the
RLP+keccak composition) is **platform-independent Rust** and should be unit-tested
in CI. Today it cannot be, on Linux: the crate declares `metal = "0.33.0"`
unconditionally, so `cargo build`/`cargo test` fail on non-Apple targets
(`E0455: link kind "framework" is only supported on Apple targets`, via
`core-graphics-types`). **Recommendation (tracked separately):** gate the GPU
path — the `metal` dep, `src/gpu/`, the `*_driver`/`dispatch_*` host code, and the
`nullforge-gpu` bin — behind `#[cfg(target_os = "macos")]` / a `gpu` cargo
feature, so the keccak-only CREATE3 reference math and its tests run on Linux CI
while the Metal kernels stay macOS-only. The Metal kernel itself still requires an
Apple device to compile and run its `gpu_stages` test.

## Security / handling

- The salt is **public** — no age encryption, no secp256k1, no key material.
  `scanned_salts.txt` is plaintext (already the case for CREATE2).
- CREATE3's init-code-independence is a deploy-time convenience, not a security
  property of the miner. The only footgun is a wrong `PROXY_HASH`/`factory` (mines
  salts for an address you can't actually deploy to) — mitigated by the vector
  test and by printing the resolved proxy at startup.

## Scope / YAGNI

- **In scope:** GPU CREATE3 salt mining (`mine_create3` kernel + `run_create3`
  host), `verify_create3` gate, CLI flags, and the reference-math unit tests.
- **Out of scope (later, separately):** a CPU-only CREATE3 miner (CREATE2 is
  GPU-only today — keep parity), multi-factory presets beyond the default,
  non-nonce-1 / non-standard proxies, and the `metal`-gating refactor itself
  (proposed above but landed as its own change so this feature isn't blocked on a
  build-system decision).
