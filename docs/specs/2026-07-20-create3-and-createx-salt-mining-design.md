# nullforge CREATE3 + CreateX vanity salt-mining — design

**Date:** 2026-07-20
**Status:** implemented (`src/miner/create3.rs`, `src/miner/createx.rs`,
`kernels/create3.metal`, `kernels/createx.metal`, `src/gpu/mod.rs`)
**Related:** `docs/specs/2026-07-17-gpu-vanity-miner-design.md` (EOA miner),
the CREATE2 mode (`src/miner/create2.rs` + `kernels/create2.metal`).

## Goal

Two keccak-only salt-mining modes that land a **contract** on a max-leading-zero
address, both **independent of the deployed init code** (unlike CREATE2, whose
address is bound to `init_code_hash`):

- **`--create3 --factory 0x..`** — a generic CREATE3 factory that CREATE2-deploys
  the standard 0xSequence/Solady proxy.
- **`--createx`** — the canonical permissionless **CreateX** factory
  (`0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed`, deployed cross-chain), which
  guards the salt before deploying.

Both emit a **public salt** (no encryption; `scanned_salts.txt`) and reuse the
CREATE2 mode's scaffold (keccak-only GPU batches, strictly-increasing best,
per-hit host re-derivation, threshold policy).

## Derivation

Shared CREATE3 core (`proxy_hash = STANDARD_CREATE3_PROXY_HASH`):

```
proxy   = keccak256(0xff ‖ deployer ‖ salt' ‖ proxy_hash)[12:]   (CREATE2 of the fixed proxy)
address = keccak256(0xd6 0x94 ‖ proxy ‖ 0x01)[12:]               (proxy CREATE at nonce 1)
```

- **`--create3`**: `deployer = factory` (CLI arg), `salt' = salt` (the mined salt
  is used directly).
- **`--createx`**: `deployer = CREATEX_ADDRESS`, `salt' = keccak256(salt)` — the
  CreateX permissionless "guarded salt" (`abi.encode(bytes32)` is the identity, so
  `guardedSalt = keccak256(salt)`). The kernel guards internally; the **emitted
  value is the ORIGINAL salt** to pass to `CreateX.deployCreate3(salt, initCode)`.

The second hop's RLP preimage is the constant 23 bytes `0xd6 0x94 ‖ proxy ‖ 0x01`
(a fresh proxy's first CREATE is nonce 1; a 20-byte address RLP-encodes as
`0x94 ‖ addr`) — no variable-length case.

## Constants (independently verified)

`STANDARD_CREATE3_PROXY_HASH = keccak256(0x67363d3d37363d34f03d5260086018f3)
= 0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`
(the 0xSequence/Solady proxy). `CREATEX_ADDRESS = 0xba5Ed0…c28ba5Ed`.

All three derivation vectors were cross-checked **twice**, independently:
against Alloy's `Address::create2`/`create` (in `tests/create3_derivation.rs`) and
against `sha3::Keccak256` (a separate library):

| vector  | inputs                                               | address               |
| ------- | ---------------------------------------------------- | --------------------- |
| create2 | deployer `0x4e59…956C`, salt `0x11..`, ich `0x22..`  | `0x95C86910…44Ab8FC3` |
| create3 | factory `0x00..42`, salt `0x33..`, standard proxy    | `0xA263972e…532C70dC` |
| createx | CreateX, `guarded=keccak256(0x55..)`, standard proxy | `0x6AB36888…9ffe6967` |

## Implementation

- **Kernels:** `kernels/create3.metal` (`mine_create3`) and `kernels/createx.metal`
  (`mine_createx` — the guarded-salt variant); keccak-only, second 23-byte hash.
- **Host (`src/gpu/mod.rs`):** `dispatch_create3` / `dispatch_createx` (→
  `Vec<Create3Hit>`) and the re-derivation gates `verify_create3` /
  `verify_createx` (the latter re-guards `keccak256(salt)` internally). Every GPU
  hit is re-derived host-side before it is trusted; a mismatch hard-aborts.
- **Drivers:** `src/miner/create3.rs` (`run_create3`) and `src/miner/createx.rs`
  (`run_createx`), mirroring `run_create2`.
- **CLI (`src/bin/gpu.rs`):** `--create3 --factory 0x..` and `--createx`,
  mutually exclusive with `--create2`.

## Correctness / testing

- **Host derivation vectors** (`tests/create3_derivation.rs`) — no GPU; pin the
  create2/create3/createx math vs Alloy (and, per the table above, sha3).
- **GPU-gated e2e** (`tests/gpu_stages.rs`, `cargo test --test gpu_stages`,
  needs a Metal device) — `create3_finds_and_verifies_low_threshold` and
  `createx_finds_and_verifies_low_threshold` mine a low threshold and assert every
  GPU hit passes the host `verify_*` gate and the claimed leading-zero nibble.

## Security / handling

Salt is **public** — no age encryption, no secp256k1, no key material. CREATE3's
init-code-independence is a deploy-time convenience, not a security property. The
only footgun is a wrong factory/proxy (mines an address you can't deploy to) —
`--createx` removes it by pinning the canonical constants.

## Build note (environment)

The crate is macOS/Metal-only (unconditional `metal` dep), so `cargo build`/`test`
run on macOS. The host derivation math (`verify_*`, `tests/create3_derivation.rs`)
is platform-independent Rust; the Metal kernels need an Apple device.
