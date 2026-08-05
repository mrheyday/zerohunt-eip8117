# Nullforge

Nullforge mines Ethereum vanity addresses with the maximum number of **leading
zero nibbles** (e.g. `0x00000000…`). Such addresses are increasingly used by
protocols to resist [address-poisoning](https://eips.ethereum.org/EIPS/eip-8117)
attacks and to save calldata gas. Nullforge ships two miners:

- **`nullforge`** — a multi-threaded **CPU** miner.
- **`nullforge-gpu`** — a **unified CPU + GPU** miner (Apple Metal) that runs both
  engines in one process against a shared best-tracker.

Results are displayed in **[ERC-8117](https://eips.ethereum.org/EIPS/eip-8117)**
"compact anti-poisoning" notation (`0x0₈abcd…1234` / `0x0(8)abcd…1234`).

> ⚠️ **This is a research/vanity tool that generates real private keys.** By
> default the key column of `scanned_keys.txt` is **encrypted to an age (X25519)
> recipient** and the decryption secret stays offline — see
> [Secure key output](#secure-key-output). Never fund an address whose key was
> produced or logged by a tool you do not fully trust.

---

## Features

- Multi-threaded CPU mining (one worker per core).
- Unified **CPU + GPU** mining in a single process: ~80% of cores run CPU
  workers while a GPU driver dispatches Metal compute batches; both funnel finds
  through one shared best-tracker.
- **Every GPU-found key is re-verified on the host against `k256`** before it is
  trusted — a mismatch hard-aborts the process. A GPU bug can waste time but can
  never emit a wrong key.
- **Strictly-increasing "new best" reporting**: each printed record has more
  leading zeros than the last, attributed to `[CPU]` or `[GPU]`.
- **ERC-8117 output** — subscript (Unicode) on screen, subscript in the log
  file (lossless), ASCII `(n)` fallback shown alongside.
- Per-engine throughput reporting (`CPU: … | GPU: … | total …`).
- Ctrl-C for a clean stop with a final summary.

---

## Requirements

- **macOS** with an Apple **Metal** GPU (developed and tested on Apple Silicon,
  M1 Pro). The crate depends on the [`metal`](https://crates.io/crates/metal)
  crate, which links `Metal.framework`, so the project currently **builds on
  macOS only**. The `nullforge-gpu` miner needs a working Metal device; the
  `nullforge` CPU miner needs only the CPU (but still builds via the macOS
  toolchain).
- **Rust** (stable, 2021 edition) — install with [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

Dependencies are fetched automatically by Cargo: `ethers` 2.0.14 (secp256k1 +
Keccak + `k256`), `rand` 0.8, `tokio`, `num_cpus`, `metal` 0.33.

---

## Build

```bash
git clone https://github.com/mrheyday/zerohunt-eip8117.git
cd zerohunt-eip8117
cargo build --release
```

This produces four binaries under `target/release/`:

- `nullforge` — CPU miner
- `nullforge-gpu` — unified CPU + GPU miner
- `nullforge-keygen` — one-time encrypted-output setup
- `nullforge-decrypt` — offline key recovery

---

## Quickstart

The complete happy path, top to bottom:

```bash
# 1. Build (see above) — produces the four binaries under target/release/.

# 2. ONE-TIME SETUP (run on a trusted machine). Generates your age identity:
#    the SECRET is printed to stdout — store it OFFLINE, it is the ONLY thing
#    that can decrypt mined keys. The PUBLIC recipient is written to
#    ./age-recipient.txt, which is all the miner needs.
./target/release/nullforge-keygen > ~/nullforge-identity.txt   # keep this file OFFLINE

# 3. MINE. Output is encrypted to that recipient by default. The argument is the
#    target number of leading-zero nibbles (default 8; valid 1–40). Ctrl-C stops.
./target/release/nullforge-gpu 8      # unified CPU + GPU (Apple Silicon)
#    …or CPU-only:
./target/release/nullforge 8

# 4. RECOVER the mined key(s) OFFLINE, using your secret identity:
./target/release/nullforge-decrypt --identity ~/nullforge-identity.txt scanned_keys.txt
#    -> prints:  <address> <TAB> <leading_zeros> <TAB> <private_key_hex>
```

> **Without step 2 the miner fails closed** — it refuses to run rather than write
> plaintext keys. For throwaway local testing only, skip the setup and pass
> `--reveal` to write raw hex instead (insecure; it prints a warning).

---

## Usage

> First-time? Do the one-time `nullforge-keygen` setup from the
> [Quickstart](#quickstart) — the miner encrypts by default and **fails closed**
> without a recipient.

Both tools take **one optional argument: the target number of leading zero
nibbles** (default `8`). The miner runs until it finds an address with at least
that many leading zeros, or until you press **Ctrl-C**. Every improvement is
printed and appended to `scanned_keys.txt` in the current directory.

The target is validated to **1–40** nibbles (an EVM address has 40 hex nibbles,
so 32 — and up to 40 — are accepted). Each additional zero is ~16× harder
(≈`16^n` keys on average), so targets above ~10–12 are astronomically infeasible
to brute-force: the miner prints a warning and will realistically run until
Ctrl-C. The extreme-leading-zero addresses you see transacting on-chain (e.g.
`0x0000…028Ae8`) are **not** found keys — they are unspendable mint/burn sinks
that a token contract merely names in its `Transfer` events; no one holds their
private key.

### Unified CPU + GPU miner (recommended on Apple Silicon)

```bash
# via cargo
cargo run --release --bin nullforge-gpu -- 9

# or the built binary directly
./target/release/nullforge-gpu 9
```

Example run (`nullforge-gpu 5`):

```
nullforge-gpu: 8 CPU workers + GPU, finding an address with 5 leading zeros
New best [CPU] 1 leading zeros: 0x05e2965cf119765e0428a7936e85277f005ed52c
New best [CPU] 2 leading zeros: 0x00a402cc0e433c0dc3bc136b7bfb359b79fa19ed
New best [CPU] 3 leading zeros: 0x000f64871583825134171ea2bae8d719cc4bb61a
New best [GPU] 4 leading zeros: 0x0₄bfa0…0d19  (0x0(4)bfa0…0d19)
New best [CPU] 5 leading zeros: 0x0₅a82e…5e5c  (0x0(5)a82e…5e5c)
Found wallet with the most leading zeros:
Address (raw):      0x00000a82e1c4e5e17d4a162016fb79189cc85e5c
Address (ERC-8117): 0x0₅a82e1c4e5e17d4a162016fb79189cc85e5c  (0x0(5)a82e1c4e5e17d4a162016fb79189cc85e5c)
Private Key: [ENCRYPTED to age recipient in scanned_keys.txt; recover offline with `nullforge-decrypt`]
```

While running, a status line prints every ~5 seconds:

```
CPU: 0.18 Mkeys/s | GPU: 0.09 Mkeys/s | total 0.27 Mkeys/s | best: 6
```

### CPU-only miner

```bash
cargo run --release --bin nullforge -- 9
# or
./target/release/nullforge 9
```

The CPU tool additionally tracks **consecutive repeating characters** after the
leading zeros (e.g. `0x0000bbbb…`) as a secondary "interestingness" signal.

### CREATE2 contract addresses (gas-optimized deploys)

For a **contract** whose _deployed address_ has leading zeros — to save calldata
gas, worth it for a contract referenced in nearly every tx — you mine a
**CREATE2 salt**, not an EOA key. This mode is **keccak-only** (no secp256k1), so
it's far faster than the EOA miner, and the result is a **public salt**: nothing
is encrypted.

```bash
nullforge-gpu 12 --create2 \
  --deployer        0x<CREATE2 factory / deployer — 20 bytes> \
  --init-code-hash  0x<keccak256(init code) of the contract — 32 bytes>
```

Each new best is printed and appended as `<total>\t<address>\t<zeros>\t<salt>`
to `scanned_salts.txt`. Deploy the contract through your CREATE2 factory with the
found salt and it lands on the vanity address:
`address = keccak256(0xff ‖ deployer ‖ salt ‖ init_code_hash)[12:]`.

Each extra leading-zero **byte** (2 nibbles) shaves ~12 gas per calldata
reference. 12 nibbles (6 bytes) ≈ 2⁴⁸ salts — roughly a day on one modern GPU;
8 nibbles (2³²) is minutes and the common cost/benefit sweet spot.

### CREATE3 contract addresses (cross-chain, bytecode-independent)

CREATE3 deploys a **constant proxy** via CREATE2, and the proxy does a nonce-1
CREATE of your contract. Because the proxy init code is fixed, the final address
depends only on **(factory, salt)** — _not_ on your contract's bytecode or
constructor args. So a mined salt yields the **same vanity address on every
chain** that has the same factory (ideal for a deterministic executor
address). This mode is also **keccak-only** (two keccaks per candidate, ~half
the CREATE2 hashrate), and the salt is **public**.

```bash
nullforge-gpu 12 --create3 \
  --factory              0x<your CREATE3 factory — 20 bytes> \
  --proxy-init-code-hash 0x<keccak256(proxy init code) — 32 bytes, factory-specific>
```

Derivation:
`proxy = keccak256(0xff ‖ factory ‖ salt ‖ proxy_init_code_hash)[12:]`, then
`address = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:]` (the proxy's nonce-1
CREATE). New bests append to `scanned_salts.txt`, same format as CREATE2.

> ⚠️ **`--proxy-init-code-hash` (alias `--proxy-hash`) is factory-specific.** It defaults to the
> Solmate/0xSequence proxy hash
> (`0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`), but
> **Solady** and **CreateX** differ, and **CreateX additionally guards the salt**
> (mixing in `msg.sender` / a redeploy flag), which changes the salt→address
> map entirely. Pass the hash for _your_ factory. Every hit is re-derived
> host-side (`verify_create3`), so a wrong hash fails loudly rather than mining
> for the wrong address.

---

## CLI reference

| Command                                                             | What it does                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nullforge [N] [--reveal]`                                          | CPU miner. `N` = target leading-zero nibbles (default `8`, valid `1–40`).                                                                                                                                                                                                |
| `nullforge-gpu [N] [--reveal]`                                      | Unified CPU + GPU miner (Apple Metal). Same args.                                                                                                                                                                                                                        |
| `nullforge-gpu [N] --create2 --deployer 0x.. --init-code-hash 0x..` | Mine a CREATE2 **salt** so a _contract_ deploys to an N-leading-zero address (keccak-only; salt is public → plaintext `scanned_salts.txt`).                                                                                                                              |
| `nullforge-gpu [N] --create3 --factory 0x.. [--proxy-hash 0x..]`    | Mine a CREATE3 **salt** so a _contract_ deploys to an N-leading-zero, **bytecode-independent** address (same on every chain with that factory). Proxy hash defaults to Solmate/0xSequence; override (`--proxy-hash`, alias `--proxy-init-code-hash`) for Solady/CreateX. |
| `nullforge-gpu [N] --createx`                                       | Mine a **CreateX** permissionless CREATE3 salt (deployer + proxy hash are the fixed canonical CreateX constants; the salt is guarded as `keccak256(salt)` in-kernel). Pass the recorded salt straight to `CreateX.deployCreate3(salt, initCode)`.                        |
| `nullforge-keygen`                                                  | One-time setup: prints the SECRET age identity to **stdout**, writes the PUBLIC recipient to `age-recipient.txt`.                                                                                                                                                        |
| `nullforge-decrypt -i <identity> [file]`                            | Offline recovery: decrypts the key column of `scanned_keys.txt` (default `file`) with your secret identity.                                                                                                                                                              |

**Flags**

- `--reveal` — write/print raw **plaintext** hex keys instead of encrypting.
  Insecure; the only way plaintext is ever emitted, and it prints a warning.

**Environment variables**

| Variable                      | Effect                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `NULLFORGE_AGE_RECIPIENT`     | `age1…` recipient the miner encrypts to (takes precedence over `age-recipient.txt`). |
| `NULLFORGE_AGE_IDENTITY_FILE` | Default secret-identity path for `nullforge-decrypt` (instead of `-i`).              |

Each extra leading zero is ~16× more work, so targets above ~10–12 nibbles are
astronomically infeasible to brute-force — the miner prints a warning and runs
until Ctrl-C.

---

## Output

### Console

- `New best [CPU|GPU] N leading zeros: <address>` — one line per improvement.
- Addresses with **≥ 4** leading zero nibbles are shown in ERC-8117 notation
  (subscript + ASCII fallback, abbreviated `first4…last4`); fewer than 4 are
  shown as raw hex.
- A final summary prints the winning address (raw **and** ERC-8117) and its
  **private key**.

### `scanned_keys.txt` (log file, tab-separated)

```
<total_keys_tried>	<address>	<leading_zeros>	<key_column>
```

The address column uses **ERC-8117 subscript, non-truncated** for ≥ 4 leading
zeros (this is lossless — `0x0₈` + the remainder reconstructs the full address);
raw hex below the threshold. The **key column** is, by default,
`base64(age_encrypt(private_key))` — recover it with `nullforge-decrypt` (see
[Secure key output](#secure-key-output)). Only with `--reveal` is it raw hex.

> ⚠️ **The address column is public; the key column is age ciphertext at rest.**
> `scanned_keys.txt` is `.gitignore`d regardless. Under `--reveal` the key column
> is PLAINTEXT — treat the whole file as a secret in that mode.

---

## Secure key output

By default, every found private key is **encrypted to an age (X25519) recipient**
before it is written — plaintext keys never touch disk or the console. The
decryption secret is generated once and kept **offline**; the mining box only
ever holds the public recipient, so a compromised box (or a leaked
`scanned_keys.txt`) yields only ciphertext.

**One-time setup (on a trusted machine):**

```bash
# Writes the PUBLIC recipient to age-recipient.txt (the miner reads this) and
# prints the SECRET identity to stdout — save it OFFLINE (password manager /
# hardware / paper), e.g. straight onto removable media:
nullforge-keygen > /Volumes/OFFLINE/age-identity.txt
```

The miner then encrypts automatically — it resolves the recipient from
`age-recipient.txt` in the working directory, or from `$NULLFORGE_AGE_RECIPIENT`.
With **no** recipient configured it **fails closed** (refuses to run) rather than
writing plaintext.

**Recover keys later (on the offline machine that holds the secret):**

```bash
nullforge-decrypt --identity /Volumes/OFFLINE/age-identity.txt scanned_keys.txt
# -> prints:  <address>	<leading_zeros>	<private_key_hex>
```

The encrypted column is standard age, so `age -d -i <identity> …` interoperates too.

**Escape hatch (insecure):** pass `--reveal` to write/print raw hex keys instead
(e.g. throwaway local testing). It is the _only_ way plaintext keys are emitted,
and it prints a warning.

---

## ERC-8117 notation

[ERC-8117](https://eips.ethereum.org/EIPS/eip-8117) is a presentation-layer
standard that compresses a long run of leading zeros into a count, surfacing the
high-entropy suffix so poisoned look-alike addresses are easier to spot. It
triggers at **n ≥ 4** leading zero nibbles and defines two forms:

| Form                | Example           | Use                    |
| ------------------- | ----------------- | ---------------------- |
| Subscript (Unicode) | `0x0₈abcd…1234`   | wallets, explorers, UI |
| Parenthesis (ASCII) | `0x0(8)abcd…1234` | logs, terminals, bots  |

It is purely visual — the underlying address bytes are unchanged. Neither
subscripts nor parentheses are valid hex, so pasting a notated address into a
legacy system fails safely instead of sending to the wrong address.

---

## How it works

For each candidate:

1. **Key derivation.** CPU: a random 32-byte key from a per-thread ChaCha CSPRNG
   (full OS entropy). GPU: one `keccak256(seed ‖ base_counter)` per thread per
   batch (a full-entropy random-oracle output), then each candidate key in
   that batch is `(base + offset) mod n` for `offset` in the batch window —
   see "Approach B" below.
2. **Public key.** One secp256k1 scalar multiply `base·G` per thread per
   batch, then a cheap Jacobian point addition (`+G`) per subsequent candidate
   in the batch — not a full scalar multiply per key.
3. **Address.** `keccak256(x ‖ y)[12..32]` (the standard Ethereum derivation).
4. **Score.** Count leading zero nibbles; report if it beats the current best.

The GPU path is implemented in Metal Shading Language (`kernels/`):
`keccak.metal` (Keccak-256), `field.metal` (secp256k1 field arithmetic),
`ec.metal` (Jacobian scalar multiply/add), and `miner.metal` (the
derive→address→count kernels, including the default `mine_incremental`
kernel and the reference/fallback `mine` kernel). The Rust host (`src/gpu/`)
dispatches batches and **re-derives every reported hit with `k256`**,
comparing all 20 address bytes, before the result is ever trusted.

**Approach B (incremental EC, default since `mine_incremental`).** Rather than
an independent full scalar multiply per candidate key (Approach A, still
available as `mine`/`dispatch_mine`), each GPU thread does one scalar multiply
per _batch_ and walks the rest of the batch via point additions — about 15x
fewer field operations per candidate. This is a different, and weaker,
security model than Approach A's "every key an independent oracle output":
keys within one batch are affinely related (their difference is public), so
recovering one doesn't help without solving discrete log, but the
correlation set is bounded to a batch instead of empty. See
`docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` for the full
cost model and security-model writeup — this codebase exists because a
previous vanity miner (`profanity`) shipped a weak model without disclosing
it, so this tradeoff is documented rather than left implicit.

Source layout:

- `src/main.rs` — `nullforge` CPU miner
- `src/bin/gpu.rs` — `nullforge-gpu` unified miner host loop + CLI
- `src/miner/` — shared state + `report_hit` funnel, CPU worker, GPU driver
- `src/erc8117.rs` — ERC-8117 formatter
- `src/keyenc.rs` — key-output encryption (age); `src/bin/keygen.rs` +
  `src/bin/decrypt.rs` — offline identity setup + recovery
- `src/gpu/` + `kernels/` — Metal compute host + kernels
- `docs/specs/`, `docs/plans/` — design spec and implementation plan

---

## Testing

```bash
cargo test --lib               # fast unit tests (ERC-8117, miner funnel, helpers)
cargo test --test gpu_stages   # GPU pipeline tests — require a Metal device
```

The `gpu_stages` integration tests each open a Metal device and compare GPU
output against `ethers`/`k256` computed at test time (no hard-coded crypto
vectors).

---

## Performance & limitations

Measured on an Apple M1 Pro (10 CPU cores, 16-core GPU), release build:

| Engine          | Throughput         |
| --------------- | ------------------ |
| CPU (all cores) | ~180,000 keys/s    |
| GPU (Metal)     | ~25,000,000 keys/s |

The GPU is now the **dominant engine** (~140× the CPU), after two host-side fixes:
the `mine` compute pipeline is compiled **once** and reused (it was recompiled on
every batch), and dispatch uses `dispatch_threads` with a full-width threadgroup
instead of one thread per threadgroup (which idled ~32× of the GPU's SIMD width).
Deeper GPU speedups (a fixed-base comb table for `k·G` and a per-thread batched
modular inverse) are specced in `docs/`. Expected effort for N leading zeros
scales as 16ᴺ, so each additional zero is ~16× more work.

---

## Roadmap

- ~~Cache the compiled Metal pipeline in `MetalContext`~~ · ~~fix threadgroup
  occupancy (`dispatch_threads`)~~ — **done** (GPU ~0.09 → ~25 Mkeys/s).
- Stage 2: fixed-base comb table for `k·G`.
- Stage 3: per-thread batched modular inverse (Montgomery's trick).

See `docs/specs/2026-07-18-unified-cpu-gpu-max-leading-zero-miner-design.md`.

---

## Credits

Fork of [`Nov1kov/zerohunt`](https://github.com/Nov1kov/zerohunt), extended with
ERC-8117 anti-poisoning notation and the unified CPU + GPU (Metal) miner.

## License

Released under the [MIT License](LICENSE).
