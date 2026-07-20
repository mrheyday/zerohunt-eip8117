# ZeroHunt

ZeroHunt mines Ethereum vanity addresses with the maximum number of **leading
zero nibbles** (e.g. `0x00000000…`). Such addresses are increasingly used by
protocols to resist [address-poisoning](https://eips.ethereum.org/EIPS/eip-8117)
attacks and to save calldata gas. ZeroHunt ships two miners:

- **`zerohunt`** — a multi-threaded **CPU** miner.
- **`zerohunt-gpu`** — a **unified CPU + GPU** miner (Apple Metal) that runs both
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
  macOS only**. The `zerohunt-gpu` miner needs a working Metal device; the
  `zerohunt` CPU miner needs only the CPU (but still builds via the macOS
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

This produces two binaries:

- `target/release/zerohunt` — CPU miner
- `target/release/zerohunt-gpu` — unified CPU + GPU miner

---

## Usage

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
cargo run --release --bin zerohunt-gpu -- 9

# or the built binary directly
./target/release/zerohunt-gpu 9
```

Example run (`zerohunt-gpu 5`):

```
zerohunt-gpu: 8 CPU workers + GPU, finding an address with 5 leading zeros
New best [CPU] 1 leading zeros: 0x05e2965cf119765e0428a7936e85277f005ed52c
New best [CPU] 2 leading zeros: 0x00a402cc0e433c0dc3bc136b7bfb359b79fa19ed
New best [CPU] 3 leading zeros: 0x000f64871583825134171ea2bae8d719cc4bb61a
New best [GPU] 4 leading zeros: 0x0₄bfa0…0d19  (0x0(4)bfa0…0d19)
New best [CPU] 5 leading zeros: 0x0₅a82e…5e5c  (0x0(5)a82e…5e5c)
Found wallet with the most leading zeros:
Address (raw):      0x00000a82e1c4e5e17d4a162016fb79189cc85e5c
Address (ERC-8117): 0x0₅a82e1c4e5e17d4a162016fb79189cc85e5c  (0x0(5)a82e1c4e5e17d4a162016fb79189cc85e5c)
Private Key: [ENCRYPTED to age recipient in scanned_keys.txt; recover offline with `zerohunt-decrypt`]
```

While running, a status line prints every ~5 seconds:

```
CPU: 0.18 Mkeys/s | GPU: 0.09 Mkeys/s | total 0.27 Mkeys/s | best: 6
```

### CPU-only miner

```bash
cargo run --release --bin zerohunt -- 9
# or
./target/release/zerohunt 9
```

The CPU tool additionally tracks **consecutive repeating characters** after the
leading zeros (e.g. `0x0000bbbb…`) as a secondary "interestingness" signal.

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
`base64(age_encrypt(private_key))` — recover it with `zerohunt-decrypt` (see
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
zerohunt-keygen > /Volumes/OFFLINE/age-identity.txt
```

The miner then encrypts automatically — it resolves the recipient from
`age-recipient.txt` in the working directory, or from `$ZEROHUNT_AGE_RECIPIENT`.
With **no** recipient configured it **fails closed** (refuses to run) rather than
writing plaintext.

**Recover keys later (on the offline machine that holds the secret):**

```bash
zerohunt-decrypt --identity /Volumes/OFFLINE/age-identity.txt scanned_keys.txt
# -> prints:  <address>	<leading_zeros>	<private_key_hex>
```

The encrypted column is standard age, so `age -d -i <identity> …` interoperates too.

**Escape hatch (insecure):** pass `--reveal` to write/print raw hex keys instead
(e.g. throwaway local testing). It is the *only* way plaintext keys are emitted,
and it prints a warning.

---

## ERC-8117 notation

[ERC-8117](https://eips.ethereum.org/EIPS/eip-8117) is a presentation-layer
standard that compresses a long run of leading zeros into a count, surfacing the
high-entropy suffix so poisoned look-alike addresses are easier to spot. It
triggers at **n ≥ 4** leading zero nibbles and defines two forms:

| Form | Example | Use |
|---|---|---|
| Subscript (Unicode) | `0x0₈abcd…1234` | wallets, explorers, UI |
| Parenthesis (ASCII) | `0x0(8)abcd…1234` | logs, terminals, bots |

It is purely visual — the underlying address bytes are unchanged. Neither
subscripts nor parentheses are valid hex, so pasting a notated address into a
legacy system fails safely instead of sending to the wrong address.

---

## How it works

For each candidate:

1. **Key derivation.** CPU: a random 32-byte key from a per-thread ChaCha CSPRNG
   (full OS entropy). GPU: `keccak256(seed ‖ counter)` with a full-entropy
   per-thread seed (a random-oracle output → full ~256-bit entropy).
2. **Public key.** secp256k1 scalar multiply `k·G`.
3. **Address.** `keccak256(x ‖ y)[12..32]` (the standard Ethereum derivation).
4. **Score.** Count leading zero nibbles; report if it beats the current best.

The GPU path is implemented in Metal Shading Language (`kernels/`):
`keccak.metal` (Keccak-256), `field.metal` (secp256k1 field arithmetic),
`ec.metal` (Jacobian scalar multiply), and `miner.metal` (the full
derive→address→count `mine` kernel). The Rust host (`src/gpu/`) dispatches
batches and **re-derives every reported hit with `k256`**, comparing all 20
address bytes, before the result is ever trusted.

Source layout:

- `src/main.rs` — `zerohunt` CPU miner
- `src/bin/gpu.rs` — `zerohunt-gpu` unified miner host loop + CLI
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

| Engine | Throughput |
|---|---|
| CPU (all cores) | ~180,000 keys/s |
| GPU (current kernel) | ~90,000 keys/s |

The GPU is currently **slower than the CPU** because `dispatch_mine` recompiles
the Metal kernel on every batch, so the GPU spends most of its time compiling
rather than mining. Caching the compiled pipeline is the top planned fix; deeper
GPU speedups (a fixed-base comb table for `k·G` and a per-thread batched modular
inverse) are specced in `docs/`. Expected effort for N leading zeros scales as
16ᴺ, so each additional zero is ~16× more work.

---

## Roadmap

- Cache the compiled Metal pipeline in `MetalContext` (biggest near-term GPU win).
- Stage 2: fixed-base comb table for `k·G`.
- Stage 3: per-thread batched modular inverse (Montgomery's trick).

See `docs/specs/2026-07-18-unified-cpu-gpu-max-leading-zero-miner-design.md`.

---

## Credits

Fork of [`Nov1kov/zerohunt`](https://github.com/Nov1kov/zerohunt), extended with
ERC-8117 anti-poisoning notation and the unified CPU + GPU (Metal) miner.

## License

Released under the [MIT License](LICENSE).
