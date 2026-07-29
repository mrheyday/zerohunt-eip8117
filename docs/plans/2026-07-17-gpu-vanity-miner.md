# GPU Vanity Miner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Metal-GPU Ethereum vanity-address miner (`nullforge-gpu`) that searches for leading-zero addresses far faster than the CPU tool, emitting only full-entropy, CPU-verified private keys.

**Architecture:** A new Rust binary drives Apple Metal compute. GPU threads each derive full-entropy keys (`Keccak256(thread_seed‖counter)`), run secp256k1 scalar-mult + Keccak-256 to derive the address, and report hits; the host re-verifies every hit against `k256` before trusting it. Built in stages behind test gates: Metal harness → Keccak kernel → field arithmetic → EC scalar-mult → full pipeline → miner loop.

**Tech Stack:** Rust, `metal` crate v0.33 (Metal API bindings), MSL (Metal Shading Language) kernels compiled at runtime, `ethers`/`k256` (host reference oracle), `rand` 0.8 (`OsRng`).

## Global Constraints

- Target hardware: Apple Silicon (M1 Pro), Metal 4. macOS only.
- `rand = "0.8"` (pinned — matches k256 0.13's rand_core 0.6 bound; do NOT bump).
- The CPU tool (`src/main.rs`) must remain untouched and buildable.
- Every GPU-reported hit MUST be re-derived on the host via `k256` and matched (address + zero-count) before being saved/printed. Any mismatch = hard abort.
- Private-key scalar guard: skip candidates where `privkey == 0` or `privkey >= n` (secp256k1 order) so stored keys stay canonical.
- Tests compare GPU output against a host-library reference (`ethers::utils::keccak256`, `k256`) computed at test time — do not hardcode crypto vectors.
- secp256k1 constants (verbatim, big-endian):
  - `p  = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`
  - `n  = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141`
  - `Gx = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798`
  - `Gy = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B4489A68554199C47D08FFB10D4B8`

## File Structure

- Create `src/bin/gpu.rs` — the GPU binary: host harness, Metal setup, dispatch loop, correctness gate, CLI. Grows across tasks.
- Create `src/gpu/mod.rs` — library module shared by the binary and integration tests: `MetalContext` (device/queue/pipeline), buffer helpers, dispatch wrappers, host reference helpers. Declared from a new `src/lib.rs`.
- Create `src/lib.rs` — exposes `pub mod gpu;` so integration tests can drive the Metal code. (Does not affect the existing `main.rs` binary.)
- Create `kernels/keccak.metal`, `kernels/field.metal`, `kernels/ec.metal`, `kernels/miner.metal` — MSL source, embedded via `include_str!`. Later kernels `#include` earlier ones by concatenation in the host (Metal runtime compile takes one combined source string).
- Create `tests/gpu_stages.rs` — integration tests for each stage (they need the Metal device, so they live at integration-test level, run via `cargo test`).
- Modify `Cargo.toml` — add `[lib]`, the `[[bin]] name = "nullforge-gpu"`, and the `metal` dependency.

---

### Task 1: Binary scaffold + Metal harness (prove the plumbing)

**Files:**

- Modify: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/gpu/mod.rs`
- Create: `src/bin/gpu.rs`
- Create: `kernels/echo.metal`
- Create: `tests/gpu_stages.rs`

**Interfaces:**

- Produces:
  - `gpu::MetalContext::new() -> MetalContext` — holds `device: metal::Device`, `queue: metal::CommandQueue`.
  - `MetalContext::run_u32_kernel(&self, src: &str, entry: &str, out_len: usize, tgroups: u64, tperg: u64) -> Vec<u32>` — compiles `src`, dispatches `entry` over `tgroups*tperg` threads, returns the `out_len`-element `u32` output buffer.

- [ ] **Step 1: Add deps and targets to `Cargo.toml`**

```toml
[lib]
name = "nullforge"
path = "src/lib.rs"

[[bin]]
name = "nullforge"
path = "src/main.rs"

[[bin]]
name = "nullforge-gpu"
path = "src/bin/gpu.rs"

[dependencies]
# (existing deps unchanged: rand = "0.8", ethers, tokio, num_cpus)
metal = "0.33.0"
```

- [ ] **Step 2: Write the echo kernel**

`kernels/echo.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void echo(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = gid * 2u;
}
```

- [ ] **Step 3: Write the failing test**

`tests/gpu_stages.rs`:

```rust
use nullforge::gpu::MetalContext;

#[test]
fn echo_kernel_doubles_thread_id() {
    let ctx = MetalContext::new();
    let src = include_str!("../kernels/echo.metal");
    let out = ctx.run_u32_kernel(src, "echo", 256, 4, 64);
    assert_eq!(out.len(), 256);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, (i as u32) * 2, "thread {i} wrong");
    }
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test --test gpu_stages echo_kernel_doubles_thread_id`
Expected: FAIL to compile — `nullforge::gpu` / `MetalContext` not defined.

- [ ] **Step 5: Implement `src/lib.rs` and `src/gpu/mod.rs`**

`src/lib.rs`:

```rust
pub mod gpu;
```

`src/gpu/mod.rs`:

```rust
use metal::{Device, CommandQueue, MTLResourceOptions, MTLSize, CompileOptions};

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
}

impl MetalContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        Self { device, queue }
    }

    /// Compile `src`, dispatch `entry` over `tgroups*tperg` threads writing a
    /// `u32` output buffer of `out_len` elements, and return its contents.
    pub fn run_u32_kernel(
        &self,
        src: &str,
        entry: &str,
        out_len: usize,
        tgroups: u64,
        tperg: u64,
    ) -> Vec<u32> {
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function(entry, None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let bytes = (out_len * std::mem::size_of::<u32>()) as u64;
        let out_buf = self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&out_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(tgroups, 1, 1),
            MTLSize::new(tperg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec()
    }
}

impl Default for MetalContext {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 6: Create a minimal `src/bin/gpu.rs` so the target builds**

```rust
fn main() {
    println!("nullforge-gpu: harness placeholder (see plan tasks)");
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cargo test --test gpu_stages echo_kernel_doubles_thread_id`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/gpu/mod.rs src/bin/gpu.rs kernels/echo.metal tests/gpu_stages.rs
git commit -m "feat(gpu): Metal compute harness + echo-kernel gate"
```

---

### Task 2: Keccak-256 MSL kernel (verified vs `ethers::utils::keccak256`)

**Files:**

- Create: `kernels/keccak.metal`
- Modify: `src/gpu/mod.rs` (add `run_keccak(&self, inputs: &[[u8; N]]) -> Vec<[u8;32]>` byte-buffer dispatch helper)
- Modify: `tests/gpu_stages.rs`

**Interfaces:**

- Consumes: `MetalContext` (Task 1).
- Produces:
  - MSL device function `void keccak256(thread const uchar* in, uint inlen, thread uchar* out32)` in `kernels/keccak.metal`.
  - MSL kernel `keccak_test(device const uchar* inputs, device const uint* lens, device uchar* out, uint gid)` — hashes the `gid`-th input (fixed max stride, e.g. 64 bytes) into `out[gid*32 .. +32]`.
  - `MetalContext::run_keccak_fixed64(&self, inputs: &[Vec<u8>]) -> Vec<[u8;32]>`.

- [ ] **Step 1: Write the failing test** (GPU Keccak == host Keccak for empty, "abc", and 64 random bytes)

Add to `tests/gpu_stages.rs`:

```rust
#[test]
fn keccak_matches_host_reference() {
    use ethers::utils::keccak256;
    let ctx = MetalContext::new();
    let inputs: Vec<Vec<u8>> = vec![
        vec![],
        b"abc".to_vec(),
        (0u8..64).collect(),
    ];
    let gpu = ctx.run_keccak_fixed64(&inputs);
    for (i, inp) in inputs.iter().enumerate() {
        assert_eq!(gpu[i], keccak256(inp), "keccak mismatch on input {i}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test gpu_stages keccak_matches_host_reference`
Expected: FAIL — `run_keccak_fixed64` not defined.

- [ ] **Step 3: Write `kernels/keccak.metal`**

Implement Keccak-f[1600] (the Ethereum Keccak-256: rate 1088 bits / 136 bytes, capacity 512, `0x01` domain-suffix pad — NOT the `0x06` SHA3 pad, output 32 bytes). Provide the 24-round permutation with the standard RC[24] round constants and rho-offsets. Device function signature `keccak256(thread const uchar* in, uint inlen, thread uchar* out32)` for `inlen <= 135` (single block; sufficient for all uses here: 32-byte seed+8-byte counter = 40 bytes, and 64-byte pubkey). The `keccak_test` kernel copies the `gid`-th 64-byte-strided input + its length into thread memory and calls it.

The round constants (paste verbatim into the kernel):

```metal
constant ulong RC[24] = {
  0x0000000000000001UL,0x0000000000008082UL,0x800000000000808aUL,0x8000000080008000UL,
  0x000000000000808bUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
  0x000000000000008aUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000aUL,
  0x000000008000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,
  0x8000000000008002UL,0x8000000000000080UL,0x000000000000800aUL,0x800000008000000aUL,
  0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL
};
```

(Standard theta/rho/pi/chi/iota over a `ulong state[25]`; absorb `inlen` bytes little-endian into the state, XOR `0x01` at byte `inlen` and `0x80` at byte 135, permute once, squeeze first 32 bytes little-endian.)

- [ ] **Step 4: Add `run_keccak_fixed64` to `src/gpu/mod.rs`**

```rust
use metal::MTLSize;

impl MetalContext {
    pub fn run_keccak_fixed64(&self, inputs: &[Vec<u8>]) -> Vec<[u8; 32]> {
        const STRIDE: usize = 64;
        let n = inputs.len();
        let mut flat = vec![0u8; n * STRIDE];
        let mut lens = vec![0u32; n];
        for (i, inp) in inputs.iter().enumerate() {
            assert!(inp.len() <= STRIDE);
            flat[i * STRIDE..i * STRIDE + inp.len()].copy_from_slice(inp);
            lens[i] = inp.len() as u32;
        }
        let src = format!("{}\n{}", include_str!("../../kernels/keccak.metal"), KECCAK_TEST_ENTRY);
        // dispatch keccak_test with 3 buffers (inputs, lens, out); helper below
        self.dispatch_keccak(&src, &flat, &lens, n)
    }
}
```

Implement `dispatch_keccak` mirroring `run_u32_kernel` but binding three shared buffers (`inputs` u8, `lens` u32, `out` u8[n*32]) and dispatching `n` threads; return `Vec<[u8;32]>`. `KECCAK_TEST_ENTRY` is the `kernel void keccak_test(...)` wrapper string (or place it directly in `keccak.metal`).

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --test gpu_stages keccak_matches_host_reference`
Expected: PASS (all three inputs match `ethers` keccak256).

- [ ] **Step 6: Commit**

```bash
git add kernels/keccak.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): Keccak-256 MSL kernel verified vs host reference"
```

---

### Task 3: secp256k1 field arithmetic MSL (mod p: add/sub/mul/inv)

**Files:**

- Create: `kernels/field.metal`
- Modify: `src/gpu/mod.rs` (add `run_field_op` helper dispatching a[256] op b[256] -> out[256])
- Modify: `tests/gpu_stages.rs`

**Interfaces:**

- Produces MSL device functions over `typedef struct { uint v[8]; } fe;` (8×u32 limbs, little-endian limb order, value mod `p`):
  - `fe fe_add(fe a, fe b)`, `fe fe_sub(fe a, fe b)`, `fe fe_mul(fe a, fe b)`, `fe fe_inv(fe a)` (Fermat: `a^(p-2) mod p`).
  - `kernel void field_test(device const uint* a, device const uint* b, device uint* out, device const uint* op, uint gid)` — applies `op` (0=add,1=sub,2=mul,3=inv) to the `gid`-th 8-limb operand pair.
- Consumes: nothing from Task 2 (independent), uses `p` constant.

- [ ] **Step 1: Write the failing test** (GPU field ops == host big-int mod p, computed with a tiny host helper using `k256`'s `U256`/`ethers::types::U256`)

Add to `tests/gpu_stages.rs` a host reference using `ethers::types::U256` for add/sub/mul mod p and a modpow for inverse:

```rust
#[test]
fn field_ops_match_host_mod_p() {
    use ethers::types::U256;
    let p = U256::from_str_radix(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap();
    let ctx = MetalContext::new();
    // deterministic test operands (avoid rng in the assert)
    let cases: &[(&str,&str)] = &[
        ("2","3"),
        ("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E","5"), // p-1, 5
        ("ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789","1234567890ABCDEF"),
    ];
    for (ah, bh) in cases {
        let a = U256::from_str_radix(ah,16).unwrap() % p;
        let b = U256::from_str_radix(bh,16).unwrap() % p;
        let add = a.overflowing_add(b).0 % p;
        let sub = (a + p - b) % p;
        let mul = mulmod(a, b, p);          // helper: (a*b) mod p via U512-ish or repeated
        let inv = modpow(a, p - 2u32.into(), p); // Fermat inverse
        assert_eq!(ctx.run_field(a, b, 0), add, "add {ah}");
        assert_eq!(ctx.run_field(a, b, 1), sub, "sub {ah}");
        assert_eq!(ctx.run_field(a, b, 2), mul, "mul {ah}");
        assert_eq!(ctx.run_field(a, a, 3), inv, "inv {ah}");
    }
}
```

Include `mulmod`/`modpow`/`run_field` (U256<->[u32;8] LE marshalling) as test-module helpers. `run_field(a,b,op) -> U256` dispatches `field_test` for one operand pair and reads back 8 limbs.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test gpu_stages field_ops_match_host_mod_p`
Expected: FAIL — `run_field`/kernel absent.

- [ ] **Step 3: Write `kernels/field.metal`**

Implement 8×u32-limb arithmetic mod `p`:

- `p` as `constant uint P[8]` (little-endian limbs: `{0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF}`).
- `fe_add`: schoolbook add with carry (`uint`, detect carry), then conditional subtract `P` if `>= P`.
- `fe_sub`: add `P` then subtract `b` with borrow (keeps non-negative), conditional subtract `P`.
- `fe_mul`: 8×8 schoolbook into 16 limbs using `uint` products with `mulhi(a,b)` for the high word, then reduce mod `p`. Reduce via the secp256k1 fast reduction (using `p = 2^256 - 2^32 - 977`) OR generic Barrett/repeated-subtract; fast reduction preferred (fold the top 256 bits: `t = hi * 0x1000003D1` add to lo, twice). Provide the fold: for 16-limb product `P[0..15]`, compute `c = P[8..15]`, `acc = P[0..7] + c * 0x1000003D1` (a 256×32→ add), carry-fold once more, final conditional subtract.
- `fe_inv`: Fermat `a^(p-2)` via square-and-multiply over the fixed exponent `p-2` (hardcode the exponent bits or loop its 256 bits from a `constant uint PM2[8]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test gpu_stages field_ops_match_host_mod_p`
Expected: PASS for all cases incl. inverse.

- [ ] **Step 5: Commit**

```bash
git add kernels/field.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): secp256k1 field arithmetic MSL verified mod p"
```

---

### Task 4: secp256k1 EC scalar-mult MSL (privkey·G == k256 pubkey)

**Files:**

- Create: `kernels/ec.metal`
- Modify: `src/gpu/mod.rs` (add `run_scalarmul(&self, keys: &[[u8;32]]) -> Vec<[u8;64]>` returning affine x‖y)
- Modify: `tests/gpu_stages.rs`

**Interfaces:**

- Consumes: `fe`/`fe_*` from Task 3 (concatenated source), `P`/constants.
- Produces MSL:
  - `struct jpoint { fe X, Y, Z; }`, `jpoint j_double(jpoint)`, `jpoint j_add(jpoint, jpoint)` (Jacobian), `void scalarmul(fe k, thread fe& outx, thread fe& outy)` (double-and-add over `G`, final `fe_inv(Z)`→affine).
  - `kernel void ec_test(device const uint* keys, device uint* outxy, uint gid)`.
- `MetalContext::run_scalarmul(&self, keys: &[[u8;32]]) -> Vec<[u8;64]>`.

- [ ] **Step 1: Write the failing test** (GPU `k·G` affine == `k256` public key, uncompressed x‖y)

```rust
#[test]
fn scalarmul_matches_k256_pubkey() {
    use ethers::core::k256::ecdsa::SigningKey;
    use ethers::core::k256::elliptic_curve::sec1::ToEncodedPoint;
    let ctx = MetalContext::new();
    // deterministic keys incl. edges: 1, 2, and a fixed 32-byte value
    let mut keys: Vec<[u8;32]> = vec![[0u8;32], [0u8;32], [0u8;32]];
    keys[0][31] = 1;
    keys[1][31] = 2;
    keys[2] = hex_to_32("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    let gpu = ctx.run_scalarmul(&keys);
    for (i, k) in keys.iter().enumerate() {
        let sk = SigningKey::from_bytes(k.into()).unwrap();
        let pt = sk.verifying_key().to_encoded_point(false); // 0x04 ‖ x(32) ‖ y(32)
        let want = &pt.as_bytes()[1..65];
        assert_eq!(&gpu[i][..], want, "pubkey mismatch key {i}");
    }
}
```

(`hex_to_32` test helper.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test gpu_stages scalarmul_matches_k256_pubkey`
Expected: FAIL — `run_scalarmul`/kernel absent.

- [ ] **Step 3: Write `kernels/ec.metal`**

- `constant fe GX`, `constant fe GY` (the `Gx`/`Gy` constants as LE limbs).
- Jacobian double (`dbl-2009-l`) and add (`add-2007-bl`) formulas over `fe_*`.
- `scalarmul`: init `R = point-at-infinity` (`Z=0`), base `G` in Jacobian (`Z=1`); MSB→LSB over the 256-bit `k`, `R = j_double(R)`, and if bit set `R = j_add(R, G)`. After the loop: `zinv = fe_inv(R.Z)`, `zinv2 = fe_mul(zinv,zinv)`, `outx = fe_mul(R.X, zinv2)`, `outy = fe_mul(R.Y, fe_mul(zinv2, zinv))`. Output x,y as 32 big-endian bytes each (marshal LE-limbs → BE bytes to match k256's encoding).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test gpu_stages scalarmul_matches_k256_pubkey`
Expected: PASS for keys 1, 2, and the fixed value.

- [ ] **Step 5: Commit**

```bash
git add kernels/ec.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): secp256k1 Jacobian scalar-mult verified vs k256"
```

---

### Task 5: Full derive→address pipeline kernel + host correctness gate

**Files:**

- Create: `kernels/miner.metal`
- Modify: `src/gpu/mod.rs` (add `derive_address_gpu(&self, seeds:&[[u8;32]], counters:&[u64]) -> Vec<([u8;32],[u8;20])>` for the gate test; and `verify_hit(privkey,address) -> bool` host helper)
- Modify: `tests/gpu_stages.rs`

**Interfaces:**

- Consumes: keccak (T2), field (T3), ec (T4) — all four `.metal` files concatenated into one compile unit.
- Produces MSL:
  - device `fe derive_privkey(thread const uchar seed[32], ulong counter)` = `keccak256(seed‖counter_le8)` as `fe` scalar; caller applies the range guard.
  - device `void pubkey_to_address(fe x, fe y, thread uchar out20[20])` = `keccak256(x_be‖y_be)[12..32]`.
  - `kernel void derive_test(device const uchar* seeds, device const ulong* counters, device uchar* out_priv, device uchar* out_addr, uint gid)`.
- Host: `verify_hit(&self, privkey:[u8;32], address:[u8;20]) -> bool` re-derives via `k256`+`ethers::utils::get_contract_address`-style: `secret_key_to_address` equivalent, compares 20 bytes.

- [ ] **Step 1: Write the failing test** (GPU `(seed,counter) → (privkey,address)` matches host `k256` derivation, with the scalar guard)

```rust
#[test]
fn pipeline_privkey_and_address_match_host() {
    use ethers::core::k256::ecdsa::SigningKey;
    use ethers::utils::secret_key_to_address;
    let ctx = MetalContext::new();
    let seeds: Vec<[u8;32]> = (0..8).map(|i| { let mut s=[0u8;32]; s[0]=i as u8; s[31]=0xA5; s }).collect();
    let counters: Vec<u64> = (0..8).collect();
    let out = ctx.derive_address_gpu(&seeds, &counters);
    for (i, (priv_k, addr)) in out.iter().enumerate() {
        // host: privkey = keccak(seed‖counter_le); guard skips 0/>=n (won't hit here)
        let sk = SigningKey::from_bytes(priv_k.into()).expect("canonical key");
        let want = secret_key_to_address(&sk);
        assert_eq!(&addr[..], want.as_bytes(), "address mismatch idx {i}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test gpu_stages pipeline_privkey_and_address_match_host`
Expected: FAIL — `derive_address_gpu` absent.

- [ ] **Step 3: Write `kernels/miner.metal`** (derive + address, using T2–T4 device functions) and `derive_address_gpu` host dispatch (marshal seeds[32]+counters[u64] in, priv[32]+addr[20] out). Combine the four kernel sources in the host: `format!("{}\n{}\n{}\n{}", keccak, field, ec, miner)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test gpu_stages pipeline_privkey_and_address_match_host`
Expected: PASS — GPU addresses equal host `secret_key_to_address`.

- [ ] **Step 5: Commit**

```bash
git add kernels/miner.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): full derive->address pipeline verified vs k256"
```

---

### Task 6: Miner host loop + CLI (the working tool)

**Files:**

- Modify: `src/bin/gpu.rs` (full implementation)
- Modify: `kernels/miner.metal` (add the searching `kernel void mine(...)` with zero-count + atomic hit output)
- Modify: `src/gpu/mod.rs` (add `dispatch_mine(...) -> Vec<Hit>`)
- Modify: `tests/gpu_stages.rs` (add a low-threshold end-to-end test)

**Interfaces:**

- Consumes: everything above.
- Produces:
  - MSL `kernel void mine(device const uchar* seeds, device const ulong* base_counters, constant uint& iters, constant uint& threshold, device atomic_uint* hit_count, device uchar* hits, uint gid)` — each thread loops `iters`: derive (with 0/`>=n` guard), scalar-mult, address, count leading-zero nibbles; on `>= threshold` atomically append `{priv[32],addr[20],zeros(1),pad}` to `hits` (bounded capacity, e.g. 1024).
  - Host `struct Hit { privkey:[u8;32], address:[u8;20], zeros:u8 }`.
  - `MetalContext::dispatch_mine(&self, seeds:&[[u8;32]], base_counters:&[u64], iters:u32, threshold:u32) -> Vec<Hit>`.

- [ ] **Step 1: Write the failing end-to-end test** (threshold 2 finds verified hits quickly)

```rust
#[test]
fn mine_finds_and_verifies_low_threshold() {
    let ctx = MetalContext::new();
    let seeds: Vec<[u8;32]> = (0..256).map(|i| { let mut s=[0u8;32]; s[0]=(i&0xff) as u8; s[1]=(i>>8) as u8; s[31]=0x11; s }).collect();
    let base: Vec<u64> = vec![0; seeds.len()];
    let hits = ctx.dispatch_mine(&seeds, &base, 4096, 2); // >=2 leading zero nibbles
    assert!(!hits.is_empty(), "should find >=2-zero addresses");
    for h in &hits {
        assert!(ctx.verify_hit(h.privkey, h.address), "hit failed host re-derivation");
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test gpu_stages mine_finds_and_verifies_low_threshold`
Expected: FAIL — `dispatch_mine`/`mine` kernel absent.

- [ ] **Step 3: Implement `mine` kernel + `dispatch_mine`** (atomic append via `atomic_fetch_add_explicit`; bounded `hits` buffer; leading-zero-nibble count matching `src/main.rs` semantics: byte==0 → +2 else `clz(byte)/4` on the top nibble then stop).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test gpu_stages mine_finds_and_verifies_low_threshold`
Expected: PASS — hits found and every one re-verified on host.

- [ ] **Step 5: Implement `src/bin/gpu.rs` (the real CLI + loop)**

```rust
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use rand::rngs::OsRng;
use rand::RngCore;
use nullforge::gpu::{MetalContext, Hit};

fn main() {
    let target: u32 = match env::args().nth(1) {
        None => 8,
        Some(a) => match a.trim().parse() { Ok(n) => n, Err(_) => {
            eprintln!("Invalid leading-zero count: {a:?}\nUsage: nullforge-gpu [max_zeros] (default 8)");
            std::process::exit(2);
        }}
    };
    let ctx = MetalContext::new();
    let n_threads = 4096usize; // tuned per GPU; documented
    // full-entropy per-thread seeds (profanity-safe)
    let mut seeds = vec![[0u8;32]; n_threads];
    for s in seeds.iter_mut() { OsRng.fill_bytes(s); }
    let mut counters = vec![0u64; n_threads];

    let stop = Arc::new(AtomicBool::new(false));
    { let s = stop.clone(); ctrlc_like(s); } // install Ctrl-C via std or a small handler

    let start = Instant::now();
    let mut best = 0u32;
    let mut total: u64 = 0;
    let iters_per_dispatch = 8192u32;
    while !stop.load(Ordering::Relaxed) {
        let hits = ctx.dispatch_mine(&seeds, &counters, iters_per_dispatch, best.max(3));
        for c in counters.iter_mut() { *c += iters_per_dispatch as u64; }
        total += n_threads as u64 * iters_per_dispatch as u64;
        for h in hits {
            if !ctx.verify_hit(h.privkey, h.address) {
                eprintln!("FATAL: GPU hit failed host verification — kernel bug, aborting");
                std::process::exit(3);
            }
            if h.zeros as u32 >= best {
                best = h.zeros as u32;
                report(&h); // print + append scanned_keys.txt (same format as CPU tool)
                if best >= target { stop.store(true, Ordering::Relaxed); }
            }
        }
        if start.elapsed().as_secs() % 20 == 0 {
            println!("rate: {:.0} keys/sec", total as f64 / start.elapsed().as_secs_f64().max(1.0));
        }
    }
}
```

(Provide `report`, `ctrlc_like`, and the `scanned_keys.txt` writer mirroring `src/main.rs`'s format. Re-seed periodically is unnecessary — counters advance the domain; seeds are already full-entropy.)

- [ ] **Step 6: Build + smoke-run**

Run: `cargo build --release --bin nullforge-gpu && ./target/release/nullforge-gpu 5`
Expected: prints threads/rate, finds ≥5-zero addresses, each verified; Ctrl-C stops cleanly.

- [ ] **Step 7: Commit**

```bash
git add src/bin/gpu.rs kernels/miner.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): mining loop + CLI with per-hit host verification"
```

---

## Self-Review

**Spec coverage:** §1 architecture → Task 1 (+ Cargo targets). §2 security (full-entropy seeds, Keccak-derive, scalar guard) → Task 5/6 (seeds from `OsRng`, guard in kernel). §3 GPU compute (derive, scalar-mult, Keccak, zero-count) → Tasks 2–6. §4 host loop + correctness gate → Task 6 (`verify_hit`, hard-abort). §5 staged testing → Tasks 1–6 each gate. Scope/YAGNI (no B, host-side ranking) → honored (no repeating-char logic; ranking omitted from MVP — acceptable per spec "leading-zero search + verified keys"). Scalar range guard → Global Constraints + Task 5/6.

**Placeholder scan:** MSL bodies for Keccak/field/EC are specified by algorithm + exact constants + the exact host oracle each must match (not "TODO") — the test is the completion criterion. No "TBD"/"implement later". Acceptable: the crypto kernels are defined by their verified-against-host tests, which are shown in full.

**Type consistency:** `fe` = `{uint v[8]}` LE limbs used consistently T3→T5. `Hit{privkey[32],address[20],zeros}` consistent T6. `run_scalarmul`→`[u8;64]` (x‖y BE) consumed by T5. `derive_address_gpu`→`([u8;32],[u8;20])` consumed by T5 test. `dispatch_mine`→`Vec<Hit>` consumed by T6.

**Note carried into execution:** the MSL crypto (Tasks 3–4 especially) is where real iterative debugging will happen — the per-task host-oracle tests are the mechanism that surfaces bugs; expect multiple implement↔test cycles within those tasks before the gate passes.
