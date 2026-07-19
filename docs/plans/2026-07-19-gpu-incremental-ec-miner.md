# GPU Incremental-EC Miner ("Approach B") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GPU miner's per-candidate full secp256k1 scalar multiplication with one scalar multiplication per thread per batch, followed by a bounded walk of cheap Jacobian point additions (`P_it = P_{it-1} + G`), cutting field-multiplications per candidate from ~4300 to ~275 (~15x).

**Architecture:** Per batch, per GPU thread: derive one `base` scalar via the existing `keccak256(seed‖base_counter)` + range-guard (unchanged from Approach A), compute `P_0 = base*G` once (Jacobian, no inversion), then for `it` in `[0, iters)` walk `P_it = P_{it-1}+G`, convert to affine (one inversion per candidate — this plan's scope, "B1"; batched inversion is future work "B2"), hash to an address, and emit `(base+it) mod n` as the private key if it clears the threshold. Every hit is still re-verified host-side via `k256` before being trusted — that gate is unchanged.

**Tech Stack:** Rust (`metal` crate host), MSL (Metal Shading Language) compute kernels, `ethers`/`k256` for host-side verification and test references.

## Global Constraints

- Every GPU-derived hit MUST still pass the host `k256` re-derivation gate (`verify_hit`) before being trusted — this plan does not touch that invariant, only what the GPU computes before the gate.
- New scalar arithmetic (`(base + it) mod n`) must be bit-exact against a host `U256` reference, including the `n`-wrap boundary (`base` within `iters` of `n`), tested explicitly since natural occurrence probability is `~2^-248`.
- A degenerate point (`base + it ≡ 0 mod n`, i.e. Jacobian `Z == 0`) must be skipped as an invalid candidate, never hashed into a fake "hit".
- Full design rationale and the security-model discussion (why this is weaker than Approach A and why that's acceptable here) live in `docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` — read it before Task 1.
- Kernel source concatenation order stays `keccak.metal` + `field.metal` + `ec.metal` + `miner.metal` (unchanged) — new miner.metal code can call any `field.metal`/`ec.metal` symbol without a new `#include`.
- MSL indent/brace style, `thread`/`device`/`constant` address-space annotations, and buffer-index conventions must match the existing kernels exactly (see any existing kernel function for reference).

---

### Task 1: Extract `scalarmul_jacobian` from `ec.metal` (pure refactor)

**Files:**
- Modify: `kernels/ec.metal:120-145` (the `scalarmul` function)
- Test: existing `tests/gpu_stages.rs::scalarmul_matches_k256_pubkey` (no new test needed — this step's own verification is that this existing test still passes bit-exact)

**Interfaces:**
- Consumes: `jpoint`, `j_infinity()`, `j_double()`, `j_add()`, `fe_mul()`, `fe_inv()`, `GX`/`GY` constants — all already defined earlier in `kernels/ec.metal`/`kernels/field.metal`.
- Produces: `inline jpoint g_point()` — the generator `G` as a Jacobian point (`Z=1`). `__attribute__((noinline)) jpoint scalarmul_jacobian(fe k)` — `k*G` left in Jacobian coordinates (no final inversion). Both are consumed by Task 3 and Task 4's new kernels. `scalarmul(fe k, thread fe& outx, thread fe& outy)` keeps its exact existing signature and behavior (affine output), now implemented as a thin wrapper.

- [ ] **Step 1: Replace the `scalarmul` function body in `kernels/ec.metal`**

Find this block (currently lines 120-145 of `kernels/ec.metal`):

```metal
// R = k*G via MSB-first double-and-add, then Jacobian -> affine.
__attribute__((noinline)) void scalarmul(fe k, thread fe& outx, thread fe& outy) {
    jpoint R = j_infinity();
    jpoint G;
    for (int i = 0; i < 8; i++) {
        G.X.v[i] = GX[i];
        G.Y.v[i] = GY[i];
    }
    G.Z = fe_from_u32(1u);

#pragma clang loop unroll(disable)
    for (int bit = 255; bit >= 0; bit--) {
        R = j_double(R);
        uint limb = k.v[bit >> 5];
        uint b = (limb >> (bit & 31)) & 1u;
        if (b) {
            R = j_add(R, G);
        }
    }

    fe zinv = fe_inv(R.Z);
    fe zinv2 = fe_mul(zinv, zinv);
    fe zinv3 = fe_mul(zinv2, zinv);
    outx = fe_mul(R.X, zinv2);
    outy = fe_mul(R.Y, zinv3);
}
```

Replace it with:

```metal
// The secp256k1 generator point G as a Jacobian point (Z=1). Shared by
// scalarmul_jacobian below and the incremental-walk kernels in miner.metal,
// which need G on its own to add into a running point each step.
inline jpoint g_point() {
    jpoint G;
    for (int i = 0; i < 8; i++) {
        G.X.v[i] = GX[i];
        G.Y.v[i] = GY[i];
    }
    G.Z = fe_from_u32(1u);
    return G;
}

// R = k*G via MSB-first double-and-add, left in Jacobian coordinates (no
// final inversion). Callers that only need affine x,y should use
// `scalarmul` below; callers building an incremental walk (miner.metal) want
// the Jacobian point directly so they can keep adding G without paying for
// repeated inversions.
__attribute__((noinline)) jpoint scalarmul_jacobian(fe k) {
    jpoint R = j_infinity();
    jpoint G = g_point();

#pragma clang loop unroll(disable)
    for (int bit = 255; bit >= 0; bit--) {
        R = j_double(R);
        uint limb = k.v[bit >> 5];
        uint b = (limb >> (bit & 31)) & 1u;
        if (b) {
            R = j_add(R, G);
        }
    }
    return R;
}

// R = k*G, converted to affine (x,y). Thin wrapper over scalarmul_jacobian.
__attribute__((noinline)) void scalarmul(fe k, thread fe& outx, thread fe& outy) {
    jpoint R = scalarmul_jacobian(k);
    fe zinv = fe_inv(R.Z);
    fe zinv2 = fe_mul(zinv, zinv);
    fe zinv3 = fe_mul(zinv2, zinv);
    outx = fe_mul(R.X, zinv2);
    outy = fe_mul(R.Y, zinv3);
}
```

- [ ] **Step 2: Run the existing EC test to confirm the refactor is behavior-preserving**

Run: `cargo test --test gpu_stages scalarmul_matches_k256_pubkey -- --nocapture`
Expected: PASS (identical to pre-refactor — this proves `scalarmul_jacobian` + affine conversion produces the same result as the original inline code).

- [ ] **Step 3: Run the full existing GPU test suite as a regression check**

Run: `cargo test --test gpu_stages`
Expected: all tests PASS (nothing else in `ec.metal`/`miner.metal` changed yet).

- [ ] **Step 4: Commit**

```bash
git add kernels/ec.metal
git commit -m "refactor(gpu): extract scalarmul_jacobian + g_point from scalarmul"
```

---

### Task 2: `scalar_add_small` — batch-offset scalar arithmetic mod n

**Files:**
- Modify: `kernels/miner.metal` (add after the `SECP_N`/`scalar_in_range` block, i.e. after line 53)
- Modify: `src/gpu/mod.rs` (add a new method to `impl MetalContext`)
- Test: `tests/gpu_stages.rs` (new test)

**Interfaces:**
- Consumes: `fe` type, `SECP_N` constant (both already in `kernels/miner.metal`).
- Produces: `inline bool scalar_ge_n(thread const fe& r)`, `inline fe scalar_add_small(thread const fe& base, uint it)` — consumed by Task 3 and Task 4's kernels. Host: `MetalContext::run_scalar_add_mod_n(&self, bases: &[ethers::types::U256], its: &[u32]) -> Vec<ethers::types::U256>` — test-only entry point, not used by production dispatch.

- [ ] **Step 1: Add `scalar_ge_n` + `scalar_add_small` to `kernels/miner.metal`**

Insert immediately after the existing `scalar_in_range` function (after line 53, before the `derive_privkey` comment/function):

```metal
// True iff r >= SECP_N (both 8-limb little-endian), i.e. r is not yet reduced
// into the canonical [0, SECP_N) range.
inline bool scalar_ge_n(thread const fe& r) {
    for (int i = 7; i >= 0; i--) {
        if (r.v[i] != SECP_N[i]) {
            return r.v[i] > SECP_N[i];
        }
    }
    return true; // r == SECP_N
}

// (base + it) mod SECP_N, via a single conditional subtraction. Valid
// whenever base < SECP_N (a scalar_in_range-guarded base) and
// base + it < 2*SECP_N -- true for every `it` this codebase's batch sizes
// use (a few hundred at most; SECP_N > 2^255), so at most one subtraction is
// ever needed. Used to derive an incremental-walk candidate's private key
// from its batch base without a full scalar multiplication.
inline fe scalar_add_small(thread const fe& base, uint it) {
    fe r = base;
    ulong carry = (ulong)it;
    for (int i = 0; i < 8 && carry != 0; i++) {
        ulong s = (ulong)r.v[i] + carry;
        r.v[i] = (uint)s;
        carry = s >> 32;
    }
    if (scalar_ge_n(r)) {
        long borrow = 0;
        for (int i = 0; i < 8; i++) {
            long d = (long)r.v[i] - (long)SECP_N[i] - borrow;
            if (d < 0) {
                d += 0x100000000L;
                borrow = 1;
            } else {
                borrow = 0;
            }
            r.v[i] = (uint)d;
        }
    }
    return r;
}

// Test-only: applies scalar_add_small to the gid-th (base, it) pair.
kernel void scalar_add_mod_n_test(device const uint* bases [[buffer(0)]],
                                   device const uint* its   [[buffer(1)]],
                                   device uint* out         [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    fe base;
    for (int i = 0; i < 8; i++) {
        base.v[i] = bases[gid * 8 + i];
    }
    fe r = scalar_add_small(base, its[gid]);
    for (int i = 0; i < 8; i++) {
        out[gid * 8 + i] = r.v[i];
    }
}
```

- [ ] **Step 2: Add the host-side dispatch method to `src/gpu/mod.rs`**

Add this method inside `impl MetalContext` (e.g. directly after `run_field`, which it closely mirrors):

```rust
    /// Test-only: apply `scalar_add_small` (the incremental-walk's
    /// `(base + it) mod n` arithmetic) on the GPU for each `(base, it)` pair
    /// via `kernels/miner.metal`'s `scalar_add_mod_n_test`, returning the
    /// result as a `U256`. `base` must already be reduced mod n (< n).
    pub fn run_scalar_add_mod_n(&self, bases: &[U256], its: &[u32]) -> Vec<U256> {
        assert_eq!(bases.len(), its.len(), "bases/its length mismatch");
        let n = bases.len();
        let mut base_limbs = vec![0u32; n * 8];
        for (i, b) in bases.iter().enumerate() {
            base_limbs[i * 8..i * 8 + 8].copy_from_slice(&u256_to_limbs(*b));
        }

        // NOTE (corrected during Task 2 review): `field.metal` + `miner.metal`
        // does NOT compile standalone -- miner.metal's pre-existing
        // `scalar_in_range` calls `fe_is_zero` (defined in `ec.metal`), and
        // other pre-existing kernels in the file call `keccak256`. MSL
        // requires every symbol referenced anywhere in the file to resolve,
        // regardless of whether the dispatched entry point reaches it. Use
        // the same 4-file concatenation `dispatch_mine` uses.
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("scalar_add_mod_n_test", None)
            .expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let base_buf = self.device.new_buffer_with_data(
            base_limbs.as_ptr() as *const std::ffi::c_void,
            (base_limbs.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let its_buf = self.device.new_buffer_with_data(
            its.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(its) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_len = n * 8;
        let out_buf = self.device.new_buffer(
            (out_len * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&base_buf), 0);
        enc.set_buffer(1, Some(&its_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const u32;
        let limbs = unsafe { std::slice::from_raw_parts(ptr, out_len) };
        (0..n)
            .map(|i| {
                let mut l = [0u32; 8];
                l.copy_from_slice(&limbs[i * 8..i * 8 + 8]);
                limbs_to_u256(&l)
            })
            .collect()
    }
```

Note: `miner.metal` references `fe`, `fe_is_zero` (from `ec.metal`) only inside functions the test kernel doesn't call, but it also references `SECP_N`/`fe` which only need `field.metal`'s `fe` typedef — `scalar_add_mod_n_test` and its dependencies (`scalar_ge_n`, `scalar_add_small`) use only `fe`/`SECP_N`, no keccak/EC symbols, so concatenating just `field.metal` + `miner.metal` (as above) compiles standalone. This is narrower than the `keccak+field+ec+miner` concatenation `dispatch_mine` uses, which is fine — MSL only requires symbols actually referenced by the compiled entry point to be defined.

- [ ] **Step 3: Add the host reference test to `tests/gpu_stages.rs`**

Add near the other field-arithmetic tests (after `field_ops_match_host_mod_p`):

```rust
// ---------------------------------------------------------------------------
// scalar_add_small (the incremental-walk's `(base+it) mod n` arithmetic):
// GPU vs U256 host reference, including the explicit n-wrap boundary.
// ---------------------------------------------------------------------------

fn secp_n() -> U256 {
    U256::from_str_radix(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
        16,
    )
    .unwrap()
}

#[test]
fn scalar_add_small_matches_host_mod_n() {
    let n = secp_n();
    let ctx = MetalContext::new();

    let bases = vec![
        U256::from(1u32),
        U256::from(1_000_000u32),
        n - U256::from(2u32), // n-wrap boundary case
        n - U256::from(2u32), // repeated: verify determinism
    ];
    let its = vec![5u32, 300u32, 0u32, 4u32];

    let gpu = ctx.run_scalar_add_mod_n(&bases, &its);
    for (i, (b, it)) in bases.iter().zip(its.iter()).enumerate() {
        let want = addmod(*b, U256::from(*it), n);
        assert_eq!(gpu[i], want, "scalar_add_small mismatch case {i}: base={b} it={it}");
    }

    // Explicit assertion that the wrap case actually wrapped (didn't just
    // happen to equal the unwrapped sum), so this test would fail loudly if
    // the conditional subtraction were missing or wrong.
    let wrapped = ctx.run_scalar_add_mod_n(&[n - U256::from(2u32)], &[4u32])[0];
    // (n-2) + 4 = n+2 ≡ 2 (mod n) -- corrected during Task 2 review; the
    // original plan text asserted 1 here, which is arithmetically wrong.
    assert_eq!(wrapped, U256::from(2u32), "n-2 + 4 mod n should wrap to 2");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test --test gpu_stages scalar_add_small_matches_host_mod_n -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kernels/miner.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): scalar_add_small — batch-offset arithmetic mod n"
```

---

### Task 3: Raw-base incremental-walk test kernel (bit-exact vs k256)

**Files:**
- Modify: `kernels/miner.metal` (add after the `scalar_add_mod_n_test` kernel added in Task 2)
- Modify: `src/gpu/mod.rs` (add a new method to `impl MetalContext`)
- Test: `tests/gpu_stages.rs` (new test)

**Interfaces:**
- Consumes: `scalarmul_jacobian`, `g_point`, `j_add`, `fe_is_zero` (Task 1, `ec.metal`), `scalar_add_small` (Task 2), `pubkey_to_address`, `fe_to_bytes_be` (already in `miner.metal`).
- Produces: host `MetalContext::run_mine_incremental_raw(&self, bases: &[[u8; 32]], iters: u32) -> Vec<Vec<([u8; 32], [u8; 20])>>` — one `Vec` of `(privkey, address)` pairs per input base, length `iters` each. Test-only (lets tests pick exact bases, including the n-wrap case, without needing a keccak preimage). Not used by production dispatch.

- [ ] **Step 1: Add the raw-base test kernel to `kernels/miner.metal`**

Append after the `scalar_add_mod_n_test` kernel from Task 2:

```metal
// Test-only: given raw 256-bit bases (not keccak-derived) and a shared
// `iters` count, walk `iters` incremental points per base and emit every
// candidate's (privkey, address) pair unconditionally (no threshold gate).
// Lets tests exercise the walk arithmetic -- including deliberately chosen
// bases near the n-wrap boundary -- without needing a keccak preimage.
// A degenerate point (base+it == 0 mod n, i.e. Jacobian Z == 0) writes an
// all-zero privkey/address pair, which the host test asserts never occurs
// for the bases it chooses (or explicitly checks for, at the wrap case).
kernel void mine_incremental_raw_test(device const uint* bases [[buffer(0)]],
                                       constant uint& iters     [[buffer(1)]],
                                       device uchar* out_priv   [[buffer(2)]],
                                       device uchar* out_addr   [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    fe base;
    for (int i = 0; i < 8; i++) {
        base.v[i] = bases[gid * 8 + i];
    }

    jpoint P = scalarmul_jacobian(base);
    jpoint G = g_point();

    for (uint it = 0; it < iters; it++) {
        if (it > 0) {
            P = j_add(P, G);
        }
        uint out = gid * iters + it;

        if (fe_is_zero(P.Z)) {
            for (uint i = 0; i < 32; i++) out_priv[out * 32 + i] = 0;
            for (uint i = 0; i < 20; i++) out_addr[out * 20 + i] = 0;
            continue;
        }

        fe zinv = fe_inv(P.Z);
        fe zinv2 = fe_mul(zinv, zinv);
        fe zinv3 = fe_mul(zinv2, zinv);
        fe x = fe_mul(P.X, zinv2);
        fe y = fe_mul(P.Y, zinv3);

        thread uchar addr[20];
        pubkey_to_address(x, y, addr);
        for (uint i = 0; i < 20; i++) out_addr[out * 20 + i] = addr[i];

        fe priv = scalar_add_small(base, it);
        thread uchar privBytes[32];
        fe_to_bytes_be(priv, privBytes);
        for (uint i = 0; i < 32; i++) out_priv[out * 32 + i] = privBytes[i];
    }
}
```

- [ ] **Step 2: Add the host-side dispatch method to `src/gpu/mod.rs`**

Add inside `impl MetalContext`, after `run_scalar_add_mod_n` from Task 2:

```rust
    /// Test-only: run the incremental Jacobian walk (Approach B's core loop)
    /// from `iters` explicit 256-bit `bases` (not keccak-derived), via
    /// `kernels/miner.metal`'s `mine_incremental_raw_test`. Returns, per base,
    /// `iters` `(privkey, address)` pairs for `(base+0)*G .. (base+iters-1)*G`.
    /// Lets tests target exact scalar values (e.g. the n-wrap boundary).
    pub fn run_mine_incremental_raw(
        &self,
        bases: &[[u8; 32]],
        iters: u32,
    ) -> Vec<Vec<([u8; 32], [u8; 20])>> {
        let n = bases.len();
        let mut in_limbs = vec![0u32; n * 8];
        for (i, k) in bases.iter().enumerate() {
            for limb in 0..8 {
                let hi = 28 - limb * 4;
                in_limbs[i * 8 + limb] =
                    u32::from_be_bytes([k[hi], k[hi + 1], k[hi + 2], k[hi + 3]]);
            }
        }

        // NOTE (corrected post-Task-2 review): `miner.metal`'s pre-existing
        // `pubkey_to_address` (called by our new kernel) and other kernels
        // call `keccak256`, so the whole file needs `keccak.metal` present
        // in the concatenation even though our entry point doesn't call it
        // directly -- MSL requires every symbol referenced anywhere in the
        // translation unit to resolve, not just in the dispatched kernel's
        // call graph. Use the same 4-file concatenation `dispatch_mine` uses.
        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib
            .get_function("mine_incremental_raw_test", None)
            .expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let in_buf = self.device.new_buffer_with_data(
            in_limbs.as_ptr() as *const std::ffi::c_void,
            (in_limbs.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_arr = [iters];
        let iters_buf = self.device.new_buffer_with_data(
            iters_arr.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let priv_len = n * (iters as usize) * 32;
        let addr_len = n * (iters as usize) * 20;
        let out_priv_buf = self
            .device
            .new_buffer(priv_len as u64, MTLResourceOptions::StorageModeShared);
        let out_addr_buf = self
            .device
            .new_buffer(addr_len as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&iters_buf), 0);
        enc.set_buffer(2, Some(&out_priv_buf), 0);
        enc.set_buffer(3, Some(&out_addr_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let priv_ptr = out_priv_buf.contents() as *const u8;
        let priv_bytes = unsafe { std::slice::from_raw_parts(priv_ptr, priv_len) };
        let addr_ptr = out_addr_buf.contents() as *const u8;
        let addr_bytes = unsafe { std::slice::from_raw_parts(addr_ptr, addr_len) };

        (0..n)
            .map(|i| {
                (0..iters as usize)
                    .map(|it| {
                        let mut pk = [0u8; 32];
                        pk.copy_from_slice(
                            &priv_bytes[(i * iters as usize + it) * 32..(i * iters as usize + it) * 32 + 32],
                        );
                        let mut addr = [0u8; 20];
                        addr.copy_from_slice(
                            &addr_bytes[(i * iters as usize + it) * 20..(i * iters as usize + it) * 20 + 20],
                        );
                        (pk, addr)
                    })
                    .collect()
            })
            .collect()
    }
```

- [ ] **Step 3: Add the bit-exact test to `tests/gpu_stages.rs`**

Add after the `scalar_add_small_matches_host_mod_n` test from Task 2:

```rust
// ---------------------------------------------------------------------------
// Incremental Jacobian walk (Approach B core loop): GPU vs k256, including
// the explicit n-wrap boundary.
// ---------------------------------------------------------------------------

#[test]
fn incremental_walk_matches_k256() {
    use ethers::core::k256::ecdsa::SigningKey;
    use ethers::utils::secret_key_to_address;
    let n = secp_n();
    let ctx = MetalContext::new();

    let mut bases: Vec<[u8; 32]> = vec![
        hex_to_32("0000000000000000000000000000000000000000000000000000000000000001"),
        hex_to_32("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"),
    ];
    // Explicit n-wrap boundary: base = n-2, walk 4 steps (crosses n at it=2).
    let mut n_minus_2 = [0u8; 32];
    (n - U256::from(2u32)).to_big_endian(&mut n_minus_2);
    bases.push(n_minus_2);

    let iters = 4u32;
    let gpu = ctx.run_mine_incremental_raw(&bases, iters);

    for (bi, base_bytes) in bases.iter().enumerate() {
        let base = U256::from_big_endian(base_bytes);
        for it in 0..iters {
            let want_scalar = addmod(base, U256::from(it), n);
            if want_scalar.is_zero() {
                // Degenerate point at infinity: GPU must emit the all-zero
                // sentinel, not a fabricated address.
                assert_eq!(gpu[bi][it as usize].0, [0u8; 32], "base {bi} it {it}: expected zero-sentinel privkey");
                assert_eq!(gpu[bi][it as usize].1, [0u8; 20], "base {bi} it {it}: expected zero-sentinel address");
                continue;
            }
            let mut want_bytes = [0u8; 32];
            want_scalar.to_big_endian(&mut want_bytes);
            let sk = SigningKey::from_bytes((&want_bytes).into()).expect("canonical scalar");
            let want_addr = secret_key_to_address(&sk);

            let (gpu_priv, gpu_addr) = gpu[bi][it as usize];
            assert_eq!(gpu_priv, want_bytes, "base {bi} it {it}: privkey mismatch");
            assert_eq!(&gpu_addr[..], want_addr.as_bytes(), "base {bi} it {it}: address mismatch");
        }
    }

    // The n-wrap base (index 2) must actually wrap within the tested window:
    // n-2, n-1, 0 (degenerate), 1 -- assert the degenerate slot is exactly
    // it=2, proving the boundary was really exercised.
    let base = U256::from_big_endian(&bases[2]);
    assert!(addmod(base, U256::from(2u32), n).is_zero(), "test setup: expected wrap at it=2");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test --test gpu_stages incremental_walk_matches_k256 -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the full GPU test suite as a regression check**

Run: `cargo test --test gpu_stages`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add kernels/miner.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): incremental Jacobian walk, verified bit-exact vs k256"
```

---

### Task 4: Production `mine_incremental` kernel + `dispatch_mine_incremental`

**Files:**
- Modify: `kernels/miner.metal` (add after `mine_incremental_raw_test`)
- Modify: `src/gpu/mod.rs:307-415` (refactor `dispatch_mine` to share a helper, add `dispatch_mine_incremental`)
- Test: `tests/gpu_stages.rs` (new test)

**Interfaces:**
- Consumes: same building blocks as Task 3, plus `derive_privkey`, `scalar_in_range`, `leading_zero_nibbles`, `HIT_STRIDE`, `MAX_HITS` (already in `miner.metal`, used by the existing `mine` kernel).
- Produces: `kernel void mine_incremental(...)` with the **exact same buffer signature** as the existing `mine` kernel (buffers 0-5: seeds, base_counters, iters, threshold, hit_count, hits) so the host dispatch code is identical except for the entry-point name. Host: `MetalContext::dispatch_mine_incremental(&self, seeds: &[[u8; 32]], base_counters: &[u64], iters: u32, threshold: u32) -> Vec<Hit>` — same signature as the existing `dispatch_mine`, consumed by Task 5.

- [ ] **Step 1: Add the production kernel to `kernels/miner.metal`**

Append after `mine_incremental_raw_test`:

```metal
// Approach B: one thread per (seed, base_counter). Derives a single batch
// base scalar (exactly like `mine`'s per-candidate derive_privkey, but
// called once per batch instead of once per candidate), computes base*G
// once, then walks `iters` candidates via cheap Jacobian point additions
// instead of `iters` full scalar multiplications. Emits any candidate with
// >= threshold leading-zero nibbles to the same bounded `hits` buffer `mine`
// uses (identical wire format), so host dispatch code is shared.
kernel void mine_incremental(device const uchar* seeds [[buffer(0)]],
                              device const ulong* base_counters [[buffer(1)]],
                              constant uint& iters [[buffer(2)]],
                              constant uint& threshold [[buffer(3)]],
                              device atomic_uint* hit_count [[buffer(4)]],
                              device uchar* hits [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    thread uchar seed[32];
    for (uint i = 0; i < 32; i++) {
        seed[i] = seeds[gid * 32 + i];
    }
    ulong base_counter = base_counters[gid];

    fe base = derive_privkey(seed, base_counter);
    if (!scalar_in_range(base)) {
        return; // batch-base guard miss (probability ~2^-128): skip whole batch
    }

    jpoint P = scalarmul_jacobian(base);
    jpoint G = g_point();

    for (uint it = 0; it < iters; it++) {
        if (it > 0) {
            P = j_add(P, G);
        }
        if (fe_is_zero(P.Z)) {
            continue; // base+it == 0 (mod n): invalid scalar, skip candidate
        }

        fe zinv = fe_inv(P.Z);
        fe zinv2 = fe_mul(zinv, zinv);
        fe zinv3 = fe_mul(zinv2, zinv);
        fe x = fe_mul(P.X, zinv2);
        fe y = fe_mul(P.Y, zinv3);

        thread uchar addr[20];
        pubkey_to_address(x, y, addr);

        uint zeros = leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

        fe priv = scalar_add_small(base, it);

        uint idx = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (idx >= MAX_HITS) {
            continue; // batch saturated; drop (host clamps its read too)
        }

        thread uchar privBytes[32];
        fe_to_bytes_be(priv, privBytes);
        uint out = idx * HIT_STRIDE;
        for (uint i = 0; i < 32; i++) {
            hits[out + i] = privBytes[i];
        }
        for (uint i = 0; i < 20; i++) {
            hits[out + 32 + i] = addr[i];
        }
        hits[out + 52] = (uchar)zeros;
        for (uint i = 53; i < HIT_STRIDE; i++) {
            hits[out + i] = 0;
        }
    }
}
```

- [ ] **Step 2: Refactor `src/gpu/mod.rs` to share dispatch logic between `mine` and `mine_incremental`**

Replace the existing `dispatch_mine` method (currently `src/gpu/mod.rs:307-415`) with a private helper plus two thin public wrappers:

```rust
    /// Search `iters` candidate keys per thread (one thread per
    /// `(seed, base_counter)` pair, counters `base_counters[i]..base_counters[i]+iters`)
    /// via `kernels/miner.metal`'s `mine` kernel (Approach A: every candidate
    /// is an independent keccak-derived scalar, full scalar-mult each),
    /// returning every derived address with `>= threshold` leading-zero
    /// nibbles (bounded to at most 1024 hits per call -- extras in a
    /// saturated batch are silently dropped on the GPU side; callers on a
    /// tight loop should keep `threshold` high enough that a batch rarely
    /// saturates).
    ///
    /// SECURITY: every returned `Hit` is a GPU-derived candidate only. Callers
    /// MUST re-verify each one with `verify_hit` before treating `address` as
    /// trustworthy (that host-side gate is what the CLI's per-hit hard-abort
    /// enforces).
    pub fn dispatch_mine(
        &self,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        self.dispatch_mine_kernel("mine", seeds, base_counters, iters, threshold)
    }

    /// Search `iters` candidate keys per thread via `kernels/miner.metal`'s
    /// `mine_incremental` kernel (Approach B: one keccak-derived base scalar
    /// per thread per batch, then `iters` cheap Jacobian point additions
    /// instead of `iters` full scalar multiplications -- see
    /// `docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` for the
    /// cost model and the security-model discussion of why this is weaker
    /// than, but not unsafe compared to, `dispatch_mine`'s Approach A).
    /// Same wire format, same buffer layout, same 1024-hit cap as `mine`.
    ///
    /// SECURITY: same requirement as `dispatch_mine` -- callers MUST
    /// re-verify every returned `Hit` with `verify_hit` before trusting it.
    pub fn dispatch_mine_incremental(
        &self,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        self.dispatch_mine_kernel("mine_incremental", seeds, base_counters, iters, threshold)
    }

    /// Shared dispatch body for `dispatch_mine`/`dispatch_mine_incremental`:
    /// both kernels have an identical buffer signature (seeds, base_counters,
    /// iters, threshold, hit_count, hits), differing only in per-candidate
    /// math, so only the compiled entry-point name changes.
    fn dispatch_mine_kernel(
        &self,
        entry: &str,
        seeds: &[[u8; 32]],
        base_counters: &[u64],
        iters: u32,
        threshold: u32,
    ) -> Vec<Hit> {
        assert_eq!(
            seeds.len(),
            base_counters.len(),
            "seeds/base_counters length mismatch"
        );
        let n = seeds.len();

        let mut seed_bytes = vec![0u8; n * 32];
        for (i, s) in seeds.iter().enumerate() {
            seed_bytes[i * 32..i * 32 + 32].copy_from_slice(s);
        }

        let keccak_src = include_str!("../../kernels/keccak.metal");
        let field_src = include_str!("../../kernels/field.metal");
        let ec_src = include_str!("../../kernels/ec.metal");
        let miner_src = include_str!("../../kernels/miner.metal");
        let src = format!("{keccak_src}\n{field_src}\n{ec_src}\n{miner_src}");

        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .expect("kernel compile failed");
        let func = lib.get_function(entry, None).expect("entry not found");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");

        let seeds_buf = self.device.new_buffer_with_data(
            seed_bytes.as_ptr() as *const std::ffi::c_void,
            seed_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counters_buf = self.device.new_buffer_with_data(
            base_counters.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(base_counters) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let iters_buf = self.device.new_buffer_with_data(
            &iters as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let threshold_buf = self.device.new_buffer_with_data(
            &threshold as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // MTLBuffer contents are undefined until written -- zero the atomic
        // counter explicitly rather than relying on incidental zero pages.
        unsafe {
            std::ptr::write_bytes(
                hit_count_buf.contents() as *mut u8,
                0,
                std::mem::size_of::<u32>(),
            );
        }
        let hits_bytes = MINE_MAX_HITS * MINE_HIT_STRIDE;
        let hits_buf = self
            .device
            .new_buffer(hits_bytes as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&seeds_buf), 0);
        enc.set_buffer(1, Some(&counters_buf), 0);
        enc.set_buffer(2, Some(&iters_buf), 0);
        enc.set_buffer(3, Some(&threshold_buf), 0);
        enc.set_buffer(4, Some(&hit_count_buf), 0);
        enc.set_buffer(5, Some(&hits_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let found = unsafe { *(hit_count_buf.contents() as *const u32) } as usize;
        let count = found.min(MINE_MAX_HITS);

        let ptr = hits_buf.contents() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, hits_bytes) };

        (0..count)
            .map(|i| {
                let rec = i * MINE_HIT_STRIDE;
                let mut privkey = [0u8; 32];
                privkey.copy_from_slice(&bytes[rec..rec + 32]);
                let mut address = [0u8; 20];
                address.copy_from_slice(&bytes[rec + 32..rec + 52]);
                let zeros = bytes[rec + 52];
                Hit {
                    privkey,
                    address,
                    zeros,
                }
            })
            .collect()
    }
```

- [ ] **Step 3: Add the production-kernel test to `tests/gpu_stages.rs`**

Add after `mine_finds_and_verifies_low_threshold` (mirrors it exactly, using the incremental dispatch):

```rust
#[test]
fn mine_incremental_finds_and_verifies_low_threshold() {
    let ctx = MetalContext::new();
    let seeds: Vec<[u8; 32]> = (0..256)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = (i & 0xff) as u8;
            s[1] = (i >> 8) as u8;
            s[31] = 0x22;
            s
        })
        .collect();
    let base: Vec<u64> = vec![0; seeds.len()];
    let hits = ctx.dispatch_mine_incremental(&seeds, &base, 4096, 2); // >=2 leading zero nibbles
    assert!(!hits.is_empty(), "should find >=2-zero addresses");
    for h in &hits {
        assert!(ctx.verify_hit(h.privkey, h.address), "hit failed host re-derivation");
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --test gpu_stages mine_incremental_finds_and_verifies_low_threshold -- --nocapture`
Expected: PASS.

Run: `cargo test --test gpu_stages` (full regression, `dispatch_mine`/`mine_finds_and_verifies_low_threshold` must still pass unchanged since `dispatch_mine`'s public behavior didn't change, only its internal implementation).
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernels/miner.metal src/gpu/mod.rs tests/gpu_stages.rs
git commit -m "feat(gpu): mine_incremental production kernel + dispatch_mine_incremental"
```

---

### Task 5: Wire `gpu_driver::run_batches` to the incremental kernel

**Files:**
- Modify: `src/miner/gpu_driver.rs:56` (the `dispatch_mine` call inside `run_batches`)
- Test: existing `tests/gpu_stages.rs::gpu_driver_finds_and_reports_low_target` (e2e, no code change needed — verifies the switch didn't break the driver loop)

**Interfaces:**
- Consumes: `MetalContext::dispatch_mine_incremental` (Task 4).
- Produces: no new public interface — `run_batches`'s signature and behavior contract (dispatch → verify-or-abort → report → throttle) are unchanged; only which kernel it dispatches changes.

- [ ] **Step 1: Switch the dispatch call in `src/miner/gpu_driver.rs`**

In `run_batches` (currently line 56), change:

```rust
        let hits = ctx.dispatch_mine(seeds, &base_counters, ITERS, threshold);
```

to:

```rust
        let hits = ctx.dispatch_mine_incremental(seeds, &base_counters, ITERS, threshold);
```

Also update the module doc comment at the top of the file (currently line 1):

```rust
//! GPU driver: dispatch Metal mine batches, verify every hit vs k256, report.
```

to:

```rust
//! GPU driver: dispatch Metal incremental-EC mine batches ("Approach B" --
//! docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md), verify every
//! hit vs k256, report.
```

- [ ] **Step 2: Run the e2e GPU driver test**

Run: `cargo test --test gpu_stages gpu_driver_finds_and_reports_low_target -- --nocapture`
Expected: PASS (driver still finds a `>=2`-zero hit, verifies it, writes the file, trips `stop`).

- [ ] **Step 3: Run the full test suite (library + integration) as a final regression check**

Run: `cargo test`
Expected: all tests PASS across `src/miner/*` unit tests and `tests/gpu_stages.rs`.

- [ ] **Step 4: Manual smoke run (throughput sanity, not correctness — optional but recommended)**

Run: `cargo run --release --bin zerohunt-gpu -- 6` and let it run for ~10-15 seconds, then Ctrl-C.
Expected: "New best [GPU] ..." lines appear noticeably faster than pre-change runs of the same command (no hard number gate — this is a sanity check that the switch is live and not, e.g., silently falling back to zero hits). No FATAL abort lines (a FATAL line means a verify_hit mismatch — treat as a correctness bug, not a perf issue, and stop before continuing to Task 6).

- [ ] **Step 5: Commit**

```bash
git add src/miner/gpu_driver.rs
git commit -m "feat(gpu): switch run_batches to the incremental-EC kernel"
```

---

### Task 6: Docs — mark Approach B implemented, update README

**Files:**
- Modify: `docs/specs/2026-07-17-gpu-vanity-miner-design.md:14-16` (the "possible follow-up" note)
- Modify: `README.MD` (brief mention of the incremental kernel, wherever the README currently describes the GPU miner's approach)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the forward-reference in the Approach A spec**

In `docs/specs/2026-07-17-gpu-vanity-miner-design.md`, replace:

```markdown
Non-goal (this spec): the incremental point-addition speed optimization
("Approach B" / profanity-style). That is a possible follow-up, gated on Approach
A being proven correct and benchmarked.
```

with:

```markdown
Non-goal (this spec): the incremental point-addition speed optimization
("Approach B" / profanity-style). Implemented as a follow-up once Approach A
was proven correct and benchmarked — see
`docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md`. Approach A's
kernel (`mine`) stays in the tree as the reference/fallback path; the
unified driver (`gpu_driver::run_batches`) now dispatches Approach B's
`mine_incremental` by default.
```

- [ ] **Step 2: Update the "How it works" section of `README.MD`**

Replace this block (currently `README.MD:171-183`):

```markdown
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
```

with:

```markdown
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
per *batch* and walks the rest of the batch via point additions — about 15x
fewer field operations per candidate. This is a different, and weaker,
security model than Approach A's "every key an independent oracle output":
keys within one batch are affinely related (their difference is public), so
recovering one doesn't help without solving discrete log, but the
correlation set is bounded to a batch instead of empty. See
`docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` for the full
cost model and security-model writeup — this codebase exists because a
previous vanity miner (`profanity`) shipped a weak model without disclosing
it, so this tradeoff is documented rather than left implicit.
```

- [ ] **Step 3: Commit**

```bash
git add docs/specs/2026-07-17-gpu-vanity-miner-design.md README.MD
git commit -m "docs: mark Approach B implemented, link from README"
```
