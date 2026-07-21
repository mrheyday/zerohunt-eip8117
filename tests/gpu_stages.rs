use nullforge::gpu::MetalContext;
use std::sync::{Mutex, MutexGuard};

/// Serializes every GPU-touching test onto one Metal command stream.
///
/// `cargo test` runs test fns concurrently by default. Each test here builds
/// its own `MetalContext` and dispatches real compute work; running ~10 of them
/// at once oversubscribes the single M1 GPU, and the OS watchdog then kills
/// whichever command buffers overrun — surfacing as nondeterministic "0 hits"
/// or host-re-derivation failures in whichever tests lost the race. Holding this
/// lock across each test's GPU section makes the suite deterministic without the
/// caller having to remember `--test-threads=1`.
///
/// Poison is deliberately recovered: a panic in one GPU test (e.g. the fail-loud
/// watchdog check in `MetalContext`) must not cascade a `PoisonError` into every
/// other test.
fn gpu_guard() -> MutexGuard<'static, ()> {
    static GPU_LOCK: Mutex<()> = Mutex::new(());
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn echo_kernel_doubles_thread_id() {
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let src = include_str!("../kernels/echo.metal");
    let out = ctx.run_u32_kernel(src, "echo", 256, 4, 64);
    assert_eq!(out.len(), 256);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, (i as u32) * 2, "thread {i} wrong");
    }
}

#[test]
fn keccak_matches_host_reference() {
    use ethers::utils::keccak256;
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let inputs: Vec<Vec<u8>> = vec![vec![], b"abc".to_vec(), (0u8..64).collect()];
    let gpu = ctx.run_keccak_fixed64(&inputs);
    for (i, inp) in inputs.iter().enumerate() {
        assert_eq!(gpu[i], keccak256(inp), "keccak mismatch on input {i}");
    }
}

// ---------------------------------------------------------------------------
// secp256k1 field arithmetic (mod p) host reference + GPU cross-check.
//
// NOTE on deviation from the brief's snippet: the brief wrote the host add as
// `a.overflowing_add(b).0 % p` and sub as `(a + p - b) % p`. Both are unsound
// on `ethers::types::U256` (primitive-types), whose `+` PANICS on overflow
// unconditionally (not just in debug). `a + p` overflows for any `a` near p —
// it fires on the brief's own given case `(p-1, 5)` where `a + p = 2p-1`. And
// `overflowing_add(b).0 % p` silently returns the wrong residue whenever the
// add actually wraps (the wrapped value differs from the true sum by 2^256,
// and 2^256 mod p = 0x1000003D1 != 0). The parent directive overrides the
// "verbatim" instruction with "the reference must itself be correct", so the
// host ARITHMETIC is corrected here (overflow-safe add/sub/mul + modpow),
// while the constants, LE limb layout, kernel signature, and opcodes stay
// exactly as the brief specifies.
// ---------------------------------------------------------------------------

use ethers::types::U256;

fn secp_p() -> U256 {
    U256::from_str_radix(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
        16,
    )
    .unwrap()
}

/// (a + b) mod p, overflow-safe. Assumes a, b < p.
fn addmod(a: U256, b: U256, p: U256) -> U256 {
    let complement = p - b; // b < p => no overflow, complement in (0, p]
    if a >= complement {
        a - complement // == a + b - p, in [0, p)
    } else {
        a + b // < p, cannot overflow
    }
}

/// (a - b) mod p, overflow-safe. Assumes a, b < p.
fn submod(a: U256, b: U256, p: U256) -> U256 {
    if a >= b {
        a - b
    } else {
        p - (b - a) // b - a in (0, p) => result in (0, p), no overflow
    }
}

/// (a * b) mod p via double-and-add; never lets a 256-bit product overflow.
fn mulmod(a: U256, b: U256, p: U256) -> U256 {
    let mut a = a % p;
    let mut b = b % p;
    let mut acc = U256::zero();
    while !b.is_zero() {
        if b.bit(0) {
            acc = addmod(acc, a, p);
        }
        a = addmod(a, a, p); // double
        b >>= 1;
    }
    acc
}

/// base^exp mod p via square-and-multiply.
fn modpow(base: U256, exp: U256, p: U256) -> U256 {
    let mut result = U256::one() % p;
    let mut base = base % p;
    let mut e = exp;
    while !e.is_zero() {
        if e.bit(0) {
            result = mulmod(result, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    result
}

// Independent host-reference sanity checks (no GPU): validate the gate itself.
#[test]
fn host_field_reference_is_self_consistent() {
    let p = secp_p();
    let two = U256::from(2u32);
    // inv(2) = (p+1)/2 in closed form; validates modpow independently of mulmod.
    let inv2_closed = (p + U256::one()) >> 1;
    assert_eq!(
        modpow(two, p - two, p),
        inv2_closed,
        "modpow(2,p-2) != (p+1)/2"
    );
    // 2 * inv(2) == 1 (mod p): validates mulmod + modpow together.
    assert_eq!(mulmod(two, inv2_closed, p), U256::one(), "2*inv(2) != 1");
    // Fermat round-trip for a large operand.
    let a = U256::from_str_radix(
        "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789",
        16,
    )
    .unwrap()
        % p;
    let inv_a = modpow(a, p - two, p);
    assert_eq!(mulmod(a, inv_a, p), U256::one(), "a * inv(a) != 1");
}

#[test]
fn field_ops_match_host_mod_p() {
    let p = secp_p();
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    // p-1 and p-2 as hex (from the brief's given secp256k1 modulus).
    let p_minus_1 = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E";
    let p_minus_2 = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D";
    let big = "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789";
    let cases: &[(&str, &str)] = &[
        ("2", "3"),
        (p_minus_1, "5"), // p-1, 5
        (big, "1234567890ABCDEF"),
        (p_minus_1, p_minus_1), // max operands: add-carry edge + a*a
        (p_minus_2, p_minus_2), // near p: a*a edge
        (big, big),             // large a*a for fe_mul
    ];
    for (ah, bh) in cases {
        let a = U256::from_str_radix(ah, 16).unwrap() % p;
        let b = U256::from_str_radix(bh, 16).unwrap() % p;
        let add = addmod(a, b, p);
        let sub = submod(a, b, p);
        let mul = mulmod(a, b, p);
        let inv = modpow(a, p - U256::from(2u32), p); // Fermat inverse of a
        assert_eq!(ctx.run_field(a, b, 0), add, "add {ah} {bh}");
        assert_eq!(ctx.run_field(a, b, 1), sub, "sub {ah} {bh}");
        assert_eq!(ctx.run_field(a, b, 2), mul, "mul {ah} {bh}");
        assert_eq!(ctx.run_field(a, a, 3), inv, "inv {ah}");
    }
}

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
    let _gpu = gpu_guard();
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
    // (n-2) + 4 = n+2, which mod n = 2
    let wrapped = ctx.run_scalar_add_mod_n(&[n - U256::from(2u32)], &[4u32])[0];
    assert_eq!(wrapped, U256::from(2u32), "n-2 + 4 mod n should wrap to 2");
}

// ---------------------------------------------------------------------------
// Incremental Jacobian walk (Approach B core loop): GPU vs k256, including
// the explicit n-wrap boundary.
// ---------------------------------------------------------------------------

#[test]
fn incremental_walk_matches_k256() {
    use ethers::core::k256::ecdsa::SigningKey;
    use ethers::utils::secret_key_to_address;
    let n = secp_n();
    let _gpu = gpu_guard();
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

// ---------------------------------------------------------------------------
// secp256k1 EC scalar-mult (k*G -> affine pubkey) GPU vs k256 cross-check.
// ---------------------------------------------------------------------------

/// Parse a 64-char hex string into a 32-byte big-endian array.
fn hex_to_32(h: &str) -> [u8; 32] {
    let bytes = (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect::<Vec<u8>>();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

#[test]
fn scalarmul_matches_k256_pubkey() {
    use ethers::core::k256::ecdsa::SigningKey;
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    // deterministic keys incl. edges: 1, 2, and a fixed 32-byte value
    let mut keys: Vec<[u8; 32]> = vec![[0u8; 32], [0u8; 32], [0u8; 32]];
    keys[0][31] = 1;
    keys[1][31] = 2;
    keys[2] = hex_to_32("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    let gpu = ctx.run_scalarmul(&keys);
    for (i, k) in keys.iter().enumerate() {
        let sk = SigningKey::from_bytes(k.into()).unwrap();
        let pt = sk.verifying_key().to_encoded_point(false); // 0x04 ‖ x(32) ‖ y(32)
        let want = &pt.as_bytes()[1..65];
        assert_eq!(
            &gpu[i][..],
            want,
            "pubkey mismatch key {i}\n gpu x={} y={}\nwant x={} y={}",
            hex(&gpu[i][..32]),
            hex(&gpu[i][32..]),
            hex(&want[..32]),
            hex(&want[32..]),
        );
    }
}

/// Lowercase hex of a byte slice (test diagnostics only).
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Full derive(seed,counter) -> (privkey,address) pipeline: keccak + field +
// ec + miner kernels concatenated, cross-checked against host k256.
// ---------------------------------------------------------------------------

#[test]
fn pipeline_privkey_and_address_match_host() {
    use ethers::core::k256::ecdsa::SigningKey;
    use ethers::utils::secret_key_to_address;
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let seeds: Vec<[u8; 32]> = (0..8)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = i as u8;
            s[31] = 0xA5;
            s
        })
        .collect();
    let counters: Vec<u64> = (0..8).collect();
    let out = ctx.derive_address_gpu(&seeds, &counters);
    for (i, (priv_k, addr)) in out.iter().enumerate() {
        // host: privkey = keccak(seed‖counter_le); guard skips 0/>=n (won't hit here)
        let sk = SigningKey::from_bytes(priv_k.into()).expect("canonical key");
        let want = secret_key_to_address(&sk);
        assert_eq!(&addr[..], want.as_bytes(), "address mismatch idx {i}");
        assert!(
            ctx.verify_hit(*priv_k, *addr),
            "verify_hit false for idx {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// mine kernel: per-thread iters loop + atomic hit-buffer append, cross-checked
// via the same host re-derivation gate the CLI uses (verify_hit).
// ---------------------------------------------------------------------------

#[test]
fn mine_finds_and_verifies_low_threshold() {
    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let seeds: Vec<[u8; 32]> = (0..256)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = (i & 0xff) as u8;
            s[1] = (i >> 8) as u8;
            s[31] = 0x11;
            s
        })
        .collect();
    let base: Vec<u64> = vec![0; seeds.len()];
    // Approach-A `mine` does a FULL secp256k1 scalar-mul per candidate, so a
    // large per-dispatch iters count keeps one command buffer on the GPU long
    // enough to trip the M1 watchdog (killed buffer -> torn output -> 0 hits).
    // 16 iters x 256 seeds is ample to find a >=2-nibble hit while staying well
    // under the watchdog. (The cheap incremental walk in
    // `mine_incremental_finds_and_verifies_low_threshold` can afford far more.)
    let hits = ctx.dispatch_mine(&seeds, &base, 16, 2); // >=2 leading zero nibbles
    assert!(!hits.is_empty(), "should find >=2-zero addresses");
    for h in &hits {
        assert!(
            ctx.verify_hit(h.privkey, h.address),
            "hit failed host re-derivation"
        );
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}

// ---------------------------------------------------------------------------
// mine_incremental kernel (Approach B): same threshold-gated hit-buffer wire
// format as `mine`, cross-checked via the same host re-derivation gate.
// ---------------------------------------------------------------------------

#[test]
fn mine_incremental_finds_and_verifies_low_threshold() {
    let _gpu = gpu_guard();
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

// ---------------------------------------------------------------------------
// End-to-end: the unified GPU driver (run_batches) dispatching real Metal
// batches through the shared state funnel, not just the raw kernel.
// ---------------------------------------------------------------------------

/// End-to-end: the GPU driver, given a low target, finds a verified hit, writes
/// the file, and trips `stop`. Mirrors the device-gated style of the other GPU
/// tests in this file (runs on the Metal device present in CI/dev machines).
#[test]
fn gpu_driver_finds_and_reports_low_target() {
    use nullforge::gpu::MetalContext;
    use nullforge::miner::gpu_driver::{run_batches, N_THREADS};
    use nullforge::miner::shared::MinerShared;
    use std::sync::Arc;
    use std::time::Instant;

    let _gpu = gpu_guard();
    let ctx = MetalContext::new();

    let file = tempfile::NamedTempFile::new().unwrap();
    // target 2 => the very first batch should surface >=2-zero hits fast.
    let shared = Arc::new(MinerShared::new(
        2,
        file.reopen().unwrap(),
        Instant::now(),
        nullforge::keyenc::KeySink::RevealPlaintext,
    ));

    // Deterministic distinct seeds (content is irrelevant to correctness).
    let mut seeds = vec![[0u8; 32]; N_THREADS];
    for (i, s) in seeds.iter_mut().enumerate() {
        s[0..8].copy_from_slice(&(i as u64).to_le_bytes());
    }

    run_batches(&ctx, Arc::clone(&shared), &seeds);

    assert!(shared.should_stop(), "should stop after reaching target 2");
    let best = shared.take_best().expect("should have a best");
    assert!(best.zeros >= 2, "best zeros {} >= target 2", best.zeros);

    use std::io::Read;
    let mut contents = String::new();
    file.reopen()
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();
    assert!(!contents.trim().is_empty(), "file should have a hit line");
}

// ---------------------------------------------------------------------------
// CREATE2 salt mining: keccak-only kernel, cross-checked against ethers'
// canonical CREATE2 address derivation.
// ---------------------------------------------------------------------------

#[test]
fn create2_finds_and_verifies_low_threshold() {
    use ethers::types::Address;
    use ethers::utils::get_create2_address_from_hash;

    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let deployer: [u8; 20] = [0x11; 20];
    let initcodehash: [u8; 32] = [0x22; 32];

    let n = 256usize;
    let base_salts: Vec<[u8; 32]> = (0..n)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = (i & 0xff) as u8;
            s[1] = (i >> 8) as u8;
            s[23] = 0x11;
            s
        })
        .collect();
    let base_counters: Vec<u64> = vec![0u64; n];

    // >= 2 leading zero nibbles across 256*4096 candidates -> plenty of hits fast.
    let hits = ctx.dispatch_create2(
        &deployer,
        &initcodehash,
        &base_salts,
        &base_counters,
        4096,
        2,
    );
    assert!(!hits.is_empty(), "should find >=2-zero CREATE2 addresses");

    for h in &hits {
        // Host re-derivation gate.
        assert!(
            ctx.verify_create2(&deployer, &initcodehash, &h.salt, &h.address),
            "GPU CREATE2 hit failed host re-derivation"
        );
        // Independent cross-check against ethers' canonical CREATE2.
        let want =
            get_create2_address_from_hash(Address::from_slice(&deployer), h.salt, initcodehash);
        assert_eq!(want.as_bytes(), h.address, "CREATE2 address != ethers");
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}

#[test]
fn create3_finds_and_verifies_low_threshold() {
    use nullforge::gpu::STANDARD_CREATE3_PROXY_HASH;

    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let factory: [u8; 20] = [0x11; 20];
    let proxy_hash = STANDARD_CREATE3_PROXY_HASH;

    let n = 256usize;
    let base_salts: Vec<[u8; 32]> = (0..n)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = (i & 0xff) as u8;
            s[1] = (i >> 8) as u8;
            s[23] = 0x11;
            s
        })
        .collect();
    let base_counters: Vec<u64> = vec![0u64; n];

    // >= 2 leading zero nibbles across 256*4096 candidates -> plenty of hits fast.
    let hits = ctx.dispatch_create3(&factory, &proxy_hash, &base_salts, &base_counters, 4096, 2);
    assert!(!hits.is_empty(), "should find >=2-zero CREATE3 addresses");

    for h in &hits {
        // Host re-derivation gate (proxy CREATE2 -> proxy CREATE nonce-1).
        assert!(
            ctx.verify_create3(&factory, &proxy_hash, &h.salt, &h.address),
            "GPU CREATE3 hit failed host re-derivation"
        );
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}

#[test]
fn createx_finds_and_verifies_low_threshold() {
    use nullforge::gpu::{CREATEX_ADDRESS, STANDARD_CREATE3_PROXY_HASH};

    let _gpu = gpu_guard();
    let ctx = MetalContext::new();
    let proxy_hash = STANDARD_CREATE3_PROXY_HASH;

    let n = 256usize;
    let base_salts: Vec<[u8; 32]> = (0..n)
        .map(|i| {
            let mut s = [0u8; 32];
            s[0] = (i & 0xff) as u8;
            s[1] = (i >> 8) as u8;
            s[23] = 0x22;
            s
        })
        .collect();
    let base_counters: Vec<u64> = vec![0u64; n];

    let hits = ctx.dispatch_createx(&CREATEX_ADDRESS, &proxy_hash, &base_salts, &base_counters, 4096, 2);
    assert!(!hits.is_empty(), "should find >=2-zero CreateX addresses");

    for h in &hits {
        // `h.salt` is the ORIGINAL (un-guarded) salt; verify_createx re-guards
        // it (keccak256(salt)) internally before the two-hop CREATE3 derivation.
        assert!(
            ctx.verify_createx(&CREATEX_ADDRESS, &proxy_hash, &h.salt, &h.address),
            "GPU CreateX hit failed host re-derivation"
        );
        assert!(h.address[0] >> 4 == 0, "claimed leading zero nibble wrong");
    }
}
