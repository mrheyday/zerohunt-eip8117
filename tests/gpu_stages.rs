use zerohunt::gpu::MetalContext;

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

#[test]
fn keccak_matches_host_reference() {
    use ethers::utils::keccak256;
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
    let hits = ctx.dispatch_mine(&seeds, &base, 4096, 2); // >=2 leading zero nibbles
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
// End-to-end: the unified GPU driver (run_batches) dispatching real Metal
// batches through the shared state funnel, not just the raw kernel.
// ---------------------------------------------------------------------------

/// End-to-end: the GPU driver, given a low target, finds a verified hit, writes
/// the file, and trips `stop`. Mirrors the device-gated style of the other GPU
/// tests in this file (runs on the Metal device present in CI/dev machines).
#[test]
fn gpu_driver_finds_and_reports_low_target() {
    use std::sync::Arc;
    use std::time::Instant;
    use zerohunt::gpu::MetalContext;
    use zerohunt::miner::gpu_driver::{run_batches, N_THREADS};
    use zerohunt::miner::shared::MinerShared;

    let ctx = MetalContext::new();

    let file = tempfile::NamedTempFile::new().unwrap();
    // target 2 => the very first batch should surface >=2-zero hits fast.
    let shared = Arc::new(MinerShared::new(
        2,
        file.reopen().unwrap(),
        Instant::now(),
        zerohunt::keyenc::KeySink::RevealPlaintext,
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
