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
    assert_eq!(modpow(two, p - two, p), inv2_closed, "modpow(2,p-2) != (p+1)/2");
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
        (p_minus_1, "5"),      // p-1, 5
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
