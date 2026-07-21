//! Host-side CREATE3 (and CREATE2) address-derivation vectors.
//!
//! These need no GPU — they reproduce the exact preimage bytes that
//! `kernels/create3.metal` hashes and that `MetalContext::verify_create3`
//! re-derives, and assert them against fixed expected addresses. The expected
//! values were cross-checked against Alloy's canonical `Address::create2` /
//! `Address::create`, so this test pins the kernel's math independently of the
//! Metal toolchain.

use ethers::utils::keccak256;

/// CREATE2: keccak256(0xff ‖ deployer ‖ salt ‖ initcodehash)[12:]
fn create2(deployer: &[u8; 20], salt: &[u8; 32], ich: &[u8; 32]) -> [u8; 20] {
    let mut pre = Vec::with_capacity(85);
    pre.push(0xff);
    pre.extend_from_slice(deployer);
    pre.extend_from_slice(salt);
    pre.extend_from_slice(ich);
    let mut a = [0u8; 20];
    a.copy_from_slice(&keccak256(&pre)[12..32]);
    a
}

/// CREATE3: proxy = CREATE2(factory, salt, proxyhash); then the proxy's nonce-1
/// CREATE = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:].
fn create3(factory: &[u8; 20], salt: &[u8; 32], proxyhash: &[u8; 32]) -> [u8; 20] {
    let proxy = create2(factory, salt, proxyhash);
    let mut rlp = Vec::with_capacity(23);
    rlp.push(0xd6);
    rlp.push(0x94);
    rlp.extend_from_slice(&proxy);
    rlp.push(0x01);
    let mut a = [0u8; 20];
    a.copy_from_slice(&keccak256(&rlp)[12..32]);
    a
}

fn hex20(s: &str) -> [u8; 20] {
    ethers::utils::hex::decode(s).unwrap().try_into().unwrap()
}
fn hex32(s: &str) -> [u8; 32] {
    ethers::utils::hex::decode(s).unwrap().try_into().unwrap()
}

#[test]
fn create2_matches_canonical_vector() {
    // deployer = canonical CREATE2 factory; salt = 0x11..; ich = 0x22..
    let got = create2(
        &hex20("4e59b44847b379578588920cA78FbF26c0B4956C"),
        &[0x11u8; 32],
        &[0x22u8; 32],
    );
    // Cross-checked against alloy_primitives::Address::create2.
    assert_eq!(
        got,
        hex20("95C869102925686D60db5b5b08B1766E44Ab8FC3")
    );
}

#[test]
fn proxy_initcode_hash_is_correct() {
    // The Solmate/0xSequence CREATE3 proxy initcode and its keccak hash.
    let initcode = ethers::utils::hex::decode("67363d3d37363d34f03d5260086018f3").unwrap();
    assert_eq!(
        keccak256(&initcode),
        hex32("21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f"),
    );
}

#[test]
fn create3_matches_canonical_two_step() {
    // factory = 0x..42; salt = 0x33..; proxyhash = Solmate/0xSequence.
    let got = create3(
        &hex20("0000000000000000000000000000000000000042"),
        &[0x33u8; 32],
        &hex32("21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f"),
    );
    // Cross-checked against alloy: factory.create2(salt, proxyhash).create(1).
    assert_eq!(
        got,
        hex20("A263972e952261862D1230316BC591A1532C70dC")
    );
}

#[test]
fn createx_permissionless_guard_and_address() {
    // CreateX permissionless branch: guardedSalt = keccak256(abi.encode(salt)),
    // and abi.encode(bytes32) is the identity, so guardedSalt = keccak256(salt).
    let createx = hex20("ba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed");
    let proxy_hash = hex32("21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f");
    let salt = [0x55u8; 32];
    let guarded: [u8; 32] = keccak256(salt);
    let addr = create3(&createx, &guarded, &proxy_hash);
    // Cross-checked against alloy: CreateX.create2(keccak256(salt), proxyhash).create(1).
    assert_eq!(addr, hex20("6AB3688850Fa6A6c22016E5788a8B90B9ffe6967"));
}

#[test]
fn create3_is_bytecode_independent() {
    // The whole point: the final address does not depend on the deployed
    // contract's init code — only on (factory, salt). Same (factory, salt) with
    // the same proxy hash always yields the same address.
    let f = hex20("00000000000000000000000000000000deadbeef");
    let salt = [0xabu8; 32];
    let ph = hex32("21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f");
    assert_eq!(create3(&f, &salt, &ph), create3(&f, &salt, &ph));
}
