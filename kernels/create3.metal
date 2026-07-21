// CREATE3 vanity salt mining.
//
// CREATE3 deploys a fixed, constant proxy via CREATE2, and that proxy then does
// a nonce-1 CREATE of the real contract. Because the proxy init code is
// constant, the final address depends ONLY on (factory, salt) — it is
// independent of the deployed contract's bytecode. That is the whole point:
// the same salt yields the same address on every chain regardless of initcode.
//
//   proxy = keccak256(0xff ‖ factory[20] ‖ salt[32] ‖ PROXY_INITCODEHASH[32])[12:]
//   addr  = keccak256(0xd6 ‖ 0x94 ‖ proxy[20] ‖ 0x01)[12:]   // proxy's nonce-1 CREATE
//
// The second preimage is the RLP of [proxy_address, nonce=1]:
//   0xd6            list header  (0xc0 + 22-byte payload)
//   0x94            string header(0x80 + 20-byte address)
//   proxy[20]
//   0x01            nonce = 1    (a single byte < 0x80 encodes as itself)
//
// PROXY_INITCODEHASH is passed as buffer(1) (NOT baked in) because it is
// FACTORY-SPECIFIC: the Solmate/0xSequence proxy differs from Solady's and from
// CreateX's, and CreateX additionally guards the salt. The host binds the hash
// for the configured factory, and `verify_create3` re-derives every hit on the
// CPU (a mismatch hard-aborts), so a wrong hash fails loudly instead of mining
// garbage. Two keccaks per candidate (vs CREATE2's one), still no secp256k1.
//
// The host concatenates keccak.metal before this source, so `keccak256()` is in
// scope. Buffer layout + hit record layout are identical to `mine_create2`, so
// the host dispatch/read path is a near-copy of `dispatch_create2`.

constant uint C3_HIT_STRIDE = 64u;   // {salt[32], address[20], zeros(1), pad(11)}
constant uint C3_MAX_HITS   = 1024u;

// Leading-zero-NIBBLE count of a 20-byte address (identical semantics to the
// CREATE2 / EOA paths): a fully-zero byte contributes 2; the first nonzero byte
// contributes 1 iff its top nibble is zero; then stop.
inline uint c3_leading_zero_nibbles(thread const uchar* addr20) {
    uint zeros = 0;
    for (uint b = 0; b < 20; b++) {
        if (addr20[b] == 0) {
            zeros += 2;
        } else {
            if ((addr20[b] >> 4) == 0) {
                zeros += 1;
            }
            break;
        }
    }
    return zeros;
}

kernel void mine_create3(device const uchar* factory       [[buffer(0)]],  // 20 bytes
                         device const uchar* proxyhash     [[buffer(1)]],  // 32 bytes (PROXY_INITCODEHASH)
                         device const uchar* base_salts    [[buffer(2)]],  // gid*32
                         device const ulong* base_counters [[buffer(3)]],  // gid
                         constant uint&      iters         [[buffer(4)]],
                         constant uint&      threshold     [[buffer(5)]],
                         device atomic_uint* hit_count     [[buffer(6)]],
                         device uchar*       hits          [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    // Fixed parts of the first preimage, loaded once.
    thread uchar fac[20];
    for (uint i = 0; i < 20; i++) {
        fac[i] = factory[i];
    }
    thread uchar ph[32];
    for (uint i = 0; i < 32; i++) {
        ph[i] = proxyhash[i];
    }
    thread uchar base[32];
    for (uint i = 0; i < 32; i++) {
        base[i] = base_salts[gid * 32 + i];
    }
    ulong base_ctr = base_counters[gid];

    for (uint it = 0; it < iters; it++) {
        thread uchar salt[32];
        for (uint i = 0; i < 24; i++) {
            salt[i] = base[i]; // high 24 bytes: per-thread random uniqueness
        }
        ulong ctr = base_ctr + (ulong)it; // low 8 bytes: big-endian counter
        for (uint i = 0; i < 8; i++) {
            salt[31 - i] = (uchar)((ctr >> (8 * i)) & 0xFFu);
        }

        // ── Step 1: proxy = CREATE2(factory, salt, PROXY_INITCODEHASH) ──
        // preimage = 0xff ‖ factory(20) ‖ salt(32) ‖ proxyhash(32) = 85 bytes
        thread uchar buf1[85];
        buf1[0] = 0xffu;
        for (uint i = 0; i < 20; i++) {
            buf1[1 + i] = fac[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf1[21 + i] = salt[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf1[53 + i] = ph[i];
        }
        thread uchar digest1[32];
        keccak256(buf1, 85u, digest1);
        thread const uchar* proxy = digest1 + 12; // low 20 bytes

        // ── Step 2: addr = proxy's nonce-1 CREATE ──────────────────────
        // preimage = 0xd6 ‖ 0x94 ‖ proxy(20) ‖ 0x01 = 23 bytes
        thread uchar buf2[23];
        buf2[0] = 0xd6u;
        buf2[1] = 0x94u;
        for (uint i = 0; i < 20; i++) {
            buf2[2 + i] = proxy[i];
        }
        buf2[22] = 0x01u;
        thread uchar digest2[32];
        keccak256(buf2, 23u, digest2);
        thread const uchar* addr = digest2 + 12; // low 20 bytes = final address

        uint zeros = c3_leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

        uint idx = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (idx >= C3_MAX_HITS) {
            continue; // batch saturated; host clamps its read too
        }
        uint out = idx * C3_HIT_STRIDE;
        for (uint i = 0; i < 32; i++) {
            hits[out + i] = salt[i];
        }
        for (uint i = 0; i < 20; i++) {
            hits[out + 32 + i] = addr[i];
        }
        hits[out + 52] = (uchar)zeros;
    }
}
