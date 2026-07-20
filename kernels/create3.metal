// CREATE3 vanity salt mining.
//
//   proxy   = keccak256(0xff ‖ factory[20] ‖ salt[32] ‖ proxyhash[32])[12:32]
//   address = keccak256(0xd6 0x94 ‖ proxy[20] ‖ 0x01)[12:32]
//
// The deployed address depends ONLY on (factory, salt) — the contract's init
// code never enters the derivation (that is CREATE3's defining property). Hop 1
// is the CREATE2 deploy of a fixed proxy (its init-code hash is `proxyhash`);
// hop 2 is the proxy's own CREATE at nonce 1, whose RLP preimage is the constant
// 23 bytes `0xd6 0x94 ‖ proxy ‖ 0x01` (a fresh proxy's first CREATE is nonce 1,
// and a 20-byte address RLP-encodes as `0x94 ‖ addr` — no variable-length case).
//
// Both preimages (85 and 23 bytes) are single Keccak blocks (rate 136), so this
// reuses `keccak256()` from keccak.metal (concatenated before this source).
// keccak-only, no secp256k1 — like the CREATE2 kernel, far cheaper than the EOA
// `mine` kernel.
//
// Per-thread salt derivation is identical to mine_create2: high 24 bytes random
// (base_salts[gid]), low 8 bytes a big-endian counter (base_counters[gid] + it).

// Bounded output ring, identical layout to create2.metal / miner.metal:
// each hit is {salt[32], address[20], zeros(1), pad(11)} in `hits`.
constant uint C3_HIT_STRIDE = 64u;
constant uint C3_MAX_HITS = 1024u;

// Leading-zero-NIBBLE count of a 20-byte address (same semantics as the CPU/EOA
// and CREATE2 paths): a fully-zero byte contributes 2; the first nonzero byte
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
                         device const uchar* proxyhash      [[buffer(1)]],  // 32 bytes
                         device const uchar* base_salts     [[buffer(2)]],  // gid*32
                         device const ulong* base_counters  [[buffer(3)]],  // gid
                         constant uint&      iters          [[buffer(4)]],
                         constant uint&      threshold      [[buffer(5)]],
                         device atomic_uint* hit_count      [[buffer(6)]],
                         device uchar*       hits           [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    // Load the fixed parts of the preimage once.
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

        // hop 1: proxy = keccak256(0xff ‖ factory(20) ‖ salt(32) ‖ proxyhash(32))[12:32]
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
        thread uchar d1[32];
        keccak256(buf1, 85u, d1);
        thread const uchar* proxy = d1 + 12; // low 20 bytes = proxy address

        // hop 2: address = keccak256(0xd6 0x94 ‖ proxy(20) ‖ 0x01)[12:32]
        thread uchar buf2[23];
        buf2[0] = 0xd6u;
        buf2[1] = 0x94u;
        for (uint i = 0; i < 20; i++) {
            buf2[2 + i] = proxy[i];
        }
        buf2[22] = 0x01u;
        thread uchar d2[32];
        keccak256(buf2, 23u, d2);
        thread const uchar* addr = d2 + 12; // low 20 bytes = deployed address

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
