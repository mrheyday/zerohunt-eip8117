// CREATE2 vanity salt mining.
//
//   address = keccak256(0xff ‖ deployer[20] ‖ salt[32] ‖ initcodehash[32])[12:32]
//
// The 85-byte preimage is a single Keccak block (rate = 136), so this reuses
// `keccak256()` from keccak.metal (the host concatenates keccak.metal before
// this source). No secp256k1 — CREATE2 mining is pure keccak, far cheaper per
// candidate than the EOA `mine` kernel.
//
// Per-thread work: `salt` = base_salts[gid] with its low 8 bytes replaced by a
// big-endian counter (base_counters[gid] + it), giving distinct salts across
// (thread, iter, batch). The high 24 bytes of each base salt are random, so
// threads never collide.

// Bounded output ring, identical layout to miner.metal's `mine`:
// each hit is {salt[32], address[20], zeros(1), pad(11)} in `hits`.
constant uint C2_HIT_STRIDE = 64u;
constant uint C2_MAX_HITS = 1024u;

// Leading-zero-NIBBLE count of a 20-byte address (same semantics as the CPU/EOA
// paths): a fully-zero byte contributes 2; the first nonzero byte contributes 1
// iff its top nibble is zero; then stop.
inline uint c2_leading_zero_nibbles(thread const uchar* addr20) {
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

kernel void mine_create2(device const uchar* deployer      [[buffer(0)]],  // 20 bytes
                         device const uchar* initcodehash  [[buffer(1)]],  // 32 bytes
                         device const uchar* base_salts    [[buffer(2)]],  // gid*32
                         device const ulong* base_counters [[buffer(3)]],  // gid
                         constant uint&      iters         [[buffer(4)]],
                         constant uint&      threshold     [[buffer(5)]],
                         device atomic_uint* hit_count     [[buffer(6)]],
                         device uchar*       hits          [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    // Load the fixed parts of the preimage once.
    thread uchar dep[20];
    for (uint i = 0; i < 20; i++) {
        dep[i] = deployer[i];
    }
    thread uchar ich[32];
    for (uint i = 0; i < 32; i++) {
        ich[i] = initcodehash[i];
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

        // preimage = 0xff ‖ deployer(20) ‖ salt(32) ‖ initcodehash(32) = 85 bytes
        thread uchar buf[85];
        buf[0] = 0xffu;
        for (uint i = 0; i < 20; i++) {
            buf[1 + i] = dep[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf[21 + i] = salt[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf[53 + i] = ich[i];
        }

        thread uchar digest[32];
        keccak256(buf, 85u, digest);
        thread const uchar* addr = digest + 12; // low 20 bytes = the address

        uint zeros = c2_leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

        uint idx = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (idx >= C2_MAX_HITS) {
            continue; // batch saturated; host clamps its read too
        }
        uint out = idx * C2_HIT_STRIDE;
        for (uint i = 0; i < 32; i++) {
            hits[out + i] = salt[i];
        }
        for (uint i = 0; i < 20; i++) {
            hits[out + 32 + i] = addr[i];
        }
        hits[out + 52] = (uchar)zeros;
    }
}
