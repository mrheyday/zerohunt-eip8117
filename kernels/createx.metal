// CreateX (pcaversaccio) permissionless CREATE3 vanity salt mining.
//
// CreateX is the widely-deployed cross-chain deployer at a fixed address on
// every chain. Its `deployCreate3(salt, initCode)` GUARDS the salt before the
// CREATE3 deployment, so the effective CREATE2 salt is not the raw salt you
// pass. For the permissionless / cross-chain case (salt[0:20] is neither the
// caller nor the zero-address sentinel, i.e. CreateX's `SenderBytes.Random`),
// the guard is simply:
//
//   guardedSalt = keccak256(abi.encode(salt)) = keccak256(salt)   // 32-byte preimage
//
// Then CreateX deploys the standard CREATE3 proxy (same 16-byte init code as
// Solmate/0xSequence) FROM ITS OWN ADDRESS:
//
//   proxy = keccak256(0xff ‖ CREATEX ‖ guardedSalt ‖ PROXY_INITCODEHASH)[12:]
//   addr  = keccak256(0xd6 ‖ 0x94 ‖ proxy ‖ 0x01)[12:]
//
// The emitted value is the ORIGINAL salt (what you pass to
// `CreateX.deployCreate3`) — the guard is re-applied on-chain. This variant
// covers ONLY the permissionless branch; CreateX's msg.sender-permissioned and
// redeploy-protected branches mix in the sender/chain-id (different address per
// chain, or a constrained salt layout) and are intentionally out of scope here.
//
// 3 keccaks per candidate (guard, proxy, final); still no secp256k1. The host
// concatenates keccak.metal before this source. Hit-record layout matches the
// other salt miners.

constant uint CX_HIT_STRIDE = 64u;   // {salt[32], address[20], zeros(1), pad(11)}
constant uint CX_MAX_HITS   = 1024u;

inline uint cx_leading_zero_nibbles(thread const uchar* addr20) {
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

kernel void mine_createx(device const uchar* createx       [[buffer(0)]],  // 20 bytes (CreateX addr)
                         device const uchar* proxyhash     [[buffer(1)]],  // 32 bytes
                         device const uchar* base_salts    [[buffer(2)]],  // gid*32
                         device const ulong* base_counters [[buffer(3)]],  // gid
                         constant uint&      iters         [[buffer(4)]],
                         constant uint&      threshold     [[buffer(5)]],
                         device atomic_uint* hit_count     [[buffer(6)]],
                         device uchar*       hits          [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    thread uchar cx[20];
    for (uint i = 0; i < 20; i++) {
        cx[i] = createx[i];
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
            salt[i] = base[i]; // high 24 bytes random -> ~never the sender/zero
        }
        ulong ctr = base_ctr + (ulong)it;
        for (uint i = 0; i < 8; i++) {
            salt[31 - i] = (uchar)((ctr >> (8 * i)) & 0xFFu);
        }

        // ── Guard: guardedSalt = keccak256(salt) (permissionless branch) ──
        thread uchar guarded[32];
        keccak256(salt, 32u, guarded);

        // ── proxy = CREATE2(CreateX, guardedSalt, PROXY_INITCODEHASH) ──
        thread uchar buf1[85];
        buf1[0] = 0xffu;
        for (uint i = 0; i < 20; i++) {
            buf1[1 + i] = cx[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf1[21 + i] = guarded[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf1[53 + i] = ph[i];
        }
        thread uchar digest1[32];
        keccak256(buf1, 85u, digest1);
        thread const uchar* proxy = digest1 + 12;

        // ── addr = proxy's nonce-1 CREATE ──────────────────────────────
        thread uchar buf2[23];
        buf2[0] = 0xd6u;
        buf2[1] = 0x94u;
        for (uint i = 0; i < 20; i++) {
            buf2[2 + i] = proxy[i];
        }
        buf2[22] = 0x01u;
        thread uchar digest2[32];
        keccak256(buf2, 23u, digest2);
        thread const uchar* addr = digest2 + 12;

        uint zeros = cx_leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

        uint idx = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (idx >= CX_MAX_HITS) {
            continue;
        }
        uint out = idx * CX_HIT_STRIDE;
        for (uint i = 0; i < 32; i++) {
            hits[out + i] = salt[i]; // emit the ORIGINAL salt (guard re-applied on-chain)
        }
        for (uint i = 0; i < 20; i++) {
            hits[out + 32 + i] = addr[i];
        }
        hits[out + 52] = (uchar)zeros;
    }
}
