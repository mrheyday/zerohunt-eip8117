// Tagged-CREATE2 vanity mining: three-stage salt derivation used by factories
// that wrap a caller-chosen "tag" through a stable per-deployer salt before
// CREATE2, e.g. mev-arbitrum's `MevSafeFactory`:
//
//   userSalt      = keccak256(prefix ‖ deployer ‖ tag)                 // packed
//   effectiveSalt = keccak256(abi.encode(owner, permissions, userSalt, deployer))
//   address       = keccak256(0xff ‖ factory ‖ effectiveSalt ‖ initCodeHash)[12:]
//
// `prefix` is caller-supplied bytes (e.g. `"MevSafe.v2:"` from
// `MevSafeFactory.saltFor`); `owner`/`permissions`/`deployer` are ABI-encoded
// as left-padded 32-byte words per Solidity's `abi.encode`. The mined value is
// `tag` — it is what gets passed to the factory's existing salt-derivation
// path unmodified (no on-chain code changes needed to consume a hit).
//
// 3 keccaks per candidate, same cost class as createx.metal. The host
// concatenates keccak.metal before this source.

constant uint C2T_HIT_STRIDE = 64u;   // {tag[32], address[20], zeros(1), pad(11)}
constant uint C2T_MAX_HITS   = 1024u;
constant uint C2T_MAX_PREFIX = 32u;   // prefix buffer capacity; prefix_len must be <= this

inline uint c2t_leading_zero_nibbles(thread const uchar* addr20) {
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

// Write a 20-byte address as an ABI-encoded (left-padded) 32-byte word.
inline void c2t_pad_address(thread const uchar* addr20, thread uchar* out32) {
    for (uint i = 0; i < 12; i++) {
        out32[i] = 0u;
    }
    for (uint i = 0; i < 20; i++) {
        out32[12 + i] = addr20[i];
    }
}

kernel void mine_create2tag(device const uchar* prefix        [[buffer(0)]],  // C2T_MAX_PREFIX bytes, first prefix_len used
                            device const uint&  prefix_len    [[buffer(1)]],
                            device const uchar* deployer      [[buffer(2)]],  // 20 bytes
                            device const uchar* owner         [[buffer(3)]],  // 20 bytes
                            device const uchar* permissions   [[buffer(4)]],  // 20 bytes
                            device const uchar* factory       [[buffer(5)]],  // 20 bytes
                            device const uchar* initcodehash  [[buffer(6)]],  // 32 bytes
                            device const uchar* base_tags     [[buffer(7)]],  // gid*32
                            device const ulong* base_counters [[buffer(8)]],  // gid
                            constant uint&      iters         [[buffer(9)]],
                            constant uint&      threshold     [[buffer(10)]],
                            device atomic_uint* hit_count     [[buffer(11)]],
                            device uchar*       hits          [[buffer(12)]],
                            uint gid [[thread_position_in_grid]]) {
    thread uchar pfx[C2T_MAX_PREFIX];
    for (uint i = 0; i < C2T_MAX_PREFIX; i++) {
        pfx[i] = prefix[i];
    }
    uint pfx_len = min(prefix_len, C2T_MAX_PREFIX);

    thread uchar dep[20];
    for (uint i = 0; i < 20; i++) {
        dep[i] = deployer[i];
    }
    thread uchar dep32[32];
    c2t_pad_address(dep, dep32);

    thread uchar own32[32];
    {
        thread uchar own[20];
        for (uint i = 0; i < 20; i++) {
            own[i] = owner[i];
        }
        c2t_pad_address(own, own32);
    }

    thread uchar perm32[32];
    {
        thread uchar perm[20];
        for (uint i = 0; i < 20; i++) {
            perm[i] = permissions[i];
        }
        c2t_pad_address(perm, perm32);
    }

    thread uchar fac[20];
    for (uint i = 0; i < 20; i++) {
        fac[i] = factory[i];
    }
    thread uchar ich[32];
    for (uint i = 0; i < 32; i++) {
        ich[i] = initcodehash[i];
    }

    thread uchar base[32];
    for (uint i = 0; i < 32; i++) {
        base[i] = base_tags[gid * 32 + i];
    }
    ulong base_ctr = base_counters[gid];

    for (uint it = 0; it < iters; it++) {
        thread uchar tag[32];
        for (uint i = 0; i < 24; i++) {
            tag[i] = base[i]; // high 24 bytes random -> per-thread uniqueness
        }
        ulong ctr = base_ctr + (ulong)it;
        for (uint i = 0; i < 8; i++) {
            tag[31 - i] = (uchar)((ctr >> (8 * i)) & 0xFFu);
        }

        // ── Stage 1: userSalt = keccak256(prefix ‖ deployer ‖ tag) (packed) ──
        thread uchar buf1[C2T_MAX_PREFIX + 20 + 32];
        for (uint i = 0; i < pfx_len; i++) {
            buf1[i] = pfx[i];
        }
        for (uint i = 0; i < 20; i++) {
            buf1[pfx_len + i] = dep[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf1[pfx_len + 20 + i] = tag[i];
        }
        thread uchar userSalt[32];
        keccak256(buf1, pfx_len + 20u + 32u, userSalt);

        // ── Stage 2: effectiveSalt = keccak256(abi.encode(owner, permissions, userSalt, deployer)) ──
        thread uchar buf2[128];
        for (uint i = 0; i < 32; i++) {
            buf2[i] = own32[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf2[32 + i] = perm32[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf2[64 + i] = userSalt[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf2[96 + i] = dep32[i];
        }
        thread uchar effectiveSalt[32];
        keccak256(buf2, 128u, effectiveSalt);

        // ── Stage 3: address = CREATE2(factory, effectiveSalt, initCodeHash) ──
        thread uchar buf3[85];
        buf3[0] = 0xffu;
        for (uint i = 0; i < 20; i++) {
            buf3[1 + i] = fac[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf3[21 + i] = effectiveSalt[i];
        }
        for (uint i = 0; i < 32; i++) {
            buf3[53 + i] = ich[i];
        }
        thread uchar digest[32];
        keccak256(buf3, 85u, digest);
        thread const uchar* addr = digest + 12;

        uint zeros = c2t_leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

        uint idx = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (idx >= C2T_MAX_HITS) {
            continue;
        }
        uint out = idx * C2T_HIT_STRIDE;
        for (uint i = 0; i < 32; i++) {
            hits[out + i] = tag[i]; // emit the mined TAG (input to factory.saltFor)
        }
        for (uint i = 0; i < 20; i++) {
            hits[out + 32 + i] = addr[i];
        }
        hits[out + 52] = (uchar)zeros;
    }
}
