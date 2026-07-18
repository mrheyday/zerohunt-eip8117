// Full derive(seed,counter) -> (privkey,address) pipeline.
//
// This file is ALWAYS compiled concatenated AFTER kernels/keccak.metal,
// kernels/field.metal, and kernels/ec.metal (the host builds the source via
// `format!("{}\n{}\n{}\n{}", keccak, field, ec, miner)`), so it reuses
// keccak256() (T2), the `fe` type + fe_add/fe_sub/fe_mul/fe_inv (T3), and
// scalarmul()/fe_is_zero() (T4) from those files without re-declaring them.

// secp256k1 curve order n, little-endian u32 limbs (limb[0] = least
// significant word), matching the `fe`/`P` limb layout in field.metal:
// n = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
constant uint SECP_N[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// Convert a 32-byte big-endian byte array into an `fe` (little-endian u32
// limbs, limb[0] = least-significant word) -- the inverse of the host's
// `u256_to_limbs`/`run_scalarmul` marshalling, done on-device.
inline fe bytes_be_to_fe(thread const uchar* b) {
    fe r;
    for (int limb = 0; limb < 8; limb++) {
        uint hi = 28 - limb * 4; // limb 0 = last 4 (least-significant) bytes
        r.v[limb] = ((uint)b[hi] << 24) | ((uint)b[hi + 1] << 16) |
                    ((uint)b[hi + 2] << 8) | (uint)b[hi + 3];
    }
    return r;
}

// Convert an `fe` back into 32 big-endian bytes (inverse of bytes_be_to_fe).
inline void fe_to_bytes_be(fe a, thread uchar* out) {
    for (int limb = 0; limb < 8; limb++) {
        uint pos = 28 - limb * 4;
        uint v = a.v[limb];
        out[pos] = (uchar)(v >> 24);
        out[pos + 1] = (uchar)(v >> 16);
        out[pos + 2] = (uchar)(v >> 8);
        out[pos + 3] = (uchar)v;
    }
}

// true iff 0 < k < SECP_N (the valid secp256k1 private-key scalar range).
inline bool scalar_in_range(thread const fe& k) {
    if (fe_is_zero(k)) {
        return false;
    }
    for (int i = 7; i >= 0; i--) {
        if (k.v[i] != SECP_N[i]) {
            return k.v[i] < SECP_N[i];
        }
    }
    return false; // k == SECP_N is out of range (n is not a valid scalar)
}

// privkey = keccak256(seed[32] || counter_le8), returned as an (unreduced --
// see scalar_in_range) field element. Caller applies the scalar-range guard.
inline fe derive_privkey(thread const uchar* seed, ulong counter) {
    thread uchar buf[40];
    for (uint i = 0; i < 32; i++) {
        buf[i] = seed[i];
    }
    for (uint i = 0; i < 8; i++) {
        buf[32 + i] = (uchar)((counter >> (8 * i)) & 0xFFu);
    }
    thread uchar digest[32];
    keccak256(buf, 40, digest);
    return bytes_be_to_fe(digest);
}

// address = keccak256(x_be || y_be)[12..32], the standard Ethereum
// pubkey->address derivation (last 20 bytes of the hash of the uncompressed,
// prefix-less 64-byte point encoding).
inline void pubkey_to_address(fe x, fe y, thread uchar* out20) {
    thread uchar xy[64];
    fe_to_bytes_be(x, xy);
    fe_to_bytes_be(y, xy + 32);
    thread uchar digest[32];
    keccak256(xy, 64, digest);
    for (uint i = 0; i < 20; i++) {
        out20[i] = digest[12 + i];
    }
}

// One thread per (seed,counter): derive the privkey, guard its scalar range,
// then (if in range) scalarmul + hash to the address. seeds are 32 bytes
// each at seeds[gid*32], counters are one ulong each at counters[gid].
// out_priv gets the raw 32-byte big-endian privkey unconditionally (even out
// of range, so callers can inspect what was derived); out_addr gets the
// 20-byte address, or all-zero to mark a scalar-range-guard miss (skip).
kernel void derive_test(device const uchar* seeds [[buffer(0)]],
                         device const ulong* counters [[buffer(1)]],
                         device uchar* out_priv [[buffer(2)]],
                         device uchar* out_addr [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    thread uchar seed[32];
    for (uint i = 0; i < 32; i++) {
        seed[i] = seeds[gid * 32 + i];
    }
    ulong counter = counters[gid];

    fe priv = derive_privkey(seed, counter);

    thread uchar privBytes[32];
    fe_to_bytes_be(priv, privBytes);
    for (uint i = 0; i < 32; i++) {
        out_priv[gid * 32 + i] = privBytes[i];
    }

    if (!scalar_in_range(priv)) {
        for (uint i = 0; i < 20; i++) {
            out_addr[gid * 20 + i] = 0;
        }
        return;
    }

    fe x, y;
    scalarmul(priv, x, y);
    thread uchar addr[20];
    pubkey_to_address(x, y, addr);
    for (uint i = 0; i < 20; i++) {
        out_addr[gid * 20 + i] = addr[i];
    }
}

// Leading-zero-NIBBLE count of a 20-byte address, matching the CPU tool's
// (src/main.rs) semantics exactly: walk bytes; a fully-zero byte contributes
// 2 (both nibbles), then continue; the first nonzero byte contributes 1 iff
// its top nibble is zero (equivalent to `byte.leading_zeros() / 4` on a u8 --
// that expression is 1 for byte < 0x10 and 0 for byte >= 0x10), then stop.
// Bytes after the first nonzero byte are never inspected (matches the CPU
// `break`).
inline uint leading_zero_nibbles(thread const uchar* addr20) {
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

// Bounded output ring for `mine`: each hit is a fixed HIT_STRIDE-byte record
// {privkey[32], address[20], zeros(1), pad(11)} in `hits`, appended via
// atomic_fetch_add on `hit_count`. Capacity is MAX_HITS records; slots beyond
// that are silently dropped (the atomic counter still reflects the true
// number found so the host can detect a saturated batch), so the host must
// clamp its read to `min(hit_count, MAX_HITS)`.
constant uint HIT_STRIDE = 64u;
constant uint MAX_HITS = 1024u;

// One thread per (seed,base_counter): iterate `iters` candidate keys
// (counter = base_counters[gid] + it, it in [0, iters)), deriving + guarding
// + scalar-multiplying + hashing exactly as derive_test does, and appending
// any hit with >= threshold leading-zero nibbles to the bounded `hits`
// buffer. Guard misses (privkey == 0 or >= secp256k1 order) are skipped, not
// counted as candidates.
kernel void mine(device const uchar* seeds [[buffer(0)]],
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
    ulong base = base_counters[gid];

    for (uint it = 0; it < iters; it++) {
        fe priv = derive_privkey(seed, base + (ulong)it);
        if (!scalar_in_range(priv)) {
            continue;
        }

        fe x, y;
        scalarmul(priv, x, y);
        thread uchar addr[20];
        pubkey_to_address(x, y, addr);

        uint zeros = leading_zero_nibbles(addr);
        if (zeros < threshold) {
            continue;
        }

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
