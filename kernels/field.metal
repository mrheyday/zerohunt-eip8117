#include <metal_stdlib>
using namespace metal;

// secp256k1 field arithmetic mod p over 8x u32 limbs, little-endian limb order
// (v[0] = least-significant 32 bits). Value is always kept reduced in [0, p).
//
// p = 2^256 - 2^32 - 977 = 0xFFFFFFFF...FFFFFEFFFFFC2F
// 2^256 mod p = 2^32 + 977 = 0x1000003D1 = R  (fast-reduction fold constant).

typedef struct { uint v[8]; } fe;

constant uint P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// p - 2, the Fermat inverse exponent, little-endian limbs.
constant uint PM2[8] = {
    0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// true iff a >= P (both 8-limb, little-endian).
inline bool ge_p(thread const fe& a) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] != P[i]) {
            return a.v[i] > P[i];
        }
    }
    return true; // equal => >= P
}

// a -= P in place (assumes a >= P so no final borrow escapes the low 256 bits;
// also used in the add carry==1 path where the discarded borrow is intended).
inline void sub_p(thread fe& a) {
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = (ulong)a.v[i] - (ulong)P[i] - borrow;
        a.v[i] = (uint)d;
        borrow = (d >> 32) & 1u; // 1 if underflow
    }
}

fe fe_add(fe a, fe b) {
    fe r;
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + carry;
        r.v[i] = (uint)s;
        carry = s >> 32;
    }
    // a,b < p  =>  a+b < 2p < 2^257. If carry set, a+b in [2^256, 2p) and the
    // low limbs are < p, so a single sub_p yields a+b-p (the discarded borrow
    // is the +2^256). Otherwise conditionally subtract when >= p.
    if (carry) {
        sub_p(r);
    } else if (ge_p(r)) {
        sub_p(r);
    }
    return r;
}

fe fe_sub(fe a, fe b) {
    fe r;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - borrow;
        r.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    if (borrow) {
        // a < b: result wrapped by 2^256; adding P restores a - b + p in [0, p).
        ulong carry = 0;
        for (int i = 0; i < 8; i++) {
            ulong s = (ulong)r.v[i] + (ulong)P[i] + carry;
            r.v[i] = (uint)s;
            carry = s >> 32;
        }
    }
    return r;
}

// One fast-reduction fold: replace the high half (t[8..15]) H by H*R added to
// the low half, using R = 2^32 + 977. H*R = (H << 32) + H*977. Repeated until
// t[8..15] == 0. t is a 16-limb little-endian accumulator.
inline void fold_once(thread uint t[16]) {
    uint H[8];
    for (int k = 0; k < 8; k++) {
        H[k] = t[8 + k];
        t[8 + k] = 0;
    }
    // t += H * 977  (offset 0)
    ulong carry = 0;
    for (int k = 0; k < 8; k++) {
        ulong s = (ulong)t[k] + (ulong)H[k] * 977UL + carry;
        t[k] = (uint)s;
        carry = s >> 32;
    }
    for (int k = 8; carry != 0 && k < 16; k++) {
        ulong s = (ulong)t[k] + carry;
        t[k] = (uint)s;
        carry = s >> 32;
    }
    // t += H << 32  (offset 1): add H[k] into limb k+1
    carry = 0;
    for (int k = 0; k < 8; k++) {
        ulong s = (ulong)t[k + 1] + (ulong)H[k] + carry;
        t[k + 1] = (uint)s;
        carry = s >> 32;
    }
    for (int k = 9; carry != 0 && k < 16; k++) {
        ulong s = (ulong)t[k] + carry;
        t[k] = (uint)s;
        carry = s >> 32;
    }
}

inline bool hi_nonzero(thread const uint t[16]) {
    for (int i = 8; i < 16; i++) {
        if (t[i] != 0) return true;
    }
    return false;
}

fe fe_mul(fe a, fe b) {
    // 8x8 schoolbook into 16 limbs.
    uint prod[16];
    for (int i = 0; i < 16; i++) prod[i] = 0;
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong s = (ulong)a.v[i] * (ulong)b.v[j] + (ulong)prod[i + j] + carry;
            prod[i + j] = (uint)s;
            carry = s >> 32;
        }
        prod[i + 8] = (uint)carry; // untouched slot for this i => assign
    }
    // Fold the top 256 bits into the bottom until only 8 limbs remain.
    while (hi_nonzero(prod)) {
        fold_once(prod);
    }
    fe r;
    for (int i = 0; i < 8; i++) r.v[i] = prod[i];
    // Now r < 2^256 < 2p; final conditional subtract(s) into [0, p).
    while (ge_p(r)) {
        sub_p(r);
    }
    return r;
}

// Fermat inverse: a^(p-2) mod p, square-and-multiply MSB-first over PM2.
fe fe_inv(fe a) {
    fe result;
    result.v[0] = 1u;
    for (int i = 1; i < 8; i++) result.v[i] = 0u;
    for (int bit = 255; bit >= 0; bit--) {
        result = fe_mul(result, result);
        uint limb = PM2[bit >> 5];
        uint b = (limb >> (bit & 31)) & 1u;
        if (b) {
            result = fe_mul(result, a);
        }
    }
    return result;
}

// Applies op (0=add,1=sub,2=mul,3=inv) to the gid-th 8-limb operand pair.
kernel void field_test(device const uint* a   [[buffer(0)]],
                       device const uint* b   [[buffer(1)]],
                       device uint* out       [[buffer(2)]],
                       device const uint* op  [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    fe fa, fb;
    for (int i = 0; i < 8; i++) {
        fa.v[i] = a[gid * 8 + i];
        fb.v[i] = b[gid * 8 + i];
    }
    fe r;
    switch (op[gid]) {
        case 0: r = fe_add(fa, fb); break;
        case 1: r = fe_sub(fa, fb); break;
        case 2: r = fe_mul(fa, fb); break;
        default: r = fe_inv(fa); break; // 3 = inv
    }
    for (int i = 0; i < 8; i++) {
        out[gid * 8 + i] = r.v[i];
    }
}
