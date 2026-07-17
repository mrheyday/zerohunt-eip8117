#include <metal_stdlib>
using namespace metal;

// Ethereum Keccak-256 (NOT NIST SHA3-256): rate = 1088 bits (136 bytes),
// capacity = 512 bits, domain-separation pad byte = 0x01 (SHA3 uses 0x06).
// Single-block only (inlen <= 135), sufficient for all callers here.

constant ulong RC[24] = {
  0x0000000000000001UL,0x0000000000008082UL,0x800000000000808aUL,0x8000000080008000UL,
  0x000000000000808bUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
  0x000000000000008aUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000aUL,
  0x000000008000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,
  0x8000000000008002UL,0x8000000000000080UL,0x000000000000800aUL,0x800000008000000aUL,
  0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL
};

// Rho rotation offsets, indexed by [x][y] (x + 5*y), standard Keccak layout.
constant uint RHO[25] = {
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14
};

inline ulong rotl64(ulong x, uint n) {
    n &= 63;
    return n == 0 ? x : ((x << n) | (x >> (64u - n)));
}

inline void keccak_f1600(thread ulong* state) {
    for (uint round = 0; round < 24; round++) {
        // Theta
        ulong C[5];
        for (uint x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        ulong D[5];
        for (uint x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (uint x = 0; x < 5; x++) {
            for (uint y = 0; y < 5; y++) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Rho + Pi
        ulong B[25];
        for (uint x = 0; x < 5; x++) {
            for (uint y = 0; y < 5; y++) {
                uint newX = y;
                uint newY = (2 * x + 3 * y) % 5;
                B[newX + 5 * newY] = rotl64(state[x + 5 * y], RHO[x + 5 * y]);
            }
        }

        // Chi
        for (uint x = 0; x < 5; x++) {
            for (uint y = 0; y < 5; y++) {
                state[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
            }
        }

        // Iota
        state[0] ^= RC[round];
    }
}

// inlen must be <= 135 (single block; rate = 136 bytes).
void keccak256(thread const uchar* in, uint inlen, thread uchar* out32) {
    thread ulong state[25];
    for (uint i = 0; i < 25; i++) {
        state[i] = 0UL;
    }

    // Absorb: XOR input bytes little-endian into the first 136 bytes (17 lanes) of state.
    thread uchar* stateBytes = (thread uchar*)state;
    for (uint i = 0; i < inlen; i++) {
        stateBytes[i] ^= in[i];
    }

    // Domain-separation pad (Keccak = 0x01, NOT SHA3's 0x06) at byte `inlen`,
    // and the rate-end bit at byte 135.
    stateBytes[inlen] ^= 0x01u;
    stateBytes[135] ^= 0x80u;

    keccak_f1600(state);

    // Squeeze: first 32 bytes little-endian.
    thread const uchar* outBytes = (thread const uchar*)state;
    for (uint i = 0; i < 32; i++) {
        out32[i] = outBytes[i];
    }
}

kernel void keccak_test(device const uchar* inputs [[buffer(0)]],
                         device const uint* lens [[buffer(1)]],
                         device uchar* out [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint STRIDE = 64;
    thread uchar buf[64];
    uint inlen = lens[gid];
    for (uint i = 0; i < STRIDE; i++) {
        buf[i] = inputs[gid * STRIDE + i];
    }
    thread uchar out32[32];
    keccak256(buf, inlen, out32);
    for (uint i = 0; i < 32; i++) {
        out[gid * 32 + i] = out32[i];
    }
}
