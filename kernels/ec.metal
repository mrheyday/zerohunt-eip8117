// secp256k1 EC scalar multiplication (k*G -> affine public key) in MSL.
//
// This file is ALWAYS compiled concatenated AFTER kernels/field.metal (the
// host builds the source via `format!("{}\n{}", field, ec)`), so it reuses
// that file's `#include <metal_stdlib>`, `using namespace metal;`, the `fe`
// type, the reduced-mod-p invariant, and `fe_add/fe_sub/fe_mul/fe_inv`.
//
// Curve: y^2 = x^3 + 7 over F_p (a = 0), so Jacobian formulas specialize to
// the a=0 EFD forms: doubling = dbl-2009-l, addition = add-2007-bl. Points
// are Jacobian (X,Y,Z) with affine (x,y) = (X/Z^2, Y/Z^3); Z == 0 is the
// point at infinity. All field elements are 8x u32 little-endian limbs kept
// reduced in [0, p) exactly as in field.metal.

// Generator G in little-endian u32 limbs (limb[0] = least-significant word).
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
constant uint GX[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};
constant uint GY[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

struct jpoint { fe X, Y, Z; };

inline bool fe_is_zero(thread const fe& a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0u) return false;
    }
    return true;
}

inline bool fe_eq(thread const fe& a, thread const fe& b) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != b.v[i]) return false;
    }
    return true;
}

inline fe fe_from_u32(uint x) {
    fe r;
    r.v[0] = x;
    for (int i = 1; i < 8; i++) r.v[i] = 0u;
    return r;
}

inline jpoint j_infinity() {
    jpoint r;
    r.X = fe_from_u32(1u); // X,Y arbitrary-nonzero; Z==0 is the infinity marker
    r.Y = fe_from_u32(1u);
    r.Z = fe_from_u32(0u);
    return r;
}

// Jacobian point doubling for a = 0 (EFD dbl-2009-l).
__attribute__((noinline)) jpoint j_double(jpoint P) {
    if (fe_is_zero(P.Z)) {
        return j_infinity();
    }
    fe A = fe_mul(P.X, P.X);              // X1^2
    fe B = fe_mul(P.Y, P.Y);              // Y1^2
    fe C = fe_mul(B, B);                  // B^2
    fe XB = fe_add(P.X, B);
    fe D = fe_sub(fe_sub(fe_mul(XB, XB), A), C); // (X1+B)^2 - A - C
    D = fe_add(D, D);                     // 2*(...)
    fe E = fe_add(fe_add(A, A), A);       // 3*A
    fe F = fe_mul(E, E);
    fe X3 = fe_sub(F, fe_add(D, D));      // F - 2*D
    fe C8 = fe_add(C, C);
    C8 = fe_add(C8, C8);
    C8 = fe_add(C8, C8);                  // 8*C
    fe Y3 = fe_sub(fe_mul(E, fe_sub(D, X3)), C8); // E*(D - X3) - 8*C
    fe Z3 = fe_mul(fe_add(P.Y, P.Y), P.Z);        // 2*Y1*Z1
    jpoint R;
    R.X = X3; R.Y = Y3; R.Z = Z3;
    return R;
}

// Jacobian point addition (EFD add-2007-bl), fully guarded for the infinity,
// equal-point (=> double), and inverse-point (=> infinity) cases.
__attribute__((noinline)) jpoint j_add(jpoint P, jpoint Q) {
    if (fe_is_zero(P.Z)) return Q;
    if (fe_is_zero(Q.Z)) return P;

    fe Z1Z1 = fe_mul(P.Z, P.Z);
    fe Z2Z2 = fe_mul(Q.Z, Q.Z);
    fe U1 = fe_mul(P.X, Z2Z2);            // X1*Z2^2
    fe U2 = fe_mul(Q.X, Z1Z1);            // X2*Z1^2
    fe S1 = fe_mul(fe_mul(P.Y, Q.Z), Z2Z2); // Y1*Z2^3
    fe S2 = fe_mul(fe_mul(Q.Y, P.Z), Z1Z1); // Y2*Z1^3

    if (fe_eq(U1, U2)) {
        if (fe_eq(S1, S2)) {
            return j_double(P);          // P == Q
        }
        return j_infinity();             // P == -Q
    }

    fe H = fe_sub(U2, U1);
    fe I = fe_add(H, H);
    I = fe_mul(I, I);                     // (2*H)^2
    fe J = fe_mul(H, I);
    fe r = fe_sub(S2, S1);
    r = fe_add(r, r);                     // 2*(S2 - S1)
    fe V = fe_mul(U1, I);
    fe X3 = fe_sub(fe_sub(fe_mul(r, r), J), fe_add(V, V)); // r^2 - J - 2*V
    fe S1J = fe_mul(S1, J);
    fe Y3 = fe_sub(fe_mul(r, fe_sub(V, X3)), fe_add(S1J, S1J)); // r*(V-X3) - 2*S1*J
    fe Zsum = fe_add(P.Z, Q.Z);
    Zsum = fe_mul(Zsum, Zsum);
    Zsum = fe_sub(fe_sub(Zsum, Z1Z1), Z2Z2);
    fe Z3 = fe_mul(Zsum, H);             // ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    jpoint R;
    R.X = X3; R.Y = Y3; R.Z = Z3;
    return R;
}

// R = k*G via MSB-first double-and-add, then Jacobian -> affine.
__attribute__((noinline)) void scalarmul(fe k, thread fe& outx, thread fe& outy) {
    jpoint R = j_infinity();
    jpoint G;
    for (int i = 0; i < 8; i++) {
        G.X.v[i] = GX[i];
        G.Y.v[i] = GY[i];
    }
    G.Z = fe_from_u32(1u);

#pragma clang loop unroll(disable)
    for (int bit = 255; bit >= 0; bit--) {
        R = j_double(R);
        uint limb = k.v[bit >> 5];
        uint b = (limb >> (bit & 31)) & 1u;
        if (b) {
            R = j_add(R, G);
        }
    }

    fe zinv = fe_inv(R.Z);
    fe zinv2 = fe_mul(zinv, zinv);
    fe zinv3 = fe_mul(zinv2, zinv);
    outx = fe_mul(R.X, zinv2);
    outy = fe_mul(R.Y, zinv3);
}

// One thread per key: read 8 LE limbs at keys[gid*8], write affine x then y
// (8 LE limbs each) at outxy[gid*16].
kernel void ec_test(device const uint* keys  [[buffer(0)]],
                    device uint* outxy        [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    fe k;
    for (int i = 0; i < 8; i++) {
        k.v[i] = keys[gid * 8 + i];
    }
    fe x, y;
    scalarmul(k, x, y);
    for (int i = 0; i < 8; i++) {
        outxy[gid * 16 + i] = x.v[i];
        outxy[gid * 16 + 8 + i] = y.v[i];
    }
}
