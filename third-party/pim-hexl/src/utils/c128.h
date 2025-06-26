#ifndef HEXL_UTILS_C128_H
#define HEXL_UTILS_C128_H

#include <stdint.h>

typedef struct {
    uint64_t lo;
    uint64_t hi;
} u128;

static inline u128 u128_from_u64(uint64_t v) {
    u128 r = {v, 0};
    return r;
}

static inline u128 u128_add(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo);
    return r;
}

// Subtract two u128 (assume a >= b)
static inline u128 u128_sub(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo);
    return r;
}

// Compare u128: return 1 if a >= b
static inline int u128_ge(u128 a, u128 b) {
    if (a.hi != b.hi)
        return a.hi > b.hi;
    return a.lo >= b.lo;
}

// Shift right by n (<128)
static inline u128 u128_shr(u128 a, unsigned n) {
    if (n >= 64) {
        a.lo = a.hi >> (n - 64);
        a.hi = 0;
    } else if (n > 0) {
        a.lo = (a.lo >> n) | (a.hi << (64 - n));
        a.hi >>= n;
    }
    return a;
}

// Shift left by n (<128)
static inline u128 u128_shl(u128 a, unsigned n) {
    if (n >= 64) {
        a.hi = a.lo << (n - 64);
        a.lo = 0;
    } else if (n > 0) {
        a.hi = (a.hi << n) | (a.lo >> (64 - n));
        a.lo <<= n;
    }
    return a;
}

// Mask lower k bits
static inline u128 u128_mask(u128 a, unsigned k) {
    if (k >= 128) return a;
    if (k == 64) {
        a.hi = 0;
    } else if (k < 64) {
        uint64_t m = (~0ULL) >> (64 - k);
        a.lo &= m;
        a.hi = 0;
    } else { // k > 64
        uint64_t m = (~0ULL) >> (128 - k);
        a.hi &= m;
    }
    return a;
}

// Multiply two 64-bit to 128-bit
static inline u128 u128_mul64(uint64_t a, uint64_t b) {
    const uint64_t MASK32 = 0xFFFFFFFFULL;
    uint64_t a_lo = a & MASK32;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & MASK32;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    // Combine partials carefully to avoid undefined behaviour
    uint64_t carry = (p0 >> 32) + (p1 & MASK32) + (p2 & MASK32);
    uint64_t lo = (p0 & MASK32) | ((carry & MASK32) << 32);

    uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (carry >> 32);

    u128 r = {lo, hi};
    return r;
}

#endif // HEXL_UTILS_C128_H