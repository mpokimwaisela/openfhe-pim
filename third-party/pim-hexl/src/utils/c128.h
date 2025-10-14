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

// Additional operations for Barrett reduction

// Compare u128: return 1 if a == b
static inline int u128_eq(u128 a, u128 b) {
    return a.hi == b.hi && a.lo == b.lo;
}

// Compare u128: return 1 if a < b
static inline int u128_lt(u128 a, u128 b) {
    if (a.hi != b.hi)
        return a.hi < b.hi;
    return a.lo < b.lo;
}

// Convert u128 to uint64_t (truncate to lower 64 bits)
static inline uint64_t u128_to_u64(u128 a) {
    return a.lo;
}

// Check if u128 is zero
static inline int u128_is_zero(u128 a) {
    return a.hi == 0 && a.lo == 0;
}

// Modulo operation: u128 % uint64_t = uint64_t
static inline uint64_t u128_mod64(u128 dividend, uint64_t divisor) {
    if (divisor == 0) return 0; // Avoid division by zero
    
    // Use built-in 128-bit if available
    #ifdef __SIZEOF_INT128__
    __uint128_t val = ((__uint128_t)dividend.hi << 64) | dividend.lo;
    return (uint64_t)(val % divisor);
    #else
    // Purely portable implementation for systems without 128-bit support
    if (dividend.hi == 0) {
        // Simple case: just 64-bit division
        return dividend.lo % divisor;
    }
    
    // We need to compute (dividend.hi * 2^64 + dividend.lo) % divisor
    // Using the property: (a*b + c) % m = ((a%m) * (b%m) + c%m) % m
    
    // First reduce the high part
    uint64_t hi_mod = dividend.hi % divisor;
    uint64_t lo_mod = dividend.lo % divisor;
    
    // Now compute (hi_mod * 2^64) % divisor
    // We'll do this by repeated doubling: 2^64 = 2^1 * 2^1 * ... (64 times)
    uint64_t power2_mod = 1;
    for (int i = 0; i < 64; i++) {
        power2_mod = (power2_mod * 2) % divisor;
    }
    
    // Now compute (hi_mod * power2_mod + lo_mod) % divisor
    // Need to be careful about overflow in hi_mod * power2_mod
    uint64_t hi_contribution = 0;
    uint64_t temp_hi = hi_mod;
    uint64_t temp_power = power2_mod;
    
    // Multiply hi_mod * power2_mod using addition to avoid overflow
    while (temp_hi > 0) {
        if (temp_hi & 1) {
            hi_contribution = (hi_contribution + temp_power) % divisor;
        }
        temp_power = (temp_power * 2) % divisor;
        temp_hi >>= 1;
    }
    
    return (hi_contribution + lo_mod) % divisor;
    #endif
}

// Division: u128 / uint64_t = uint64_t
static inline uint64_t u128_div64(u128 dividend, uint64_t divisor) {
    if (divisor == 0) return 0; // Avoid division by zero
    
    #ifdef __SIZEOF_INT128__
    __uint128_t val = ((__uint128_t)dividend.hi << 64) | dividend.lo;
    __uint128_t result = val / divisor;
    if (result > UINT64_MAX) return UINT64_MAX; // Overflow
    return (uint64_t)result;
    #else
    // Simple implementation for systems without 128-bit
    // Check for overflow first
    if (dividend.hi >= divisor) {
        return UINT64_MAX; // Would overflow
    }
    
    // If high part is 0, it's just a 64-bit division
    if (dividend.hi == 0) {
        return dividend.lo / divisor;
    }
    
    // General case: use long division
    uint64_t quotient = 0;
    u128 remainder = {0, 0};
    
    for (int i = 127; i >= 0; i--) {
        remainder = u128_shl(remainder, 1);
        
        // Extract bit i from dividend
        if (i >= 64) {
            if (dividend.hi & (1ULL << (i - 64))) {
                remainder.lo |= 1;
            }
        } else {
            if (dividend.lo & (1ULL << i)) {
                remainder.lo |= 1;
            }
        }
        
        if (u128_ge(remainder, u128_from_u64(divisor))) {
            remainder = u128_sub(remainder, u128_from_u64(divisor));
            if (i < 64) {
                quotient |= (1ULL << i);
            }
        }
    }
    
    return quotient;
    #endif
}

// Multiply u128 * uint64_t = u128 (keep lower 128 bits)
static inline u128 u128_mul_u64(u128 a, uint64_t b) {
    #ifdef __SIZEOF_INT128__
    __uint128_t val = ((__uint128_t)a.hi << 64) | a.lo;
    __uint128_t result = val * b;
    u128 r = {(uint64_t)result, (uint64_t)(result >> 64)};
    return r;
    #else
    // (a.hi * 2^64 + a.lo) * b = a.hi * b * 2^64 + a.lo * b
    u128 lo_product = u128_mul64(a.lo, b);
    uint64_t hi_contrib = a.hi * b; // Only lower 64 bits matter for 128-bit result
    
    u128 result = {lo_product.lo, lo_product.hi + hi_contrib};
    return result;
    #endif
}

// Create u128 from high and low parts
static inline u128 u128_from_parts(uint64_t hi, uint64_t lo) {
    u128 r = {lo, hi};
    return r;
}

#endif // HEXL_UTILS_C128_H