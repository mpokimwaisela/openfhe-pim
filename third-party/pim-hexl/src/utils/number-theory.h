#ifndef PIM_NUMBER_THEORY_H
#define PIM_NUMBER_THEORY_H

#include <stdint.h>
#include <stdio.h>
#include "c128.h"

typedef uint64_t dpu_word_t;


#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  CMP_EQ,
  CMP_NE,
  CMP_LT,
  CMP_LE,
  CMP_NLT, /* >= */
  CMP_NLE, /* >  */
  CMP_TRUE,
  CMP_FALSE
} cmp_t;


static inline uint32_t ilog2(uint32_t n) { return 31u - __builtin_clz(n); }


static inline dpu_word_t add_mod_u64(dpu_word_t x, dpu_word_t y, dpu_word_t m) {
  dpu_word_t s = x + y;
  return (s >= m) ? s - m : s;
}

static inline dpu_word_t sub_mod_u64(dpu_word_t x, dpu_word_t y, dpu_word_t m) {
  return (x >= y) ? (x - y) : (x + m - y);
}

static inline uint64_t mul_mod_u64(uint64_t a,
                                   uint64_t b,
                                   uint64_t m)
{
    uint64_t result = 0;
    // uint64_t b_orig = b;
    // uint64_t a_orig = a;
    
    while (a) {
        if (a & 1) {
            // safe modular add without ever overflow-wrapping
            if (result >= m - b)
                result = result - (m - b);
            else
                result = result + b;
        }
        a >>= 1;

        // safe modular double without overflow
        if (b >= m - b)
            b = b - (m - b);
        else
            b = b + b;
    }

    // if (result ==0) {
    //   printf("mul_mod_u64: result is zero, a=%lu, b=%lu, m=%lu\n", a_orig, b_orig, m);
    // }
    return result;
}



// // Compute bit-length of x using built-in CLZ
// static uint32_t bit_length(uint64_t x) {
//     return 64 - __builtin_clzll(x);
// }

// #if defined(__SIZEOF_INT128__)
// // Precompute mu = floor(2^(2*k) / m) using built-in __uint128_t
// static uint64_t compute_mu(uint64_t m) {
//     uint32_t k = bit_length(m);
//     __uint128_t numerator = 1;
//     numerator <<= k;
//     numerator <<= k;  // now 2^(2*k)
//     return (uint64_t)(numerator / m);
// }
// #endif

// // Barrett modular multiplication: returns (a * b) mod m
// static inline uint64_t mul_mod_u64_barrett(uint64_t a, uint64_t b, uint64_t m, uint64_t mu) {
//     u128 product = u128_mul64(a, b);

//     uint32_t k = bit_length(m);

//     // q1 = product >> (k-1)
//     u128 q1_128 = u128_shr(product, k - 1);
//     uint64_t q1 = q1_128.lo;

//     // q2 = q1 * mu, q3 = q2 >> (k+1)
//     u128 q2 = u128_mul64(q1, mu);
//     u128 q3_128 = u128_shr(q2, k + 1);
//     uint64_t q3 = q3_128.lo;

//     // mask = 2^(k+1)-1
//     u128 mask = u128_from_u64(1);
//     mask = u128_shl(mask, k + 1);
//     mask = u128_sub(mask, u128_from_u64(1));

//     // r1 = product & mask, r2 = (q3*m) & mask
//     u128 r1 = u128_mask(product, k + 1);
//     u128 r2_full = u128_mul64(q3, m);
//     u128 r2 = u128_mask(r2_full, k + 1);

//     // r = r1 - r2 (+ mask+1 if negative)
//     u128 r128;
//     if (u128_ge(r1, r2)) {
//         r128 = u128_sub(r1, r2);
//     } else {
//         u128 tmp = u128_add(r1, u128_add(mask, u128_from_u64(1)));
//         r128 = u128_sub(tmp, r2);
//     }

//     uint64_t r = r128.lo;
//     while (r >= m) r -= m;
//     return r;
// }


static inline dpu_word_t mul_mod_u64_g(dpu_word_t a, dpu_word_t b, dpu_word_t mod, dpu_word_t mu) {
  if (mu == 0) 
    return mul_mod_u64(a, b, mod);

  return (a * b) % mod;  
  // return mul_mod_u64_barrett(a, b, mod, mu);
}

static inline dpu_word_t inverse_mod_u64(dpu_word_t a, dpu_word_t m) {
  dpu_word_t b = m, u = 1, v = 0;
  while (b) {
    dpu_word_t t = a / b;
    dpu_word_t tmp = a - t * b; a = b; b = tmp;
    tmp = u - t * v; u = v; v = tmp;
  }
  if (a != 1) return 0; /* not invertible */
  return (u + m) % m;
}

static inline dpu_word_t pow_mod_u64(dpu_word_t base, dpu_word_t exp, dpu_word_t mod) {
#if defined(__SIZEOF_INT128__)
  __uint128_t acc = 1, b = base % mod;
  while (exp) {
    if (exp & 1) acc = (acc * b) % mod;
    b = (b * b) % mod; exp >>= 1;
  }
  return (dpu_word_t)acc;
#else
  dpu_word_t acc = 1; 
  base %= mod;
  while (exp) {
    if (exp & 1) 
      acc = mul_mod_u64(acc, base, mod);
    base = mul_mod_u64(base, base, mod); 
    exp >>= 1;
  }
  return acc;
#endif
}

static inline int cmp_ok(dpu_word_t v, dpu_word_t bound, cmp_t code) {
  switch (code) {
    case CMP_EQ:   return v == bound;
    case CMP_NE:   return v != bound;
    case CMP_LT:   return v <  bound;
    case CMP_LE:   return v <= bound;
    case CMP_NLT:  return v >= bound;
    case CMP_NLE:  return v >  bound;
    case CMP_TRUE: return 1;
    default:       return 0;
  }
}

static inline void butterfly_u64(dpu_word_t *x, dpu_word_t *y, dpu_word_t w,
                                 dpu_word_t q, dpu_word_t twoq) {
  dpu_word_t t = mul_mod_u64(*y, w, q);
  dpu_word_t u = *x;

  dpu_word_t a = u + t; if (a >= twoq) a -= twoq;
  dpu_word_t b = (u >= t) ? u - t : u + twoq - t;

  *x = (a >= q) ? a - q : a;
  *y = (b >= q) ? b - q : b;
}


static inline uint32_t bitrev(uint32_t x, uint32_t logn) {
  uint32_t r = 0;
  for (uint32_t i = 0; i < logn; ++i) {
    r = (r << 1) | (x & 1); x >>= 1;
  }
  return r;
}

static inline dpu_word_t find_root(uint32_t N, dpu_word_t p) {
  dpu_word_t step = (p - 1) / N;
  for (dpu_word_t g = 2; g < p; ++g) {
    dpu_word_t w = pow_mod_u64(g, step, p);
    if (pow_mod_u64(w, N, p) == 1 && pow_mod_u64(w, N / 2, p) != 1)
      return w;
  }
  return 0; 
}

#ifdef __cplusplus
}
#endif

#endif 
