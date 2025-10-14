#ifndef PIM_NUMBER_THEORY_H
#define PIM_NUMBER_THEORY_H

#include <stdint.h>
#include <stdio.h>
#include "c128.h"
#include "log.h"

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


static inline dpu_word_t add_mod(dpu_word_t x, dpu_word_t y, dpu_word_t m) {
  dpu_word_t s = x + y;
  return (s >= m) ? s - m : s;
}

static inline dpu_word_t sub_mod(dpu_word_t x, dpu_word_t y, dpu_word_t m) {
  return (x >= y) ? (x - y) : (x + m - y);
}

static inline dpu_word_t mul_mod_n(dpu_word_t a,
                                   dpu_word_t b,
                                   dpu_word_t m)
{
    dpu_word_t result = 0;
    
    while (a) {
        if (a & 1) {
            if (result >= m - b)
                result = result - (m - b);
            else
                result = result + b;
        }
        a >>= 1;

        if (b >= m - b)
            b = b - (m - b);
        else
            b = b + b;
    }

    // if (result ==0) {
    //   printf("mul_mod: result is zero, a=%lu, b=%lu, m=%lu\n", a_orig, b_orig, m);
    // }
    return result;
}

static inline dpu_word_t barrett_init(dpu_word_t m) {
    dpu_word_t mu = 0;
    
    if (m >= (1ULL << 60)) {
      LOG_WARN("Barrett reduction may not be accurate for modulus >= 2^60 %lu ", m);
    }
    
    // Compute floor(2^64 / m)
    u128 numerator = u128_shl(u128_from_u64(1), 64);  
    mu = u128_div64(numerator, m);
    
    return mu;
}

static inline dpu_word_t barrett_reduce(u128 x, dpu_word_t m, dpu_word_t mu) {

    u128 threshold = u128_shl(u128_from_u64(m), 64); // m * 2^64
    if (u128_ge(x, threshold)) {
        return u128_mod64(x, m);
    }
    
    // Barrett reduction: q ≈ (x * mu) >> 64
    u128 q_full = u128_mul_u64(x, mu);
    dpu_word_t q = u128_shr(q_full, 64).lo;  
    
    // Compute remainder: r = x - q * m
    u128 qm = u128_mul64(q, m);
    
    // FIX: Check for underflow before subtraction
    if (u128_ge(x, qm)) {
        u128 remainder_128 = u128_sub(x, qm);
        dpu_word_t r = u128_to_u64(remainder_128);
        
        // Multiple correction steps for better accuracy
        while (r >= m) {
            r -= m;
        }
        
        return r;
    } else {
        // Underflow case: qm > x, Barrett approximation is too large
        // Fall back to standard modulo
        return u128_mod64(x, m);
    }
}

static inline dpu_word_t barrett_mul(dpu_word_t a, dpu_word_t b, dpu_word_t m, dpu_word_t mu) {
    u128 product = u128_mul64(a, b);
    return barrett_reduce(product, m, mu);
}


static inline dpu_word_t mul_mod(dpu_word_t a, dpu_word_t b, dpu_word_t mod, dpu_word_t mu) {
  if (mu == 0) 
    return mul_mod_n(a, b, mod); 

  return barrett_mul(a, b, mod, mu);
}

static inline dpu_word_t inverse_mod(dpu_word_t a, dpu_word_t m) {
  dpu_word_t b = m, u = 1, v = 0;
  while (b) {
    dpu_word_t t = a / b;
    dpu_word_t tmp = a - t * b; a = b; b = tmp;
    tmp = u - t * v; u = v; v = tmp;
  }
  if (a != 1) return 0; /* not invertible */
  return (u + m) % m;
}

static inline dpu_word_t pow_mod(dpu_word_t base, dpu_word_t exp, dpu_word_t mod) {
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
      acc = mul_mod(acc, base, mod,0);
    base = mul_mod(base, base, mod,0); 
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

static inline void butterfly(dpu_word_t *x, dpu_word_t *y, dpu_word_t w,
                                 dpu_word_t q, dpu_word_t twoq) {
  dpu_word_t t = mul_mod(*y, w, q,0);
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
    dpu_word_t w = pow_mod(g, step, p);
    if (pow_mod(w, N, p) == 1 && pow_mod(w, N / 2, p) != 1)
      return w;
  }
  return 0; 
}

#ifdef __cplusplus
}
#endif

#endif 
