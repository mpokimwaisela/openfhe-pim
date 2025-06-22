#ifndef PIM_NUMBER_THEORY_H
#define PIM_NUMBER_THEORY_H

#include <stdint.h>

typedef enum {
  CMP_EQ,
  CMP_NE,
  CMP_LT,
  CMP_LE,
  CMP_NLT, // >=
  CMP_NLE, // >
  CMP_TRUE,
  CMP_FALSE
} cmp_t;

static inline uint32_t ilog2(uint32_t n) { return 31u - __builtin_clz(n); }

static inline uint64_t add_mod_u64(uint64_t x, uint64_t y, uint64_t m) {
  uint64_t s = x + y;
  return (s >= m) ? s - m : s;
}

static inline uint64_t sub_mod_u64(uint64_t x, uint64_t y, uint64_t m) {
  return (x >= y) ? (x - y) : (x + m - y);
}

static inline uint64_t inverse_mod_u64(uint64_t a, uint64_t m) {
  uint64_t b = m, u = 1, v = 0;
  while (b) {
    uint64_t t = a / b;
    uint64_t tmp = a - t * b;
    a = b;
    b = tmp;
    tmp = u - t * v;
    u = v;
    v = tmp;
  }
  if (a != 1) /* not invertible */
    return 0;
  return (u + m) % m;
}

#if defined(__SIZEOF_INT128__)
static inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t m) {
  return (uint64_t)(((__uint128_t)a * b) % m);
}
#else /*  no native 128-bit → shift-add  */
static inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t m) {
  uint64_t res = 0;
  while (b) {
    if (b & 1) {
      res = (res >= m - a) ? res + a - m : res + a;
    }
    a = (a >= m - a) ? a + a - m : a + a;
    b >>= 1;
  }
  return res;
}
#endif

static uint64_t pow_mod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
#ifdef __SIZEOF_INT128__
  __uint128_t acc = 1, b = base % mod;
  while (exp) {
    if (exp & 1)
      acc = (acc * b) % mod;
    b = (b * b) % mod;
    exp >>= 1;
  }
  return static_cast<uint64_t>(acc);
#else
  uint64_t acc = 1;
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

static inline int cmp_ok(uint64_t v, uint64_t bound, uint8_t code) {
  switch (code) {
  case CMP_EQ:
    return v == bound;
  case CMP_NE:
    return v != bound;
  case CMP_LT:
    return v < bound;
  case CMP_LE:
    return v <= bound;
  case CMP_NLT:
    return v >= bound;
  case CMP_NLE:
    return v > bound;
  case CMP_TRUE:
    return 1;
  default /*CMP_FALSE*/:
    return 0;
  }
}

/* --------------------------------------------------------------------- */
/*  Radix-2 butterfly                                                    */
/* --------------------------------------------------------------------- */
static inline void butterfly_u64(uint64_t *x, uint64_t *y, uint64_t w,
                                 uint64_t q, uint64_t twoq) {
  uint64_t t = mul_mod_u64(*y, w, q); /* t = w*y (mod q)  */
  uint64_t u = *x;

  uint64_t a = u + t;
  if (a >= twoq)
    a -= twoq;                                  /* keep in [0,2q) */
  uint64_t b = (u >= t) ? u - t : u + twoq - t; /* same range     */

  /* optional final reduction to [0,q) left to the caller         */
  *x = (a >= q) ? a - q : a;
  *y = (b >= q) ? b - q : b;
}

static uint32_t bitrev(uint32_t x, uint32_t logn) {
  uint32_t r = 0;
  for (uint32_t i = 0; i < logn; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

static uint64_t find_root(uint32_t N, uint64_t p) {
  uint64_t step = (p - 1) / N;
  for (uint64_t g = 2; g < p; ++g) {
    uint64_t w = pow_mod_u64(g, step, p);
    if (pow_mod_u64(w, N, p) == 1 && pow_mod_u64(w, N / 2, p) != 1)
      return w;
  }
  return 0; // no root found
}

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

#endif /* PIM_NUMBER_THEORY_H */
