#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t u32;
typedef uint64_t u64;
#if defined(__SIZEOF_INT128__)
typedef unsigned __int128 u128;
#endif

struct NttDpuArgs {
  u32 len;
  u32 blocks;
  u64 modulus;
  u64 root_step;
};

#if defined(__SIZEOF_INT128__)
static inline u64 modadd(u64 a, u64 b, u64 m) {
  u128 s = (u128)a + (u128)b;
  s %= (u128)m;
  return (u64)s;
}
#else
static inline u64 modadd(u64 a, u64 b, u64 m) {
  if (a >= m - b)
    return a - (m - b);
  return a + b;
}
#endif

static inline u64 modsub(u64 a, u64 b, u64 m) {
  return (a >= b) ? (a - b) : (a + m - b);
}

#if defined(__SIZEOF_INT128__)
static inline u64 modmul(u64 a, u64 b, u64 m) {
  return (u64)(((u128)a * (u128)b) % (u128)m);
}
#else
static inline u64 modmul(u64 a, u64 b, u64 m) {
  u64 result = 0;

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
  //   printf("mul_mod: result is zero, a=%lu, b=%lu, m=%lu\n", a_orig, b_orig,
  //   m);
  // }
  return result;
}
#endif

static inline u64 modpow(u64 a, unsigned long long e, u64 m) {
  a %= m;
  u64 r = (m == 1) ? 0 : 1 % m;
  while (e) {
    if (e & 1)
      r = modmul(r, a, m);
    a = modmul(a, a, m);
    e >>= 1;
  }
  return r;
}

static inline u64 modinv(u64 a, u64 p) { return modpow(a, p - 2, p); }

#ifdef __cplusplus
}
#endif
