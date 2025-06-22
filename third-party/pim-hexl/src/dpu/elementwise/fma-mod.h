#pragma once
#include "../memory.h"
#include <stdint.h>

typedef struct
{
  uint64_t mod;       /* modulus  (m)                     */
  uint64_t scalar;    /* multiplier (k)                   */
  uint8_t mod_factor; /* 1 / 2 / 4 / 8  (range check only)*/
  uint8_t has_addend; /* 0 → arg3 absent, 1 → present     */
} ctx_fma_t;

/* Reduce x < 8·m into [0,m) with ≤3 subtractions */
static inline uint64_t reduce_8m(uint64_t x, uint64_t m)
{
  if (x >= m)
    x -= m;
  if (x >= m)
    x -= m;
  if (x >= m)
    x -= m; /* mod_factor ≤ 8 ⇒ max three steps */
  return x;
}

static void fma_mod_compute(uint64_t *out, const uint64_t *a, const uint64_t *b,
                            uint32_t n, void *ctx_)
{
  ctx_fma_t *ctx = (ctx_fma_t *)ctx_;
  const uint64_t m = ctx->mod;
  const uint64_t scalar = ctx->scalar;
  const uint8_t addFlg = ctx->has_addend;

  for (uint32_t i = 0; i < n; ++i)
  {

    /* 1) bring inputs into [0,m) (at most 3 subtractions) */
    uint64_t x = reduce_8m(a[i], m);
    uint64_t y = addFlg ? reduce_8m(b[i], m) : 0;

    /* 2) multiply-mod without 128-bit */
    uint64_t prod = mul_mod_u64(x, scalar, m);

    /* 3) add + final reduction (single conditional) */
    uint64_t sum = prod + y;
    if (sum >= m)
      sum -= m;

    out[i] = sum;
  }
}

static inline int fma_mod()
{
  printf("FMA_MOD kernel started\n");
  ctx_fma_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                   .scalar = DPU_INPUT_ARGUMENTS.scalar,
                   .mod_factor = DPU_INPUT_ARGUMENTS.mod_factor,
                   .has_addend = (DPU_INPUT_ARGUMENTS.B.size != 0)};

  uint32_t base_B =
      ctx.has_addend ? DPU_INPUT_ARGUMENTS.B.offset : 0;
  // printf("FMA_MOD: mod=%" PRIu64 ", scalar=%" PRIu64
  //        ", mod_factor=%u, has_addend=%u\n",
  //        ctx.mod, ctx.scalar, ctx.mod_factor, ctx.has_addend);

  process_mram_blocks(
      fma_mod_compute, DPU_INPUT_ARGUMENTS.A.offset, base_B,
      DPU_INPUT_ARGUMENTS.C.offset,
      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}
