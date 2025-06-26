#pragma once
#include "../memory.h"
#include <stdint.h>

typedef struct {
  dpu_word_t mod;
  dpu_word_t scalar; // scalar for scalar multiplication
  dpu_word_t mu; // precomputed mu for Barrett reduction
} ctx_mult_t;

static void mult_mod_compute(dpu_word_t *out, const dpu_word_t *a,
                             const dpu_word_t *b, uint32_t n, void *ctx_) {
  ctx_mult_t *ctx = (ctx_mult_t *)ctx_;
  const dpu_word_t m = ctx->mod;
  const dpu_word_t mu = ctx->mu;
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = mul_mod_u64_g(a[i], b[i], m, mu);
  }
}

static void mult_mod_scalar_compute(dpu_word_t *out, const dpu_word_t *a,
                                   const dpu_word_t *_, uint32_t n, void *ctx_) {
  ctx_mult_t *ctx = (ctx_mult_t *)ctx_;
  dpu_word_t m = ctx->mod, s = ctx->scalar, mu = ctx->mu;
  for (uint32_t i = 0; i < n; ++i)
    out[i] = mul_mod_u64_g(a[i], s, m, mu);
}

static inline int mod_mul() {
  ctx_mult_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                    .mu = DPU_INPUT_ARGUMENTS.mu};

  process_mram_blocks(mult_mod_compute,
                      DPU_INPUT_ARGUMENTS.A.offset,
                      DPU_INPUT_ARGUMENTS.B.offset,
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}

static inline int mod_mul_scalar() {
  ctx_mult_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                      .mu = DPU_INPUT_ARGUMENTS.mu,
                      .scalar = DPU_INPUT_ARGUMENTS.scalar};
  process_mram_blocks(
      mult_mod_scalar_compute, 
      DPU_INPUT_ARGUMENTS.A.offset,
       0, /* no B array */
      DPU_INPUT_ARGUMENTS.C.offset, 
      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}