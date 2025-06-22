#pragma once
#include "../memory.h"
#include <stdint.h>

static void add_mod_compute(dpu_word_t *out, const dpu_word_t *a, const dpu_word_t *b,
                            uint32_t n, void *ctx_) {
  dpu_word_t m = ((ctx_binop_t *)ctx_)->mod;
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = add_mod_u64(a[i], b[i], m);
  }
}

static void add_mod_scalar_compute(dpu_word_t *out, const dpu_word_t *a,
                                   const dpu_word_t *_, uint32_t n, void *ctx_) {
  ctx_scalar_t *ctx = (ctx_scalar_t *)ctx_;
  dpu_word_t m = ctx->mod, s = ctx->scalar;
  for (uint32_t i = 0; i < n; ++i)
    out[i] = add_mod_u64(a[i], s, m);
}

static inline int mod_add() {
  ctx_binop_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod};
  process_mram_blocks(add_mod_compute, DPU_INPUT_ARGUMENTS.A.offset,
                      DPU_INPUT_ARGUMENTS.B.offset,
                      DPU_INPUT_ARGUMENTS.C.offset, DPU_INPUT_ARGUMENTS.A.size,
                      &ctx);
  return 0;
}

static inline int mod_add_scalar() {
  ctx_scalar_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                      .scalar = DPU_INPUT_ARGUMENTS.scalar};
  process_mram_blocks(
      add_mod_scalar_compute, DPU_INPUT_ARGUMENTS.A.offset, 0, /* no B array */
      DPU_INPUT_ARGUMENTS.C.offset, DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}