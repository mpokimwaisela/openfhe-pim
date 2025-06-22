#pragma once
#include "../memory.h"
#include <stdint.h>

static void sub_mod_compute(uint64_t *out, const uint64_t *a, const uint64_t *b,
                            uint32_t n, void *ctx_) {
  uint64_t mod = ((ctx_binop_t *)ctx_)->mod;
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = sub_mod_u64(a[i], b[i], mod);
  }
}

static void sub_mod_scalar_compute(uint64_t *out, const uint64_t *a,
                                   const uint64_t *_, uint32_t n, void *ctx_) {
  ctx_scalar_t *ctx = (ctx_scalar_t *)ctx_;
  uint64_t mod = ctx->mod;
  uint64_t scalar = ctx->scalar;

  for (uint32_t i = 0; i < n; ++i) {
    out[i] = sub_mod_u64(a[i], scalar, mod);
  }
}

static inline int mod_sub() {
  ctx_binop_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod};
  process_mram_blocks(sub_mod_compute,
                      DPU_INPUT_ARGUMENTS.A.offset,
                      DPU_INPUT_ARGUMENTS.B.offset,
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}

static inline int mod_sub_scalar() {
  ctx_scalar_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                      .scalar = DPU_INPUT_ARGUMENTS.scalar};
  process_mram_blocks(sub_mod_scalar_compute,
                      DPU_INPUT_ARGUMENTS.A.offset,
                      0, /* no B array */
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}
