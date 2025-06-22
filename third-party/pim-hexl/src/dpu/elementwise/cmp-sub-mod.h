#pragma once
#include "../memory.h"
#include <stdint.h>

typedef struct {
  uint64_t mod;
  uint64_t diff;
  uint64_t bound;
  uint8_t cmp_code;
} ctx_cmpsub_t;

static void cmp_sub_mod_compute(uint64_t *out, const uint64_t *in,
                                const uint64_t *_ /* unused */, uint32_t n,
                                void *ctx_) {
  ctx_cmpsub_t *ctx = (ctx_cmpsub_t *)ctx_;
  const uint64_t mod = ctx->mod;
  const uint64_t diff = ctx->diff;
  const uint64_t bound = ctx->bound;
  const uint8_t code = ctx->cmp_code;

  for (uint32_t i = 0; i < n; ++i) {
    uint64_t v = in[i];

    /* optional sanity-cut: ensure < mod (mimics Intel’s v %= mod) */
    if (v >= mod)
      v -= mod;

    if (cmp_ok(v, bound, code)) {
      v = sub_mod_u64(v, diff, mod);
    }
    out[i] = v;
  }
}

static inline int cmp_sub_mod() {
  ctx_cmpsub_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                      .diff = DPU_INPUT_ARGUMENTS.scalar,
                      .bound = DPU_INPUT_ARGUMENTS.bound,
                      .cmp_code = DPU_INPUT_ARGUMENTS.cmp};

  process_mram_blocks(cmp_sub_mod_compute,
                      DPU_INPUT_ARGUMENTS.A.offset, 0,
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}
