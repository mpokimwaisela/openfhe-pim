#pragma once
#include "../memory.h"
#include <stdint.h>

typedef struct {
  dpu_word_t bound;
  dpu_word_t diff;
  uint8_t cmp_code;
} ctx_cmp_t;

static void cmp_add_compute(dpu_word_t *out, const dpu_word_t *in,
                            const dpu_word_t *_, uint32_t n, void *ctx_) {
  ctx_cmp_t *ctx = (ctx_cmp_t *)ctx_;
  dpu_word_t bound = ctx->bound, diff = ctx->diff;
  uint8_t code = ctx->cmp_code;
  for (uint32_t i = 0; i < n; ++i)
    out[i] = cmp_ok(in[i], bound, code) ? (in[i] + diff) : in[i];
}

int cmp_add() {
  ctx_cmp_t ctx = {.bound = DPU_INPUT_ARGUMENTS.bound,
                   .diff = DPU_INPUT_ARGUMENTS.scalar,
                   .cmp_code = DPU_INPUT_ARGUMENTS.cmp};
  process_mram_blocks(cmp_add_compute,
                      DPU_INPUT_ARGUMENTS.A.offset, 0,
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}
