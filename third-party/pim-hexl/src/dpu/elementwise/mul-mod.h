#pragma once
#include "../memory.h"
#include <stdint.h>

typedef struct {
  dpu_word_t mod;
  uint8_t factor;
} ctx_mult_t;

/* Reduce x < 4·m into [0,m) with ≤2 subtractions*/
static inline dpu_word_t reduce_4m(dpu_word_t x, dpu_word_t m) {
  if (x >= m)
    x -= m;
  if (x >= m)
    x -= m; /* factor ≤4 ⇒ max two steps */
  return x;
}

/*─────────────────  per-chunk compute callback ────────────────*/
static void mult_mod_compute(dpu_word_t *out, const dpu_word_t *a,
                             const dpu_word_t *b, uint32_t n, void *ctx_) {
  ctx_mult_t *ctx = (ctx_mult_t *)ctx_;
  const dpu_word_t m = ctx->mod;
  const uint8_t factor = ctx->factor; /* 1 / 2 / 4 */

  for (uint32_t i = 0; i < n; ++i) {
    dpu_word_t x = a[i];
    dpu_word_t y = b[i];

    /* bring each operand into [0,m) */
    if (factor == 2 || factor == 4) {
      x = reduce_4m(x, m);
      y = reduce_4m(y, m);
    } else { /* factor == 1 */
      if (x >= m)
        x -= m; /* just in case */
      if (y >= m)
        y -= m;
    }

    out[i] = mul_mod_u64(x, y, m);
  }
}

static inline int mod_mul() {
  ctx_mult_t ctx = {.mod = DPU_INPUT_ARGUMENTS.mod,
                    .factor = DPU_INPUT_ARGUMENTS.mod_factor};

  process_mram_blocks(mult_mod_compute,
                      DPU_INPUT_ARGUMENTS.A.offset,
                      DPU_INPUT_ARGUMENTS.B.offset,
                      DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size, &ctx);
  return 0;
}
