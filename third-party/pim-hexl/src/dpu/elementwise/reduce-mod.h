#include "../memory.h"
#include <stdint.h>

typedef struct
{
  dpu_word_t mod;
  dpu_word_t in_factor;
  dpu_word_t out_factor;
} ctx_reduce_t;

static inline dpu_word_t reduce_to_bound(dpu_word_t x,
                                       dpu_word_t m,
                                       dpu_word_t bound)
{
  while (x >= bound)
    x -= m;
  return x;
}

static void reduce_mod_compute(dpu_word_t *out,
                               const dpu_word_t *in,
                               const dpu_word_t *_ /*unused*/,
                               uint32_t n,
                               void *ctx_)
{
  const ctx_reduce_t *ctx = (const ctx_reduce_t *)ctx_;
  const dpu_word_t m = ctx->mod;
  const dpu_word_t f_in = ctx->in_factor;
  const dpu_word_t f_out = ctx->out_factor;

  const dpu_word_t bound_1m = m;
  const dpu_word_t bound_2m = m << 1; /* 2·m */

  for (uint32_t i = 0; i < n; ++i)
  {
    dpu_word_t x = in[i];

    if (f_in == 2)
    {
      if (f_out == 1 && x >= m)
        x -= m;
    }

    else if (f_in == 4)
    {
      if (f_out == 1) /* want [0, m)  */
        x = reduce_to_bound(x, m, bound_1m);
      else
      { /* want [0, 2m) */
        if (x >= bound_2m)
          x -= bound_2m; /* x mod 2m */
      }
    }

    else
    {
      dpu_word_t target = (f_out == 2) ? bound_2m : bound_1m;
      while (x >= target)
        x -= m; /* worst-case m times */
    }

    out[i] = x;
  }
}

int reduce_mod(void)
{
  ctx_reduce_t ctx = {
      .mod = DPU_INPUT_ARGUMENTS.mod,
      .in_factor = DPU_INPUT_ARGUMENTS.input_mod_factor,
      .out_factor = DPU_INPUT_ARGUMENTS.output_mod_factor};

  process_mram_blocks(reduce_mod_compute,
                      /* A */ DPU_INPUT_ARGUMENTS.A.offset,
                      /* B */ 0,
                      /* C */ DPU_INPUT_ARGUMENTS.C.offset,
                      DPU_INPUT_ARGUMENTS.A.size,
                      &ctx);
  return 0;
}
