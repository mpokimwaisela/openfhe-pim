#include "../memory.h"
#include <stdint.h>

typedef struct
{
  uint64_t mod;
  uint64_t in_factor;
  uint64_t out_factor;
} ctx_reduce_t;

static inline uint64_t reduce_to_bound(uint64_t x,
                                       uint64_t m,
                                       uint64_t bound)
{
  while (x >= bound)
    x -= m;
  return x;
}

static void reduce_mod_compute(uint64_t *out,
                               const uint64_t *in,
                               const uint64_t *_ /*unused*/,
                               uint32_t n,
                               void *ctx_)
{
  const ctx_reduce_t *ctx = (const ctx_reduce_t *)ctx_;
  const uint64_t m = ctx->mod;
  const uint64_t f_in = ctx->in_factor;
  const uint64_t f_out = ctx->out_factor;

  const uint64_t bound_1m = m;
  const uint64_t bound_2m = m << 1; /* 2·m */

  for (uint32_t i = 0; i < n; ++i)
  {
    uint64_t x = in[i];

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
      uint64_t target = (f_out == 2) ? bound_2m : bound_1m;
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
