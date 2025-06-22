#include "elementwise/add-mod.h"
#include "elementwise/cmp-add.h"
#include "elementwise/cmp-sub-mod.h"
#include "elementwise/fma-mod.h"
#include "elementwise/mul-mod.h"
#include "elementwise/reduce-mod.h"
#include "elementwise/sub-mod.h"
#include "ntt/ntt-stage.h"

#include <barrier.h>

BARRIER_INIT(my_barrier, NR_TASKLETS);

int (*kernels[NR_KERNELS])(void) = {mod_add, mod_add_scalar, cmp_add,
                                    cmp_sub_mod, fma_mod, mod_sub,
                                    mod_sub_scalar, mod_mul, reduce_mod, ntt_stage};

int main()
{
  unsigned tid = me();
  if (tid == 0)
    mem_reset();

  barrier_wait(&my_barrier);
  return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}