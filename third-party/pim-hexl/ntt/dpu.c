#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include <alloc.h>
#include "common.h"

__host struct NttDpuArgs NTT_ARGS;

#define BLOCK_SIZE 256
#define BLOCK_LEN (BLOCK_SIZE / sizeof(u64))

BARRIER_INIT(sync_barrier, NR_TASKLETS);

int main() {
  const u32 tasklet_id = me();

  if (tasklet_id == 0) {
    mem_reset(); // reset WRAM allocator once per kernel launch
  }
  barrier_wait(&sync_barrier);

  const u32 len = NTT_ARGS.len;
  const u32 blocks = NTT_ARGS.blocks;
  if (len == 0 || blocks == 0) return 0;

  const u64 modulus = NTT_ARGS.modulus;
  const u64 step = NTT_ARGS.root_step % modulus;
  const u32 span = 2 * len;
  __mram_ptr u64 *base = (__mram_ptr u64 *)DPU_MRAM_HEAP_POINTER;

  /* Work on 256-byte tiles for improved MRAM throughput without exhausting WRAM */
  u64 *cache_even = (u64 *)mem_alloc(BLOCK_SIZE);
  u64 *cache_odd = (u64 *)mem_alloc(BLOCK_SIZE);

  const u32 tiles = (len + BLOCK_LEN - 1) / BLOCK_LEN;

  for (u32 block = 0; block < blocks; ++block) {
    __mram_ptr u64 *block_base = base + (u64)block * span;

    for (u32 tile = tasklet_id; tile < tiles; tile += NR_TASKLETS) {
      u32 offset = tile * BLOCK_LEN;
      if (offset >= len)
        continue;

      u32 remaining = len - offset;
      u32 chunk_len = remaining < BLOCK_LEN ? remaining : BLOCK_LEN;
      u32 chunk_bytes = chunk_len * sizeof(u64);

      mram_read((__mram_ptr void const *)(block_base + offset), cache_even,
                chunk_bytes);
      mram_read((__mram_ptr void const *)(block_base + len + offset), cache_odd,
                chunk_bytes);

      u64 w = modpow(step, offset, modulus);

      for (u32 j = 0; j < chunk_len; ++j) {
        u64 u = ((u64)cache_even[j]);
        u64 v = ((u64)cache_odd[j]);
        u64 t = modmul(v, w, modulus);
        u64 sum = modadd(u, t, modulus);
        u64 diff = modsub(u, t, modulus);
        cache_even[j] = (u64)sum;
        cache_odd[j] = (u64)diff;
        w = modmul(w, step, modulus);
      }

      mram_write(cache_even, (__mram_ptr void *)(block_base + offset),
                 chunk_bytes);
      mram_write(cache_odd, (__mram_ptr void *)(block_base + len + offset),
                 chunk_bytes);
    }
  }
  // printf("DPU computation complete.\n");
  return 0;
}
