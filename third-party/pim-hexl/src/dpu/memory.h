#pragma once
#include "../utils/common.h"
#include <alloc.h>
#include <defs.h>
#include <inttypes.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define CHUNK_BYTES 1 << 8 // 256 bytes per chunk
#define CHUNK_ELEMS (CHUNK_BYTES / sizeof(uint64_t)) // 32 elements per chunk

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

typedef struct {
  uint64_t mod;
} ctx_binop_t;

typedef struct {
  uint64_t mod;
  uint64_t scalar;
} ctx_scalar_t;

typedef void (*compute_fn_t)(uint64_t *, const uint64_t *, const uint64_t *,
                             uint32_t, void *);

static inline void
process_mram_blocks(compute_fn_t compute, uint32_t offset_A,
                    uint32_t offset_B, // pass NULL for scalar kernels
                    uint32_t offset_C, uint32_t total_elems, void *ctx) {

  const uint32_t tid = me();

  uint64_t *buf_A = (uint64_t *)mem_alloc(CHUNK_BYTES);
  uint64_t *buf_B = (offset_B != 0) ? (uint64_t *)mem_alloc(CHUNK_BYTES) : NULL;
  uint64_t *buf_C = (uint64_t *)mem_alloc(CHUNK_BYTES);
  
  // Check for allocation failures
  if (!buf_A || !buf_C || (offset_B != 0 && !buf_B)) {
    printf("Tasklet %u: Memory allocation failed!\n", tid);
    return;
  }

  __mram_ptr uint64_t *A =
      (__mram_ptr uint64_t *)(offset_A + DPU_MRAM_HEAP_POINTER);
  __mram_ptr uint64_t *B =
      (__mram_ptr uint64_t *)(offset_B + DPU_MRAM_HEAP_POINTER);
  __mram_ptr uint64_t *C =
      (__mram_ptr uint64_t *)(offset_C + DPU_MRAM_HEAP_POINTER);

  // Better work distribution: each tasklet processes elements round-robin
  // Calculate elements per tasklet with proper load balancing
  const uint32_t num_tasklets = NR_TASKLETS;
  uint32_t elems_per_tasklet = (total_elems + num_tasklets - 1) / num_tasklets;
  uint32_t start_elem = tid * elems_per_tasklet;
  uint32_t end_elem = (start_elem + elems_per_tasklet > total_elems) 
                        ? total_elems : start_elem + elems_per_tasklet;
  
  // Process in chunks, but ensure all tasklets get work
  for (uint32_t i = start_elem; i < end_elem; i += CHUNK_ELEMS) {
    uint32_t chunk = (i + CHUNK_ELEMS > end_elem) ? (end_elem - i) : CHUNK_ELEMS;

    mram_read(&A[i], buf_A, chunk * sizeof(uint64_t));

    if (offset_B != 0)
      mram_read(&B[i], buf_B, chunk * sizeof(uint64_t));

#if 0  // Disable DEBUG output to reduce noise
    printf("Total elements: %u, Chunk size: %u, Tasklet ID: %u\n", total_elems,
           chunk, tid);
    for (uint32_t j = 0; j < chunk; ++j) {
      printf("  A[%u] = %" PRIu64, i + j, (uint64_t)buf_A[j]);
      if (offset_B != 0)
        printf("  B[%u] = %" PRIu64, i + j, (uint64_t)buf_B[j]);
      printf("\n");
    }
#endif

    compute(buf_C, buf_A, (offset_B != 0) ? buf_B : 0, chunk, ctx);

#if 0  // Disable DEBUG output to reduce noise
    for (uint32_t j = 0; j < chunk; ++j)
      printf("  C[%u] = %" PRIu64 "\n", i + j, (uint64_t)buf_C[j]);
#endif
    mram_write(buf_C, &C[i], chunk * sizeof(uint64_t));
  }
}

static inline uint64_t mram_read_u64(__mram_ptr const uint64_t *src) {
  uint64_t tmp;
  /* NB: sizeof(uint64_t) == 8, MRAM address is already word-aligned */
  mram_read((__mram_ptr void const *)src, &tmp, sizeof(uint64_t));
  return tmp;
}
