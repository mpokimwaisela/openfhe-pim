#pragma once
#include "../utils/common.h"
#include <alloc.h>
#include <defs.h>
#include <inttypes.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define CHUNK_BYTES 1 << 9 // 256 bytes per chunk
#define CHUNK_ELEMS (CHUNK_BYTES / sizeof(dpu_word_t)) // 32 elements per chunk

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

typedef struct {
  dpu_word_t mod;
} ctx_binop_t;

typedef struct {
  dpu_word_t mod;
  dpu_word_t scalar;
} ctx_scalar_t;

typedef void (*compute_fn_t)(dpu_word_t *, const dpu_word_t *, const dpu_word_t *,
                             uint32_t, void *);

static inline void
process_mram_blocks(compute_fn_t compute, uint32_t offset_A,
                    uint32_t offset_B, // pass NULL for scalar kernels
                    uint32_t offset_C, uint32_t total_elems, void *ctx) {

  const uint32_t tid = me();

  dpu_word_t *buf_A = (dpu_word_t *)mem_alloc(CHUNK_BYTES);
  dpu_word_t *buf_B = (offset_B != 0) ? (dpu_word_t *)mem_alloc(CHUNK_BYTES) : NULL;
  dpu_word_t *buf_C = (dpu_word_t *)mem_alloc(CHUNK_BYTES);

  __mram_ptr dpu_word_t *A =
      (__mram_ptr dpu_word_t *)(offset_A + DPU_MRAM_HEAP_POINTER);
  __mram_ptr dpu_word_t *B =
      (__mram_ptr dpu_word_t *)(offset_B + DPU_MRAM_HEAP_POINTER);
  __mram_ptr dpu_word_t *C =
      (__mram_ptr dpu_word_t *)(offset_C + DPU_MRAM_HEAP_POINTER);

  uint32_t elems_per_tasklet = (total_elems + NR_TASKLETS - 1) / NR_TASKLETS;
  uint32_t start_elem = tid * elems_per_tasklet;
  uint32_t end_elem = (start_elem + elems_per_tasklet > total_elems) 
                        ? total_elems : start_elem + elems_per_tasklet;
  
  for (uint32_t i = start_elem; i < end_elem; i += CHUNK_ELEMS) {
    uint32_t chunk = (i + CHUNK_ELEMS > end_elem) ? (end_elem - i) : CHUNK_ELEMS;

    mram_read(&A[i], buf_A, chunk * sizeof(dpu_word_t));

    if (offset_B != 0)
      mram_read(&B[i], buf_B, chunk * sizeof(dpu_word_t));

    compute(buf_C, buf_A, (offset_B != 0) ? buf_B : 0, chunk, ctx);

    mram_write(buf_C, &C[i], chunk * sizeof(dpu_word_t));
  }
}

static inline dpu_word_t mram_read_u64(__mram_ptr const dpu_word_t *src) {
  dpu_word_t tmp;
  /* NB: sizeof(dpu_word_t) == 8, MRAM address is already word-aligned */
  mram_read((__mram_ptr void const *)src, &tmp, sizeof(dpu_word_t));
  return tmp;
}