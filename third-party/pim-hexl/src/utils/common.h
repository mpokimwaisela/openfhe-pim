#pragma once
#include <stdint.h>
#include "number-theory.h"

#define NR_KERNELS 10
// #define DEBUG 0

typedef enum {
  MOD_ADD,
  MOD_ADD_SCALAR,
  CMP_ADD,
  CMP_SUB_MOD,
  FMA_MOD,
  MOD_SUB,
  MOD_SUB_SCALAR,
  MOD_MUL,
  MOD_REDUCE,
  NTT_STAGE,
} pimop_t;

typedef struct DPU_ARRAY {
  uint32_t offset;
  uint32_t size; // in elements
  uint32_t size_in_bytes;
} dpu_array_t;

typedef struct dpu_arguments_t {
  dpu_array_t A;
  dpu_array_t B;
  dpu_array_t C;
  pimop_t kernel;
  uint64_t mod;
  uint64_t scalar;
  cmp_t cmp;
  uint64_t bound;
  uint32_t mod_factor;
  uint32_t input_mod_factor;
  uint32_t output_mod_factor;
} dpu_arguments_t;
