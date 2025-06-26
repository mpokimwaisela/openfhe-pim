#pragma once
#include <stdint.h>
#include <stdio.h>
#include "number-theory.h"

#define NR_KERNELS 11
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
  MOD_MUL_SCALAR,
  MOD_REDUCE,
  NTT_STAGE,
} pimop_t;

typedef struct dpu_array_t {
  uint32_t offset;
  uint32_t size; // in elements
  uint32_t size_in_bytes;
} dpu_array_t;


typedef struct dpu_arguments_t {
  dpu_array_t A;
  dpu_array_t B;
  dpu_array_t C;
  pimop_t kernel;
  dpu_word_t mod;
  dpu_word_t mu; // precomputed mu for Barrett reduction
  dpu_word_t scalar;
  cmp_t cmp;
  dpu_word_t bound;
  uint32_t mod_factor;
  uint32_t input_mod_factor;
  uint32_t output_mod_factor;
} dpu_arguments_t;



// Use PIM_ prefix to avoid conflicts with Google Test
#define PIM_COLOR_YELLOW "\033[1;33m"  // Bold yellow
#define PIM_COLOR_RED    "\033[1;31m"  // Bold red
#define PIM_COLOR_GREEN  "\033[1;32m"  // Bold green
#define PIM_COLOR_BLUE   "\033[1;34m"  // Bold blue
#define PIM_COLOR_RESET  "\033[0m"

#define LOG(color, fmt, ...) \
    printf(color fmt PIM_COLOR_RESET "\n", ##__VA_ARGS__)

#define LOG_INFO(fmt, ...)    LOG(PIM_COLOR_BLUE, "[ PIM INFO ] " fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)    LOG(PIM_COLOR_YELLOW, "[ PIM WARN ] " fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)   LOG(PIM_COLOR_RED, "[ PIM ERROR ] " fmt, ##__VA_ARGS__)
#define LOG_SUCCESS(fmt, ...) LOG(PIM_COLOR_GREEN, "[ PIM SUCCESS ] " fmt, ##__VA_ARGS__)