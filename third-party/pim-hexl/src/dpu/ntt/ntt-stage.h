#include "../memory.h"

static int ntt_stage_kernel(void) {
  const unsigned tid = me();
  // unpack arguments
  uint32_t offA = DPU_INPUT_ARGUMENTS.A.offset;
  uint32_t total = DPU_INPUT_ARGUMENTS.A.size;
  uint32_t offW = DPU_INPUT_ARGUMENTS.B.offset;
  uint32_t span = DPU_INPUT_ARGUMENTS.mod_factor;       // hop
  uint32_t step = DPU_INPUT_ARGUMENTS.input_mod_factor; // stride
  bool inv = (DPU_INPUT_ARGUMENTS.output_mod_factor & 1);
  bool last = (DPU_INPUT_ARGUMENTS.output_mod_factor & 2);
  dpu_word_t q = DPU_INPUT_ARGUMENTS.mod;
  dpu_word_t twoq = q << 1;
  dpu_word_t ninv = DPU_INPUT_ARGUMENTS.scalar; // n⁻¹ if last

  __mram_ptr dpu_word_t *A =
      (__mram_ptr dpu_word_t *)(DPU_MRAM_HEAP_POINTER + offA);
  __mram_ptr dpu_word_t *W =
      (__mram_ptr dpu_word_t *)(DPU_MRAM_HEAP_POINTER + offW);
  dpu_word_t *buf = (dpu_word_t *)mem_alloc(CHUNK_BYTES);

  // each tasklet takes blocks of 2·span
  for (uint32_t base = tid * span * 2; base < total;
       base += span * 2 * NR_TASKLETS) {

    uint32_t left = span * 2;
    uint32_t off = 0;
    while (left) {
      uint32_t take = (left > CHUNK_ELEMS ? CHUNK_ELEMS : left);

      // read chunk of A
      mram_read(A + base + off, buf, take * sizeof(dpu_word_t));

      // do butterflies
      for (uint32_t j = 0; j < take; ++j) {
        uint32_t x = off + j;
        uint32_t mate = x ^ span;
        if (mate < x || mate >= span * 2)
          continue;

        // corrected twiddle index:
        uint32_t tw_idx = (x & (span - 1)) * step;
        dpu_word_t w = mram_read_uw(W + tw_idx);

        butterfly(&buf[j], &buf[mate - off], w, q, twoq);

        if (inv && last) {
          buf[j] = mul_mod(buf[j], ninv, q,0);
          buf[mate - off] = mul_mod(buf[mate - off], ninv, q,0);
        }
      }

      // write back
      mram_write(buf, A + base + off, take * sizeof(dpu_word_t));
      left -= take;
      off += take;
    }
  }
  return 0;
}

int ntt_stage(void) { return ntt_stage_kernel(); }
