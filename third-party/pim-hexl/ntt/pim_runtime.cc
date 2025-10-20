#include "pim_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <utility>

PIMRuntime::PIMRuntime(const PIMConfig &cfg) : enabled_(cfg.enable)
#ifdef NTT_USE_PIM
                                               ,
                                               requested_dpus_(cfg.dpus),
                                               binary_path_(cfg.binary_path ? cfg.binary_path : "")
#endif
{
}

PIMRuntime::~PIMRuntime() { shutdown(); }

bool PIMRuntime::execute_stage_on_pim(std::vector<u64> &data, int len, u64 step,
                                      u64 modulus, int total_size) const {
#ifdef NTT_USE_PIM
  if (!enabled_ || len <= 0)
    return enabled_;
  if (!ensure_initialized() || actual_dpus_ == 0)
    return false;
  return run_stage(data, len, step, modulus, total_size);
#else
  (void)data;
  (void)len;
  (void)step;
  (void)modulus;
  (void)total_size;
  return false;
#endif
}

void PIMRuntime::shutdown() const {
#ifdef NTT_USE_PIM
  if (initialized_) {
    dpu_free(dpu_set_);
    initialized_ = false;
    actual_dpus_ = 0;
  }
#endif
}

void PIMRuntime::reset_stats() const { stats_ = PIMExecutionStats{}; }

#ifdef NTT_USE_PIM
bool PIMRuntime::ensure_initialized() const {
  if (!enabled_)
    return false;
  if (initialized_)
    return actual_dpus_ > 0;
  if (requested_dpus_ == 0)
    return false;
  const char *path = binary_path_.empty() ? "dpu" : binary_path_.c_str();
  DPU_ASSERT(dpu_alloc(requested_dpus_, NULL, &dpu_set_));
  DPU_ASSERT(dpu_load(dpu_set_, path, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set_, &actual_dpus_));
  initialized_ = true;
  return actual_dpus_ > 0;
}

bool PIMRuntime::run_stage(std::vector<u64> &data, int len, u64 step,
                           u64 modulus, int total_size) const {
  const auto stage_start = std::chrono::steady_clock::now();

  const uint32_t span = static_cast<uint32_t>(2 * len);
  if (span == 0)
    return false;
  const uint32_t total_blocks = total_size / span;
  if (total_blocks == 0)
    return true;

  std::vector<Assignment> assignments;
  assignments.reserve(actual_dpus_);

  uint32_t blocks_remaining = total_blocks;
  uint32_t next_block = 0;
  const uint32_t max_per =
      actual_dpus_ ? (total_blocks + actual_dpus_ - 1) / actual_dpus_
                   : total_blocks;

  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set_, dpu) {
    Assignment assign{dpu, next_block, 0, {}};
    if (blocks_remaining > 0) {
      uint32_t take = std::min<uint32_t>(max_per, blocks_remaining);
      assign.blocks = take;
      assign.buffer.resize(static_cast<size_t>(take) * span);
      for (uint32_t b = 0; b < take; ++b) {
        size_t src = static_cast<size_t>(assign.start_block + b) * span;
        auto dst = assign.buffer.begin() + static_cast<ptrdiff_t>(b) * span;
        std::copy_n(data.begin() + static_cast<ptrdiff_t>(src), span, dst);
      }
      next_block += take;
      blocks_remaining -= take;
    }
    assignments.push_back(std::move(assign));
  }

  if (assignments.empty())
    return true;

  const auto copy_in_start = std::chrono::steady_clock::now();

  std::vector<NttDpuArgs> arg_buffers;
  arg_buffers.reserve(assignments.size());
  for (auto &assign : assignments) {
    NttDpuArgs payload{};
    payload.len = static_cast<uint32_t>(len);
    payload.blocks = assign.blocks;
    payload.modulus = static_cast<uint64_t>(modulus);
    payload.root_step = static_cast<uint64_t>(step);
    arg_buffers.push_back(payload);
    DPU_ASSERT(
        dpu_prepare_xfer(assign.dpu, &arg_buffers.back()));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set_, DPU_XFER_TO_DPU, "NTT_ARGS", 0,
                           sizeof(NttDpuArgs), DPU_XFER_DEFAULT));

  size_t uniform_in_bytes = 0;
  bool have_in_work = false;
  bool uniform_in = true;
  for (auto &assign : assignments) {
    if (assign.blocks == 0)
      continue;
    size_t bytes = assign.buffer.size() * sizeof(u64);
    if (!have_in_work) {
      uniform_in_bytes = bytes;
      have_in_work = true;
    } else if (bytes != uniform_in_bytes) {
      uniform_in = false;
      break;
    }
  }

  if (uniform_in && have_in_work) {
    for (auto &assign : assignments) {
      if (assign.blocks == 0)
        continue;
      DPU_ASSERT(dpu_prepare_xfer(assign.dpu, assign.buffer.data()));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set_, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME, 0, uniform_in_bytes,
                             DPU_XFER_DEFAULT));
  } else {
    for (auto &assign : assignments) {
      if (assign.blocks == 0)
        continue;
      size_t bytes = assign.buffer.size() * sizeof(u64);
      DPU_ASSERT(dpu_copy_to(assign.dpu, DPU_MRAM_HEAP_POINTER_NAME, 0,
                             assign.buffer.data(), bytes));
    }
  }

  const auto copy_in_end = std::chrono::steady_clock::now();
  const auto dpu_start = copy_in_end;

  DPU_ASSERT(dpu_launch(dpu_set_, DPU_SYNCHRONOUS));
  // DPU_FOREACH(dpu_set_, dpu) {
  //   DPU_ASSERT(dpu_log_read(dpu, stdout));
  // }

  const auto dpu_end = std::chrono::steady_clock::now();

  const auto copy_out_start = dpu_end;

  size_t uniform_out_bytes = 0;
  bool have_out_work = false;
  bool uniform_out = true;
  for (auto &assign : assignments) {
    if (assign.blocks == 0)
      continue;
    size_t bytes = assign.buffer.size() * sizeof(u64);
    if (!have_out_work) {
      uniform_out_bytes = bytes;
      have_out_work = true;
    } else if (bytes != uniform_out_bytes) {
      uniform_out = false;
      break;
    }
  }

  if (uniform_out && have_out_work) {
    for (auto &assign : assignments) {
      if (assign.blocks == 0)
        continue;
      DPU_ASSERT(dpu_prepare_xfer(assign.dpu, assign.buffer.data()));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set_, DPU_XFER_FROM_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME, 0, uniform_out_bytes,
                             DPU_XFER_DEFAULT));
  } else {
    for (auto &assign : assignments) {
      if (assign.blocks == 0)
        continue;
      size_t bytes = assign.buffer.size() * sizeof(u64);
      DPU_ASSERT(dpu_copy_from(assign.dpu, DPU_MRAM_HEAP_POINTER_NAME, 0,
                               assign.buffer.data(), bytes));
    }
  }

  for (auto &assign : assignments) {
    if (assign.blocks == 0)
      continue;
    for (uint32_t b = 0; b < assign.blocks; ++b) {
      size_t dst = static_cast<size_t>(assign.start_block + b) * span;
      auto src = assign.buffer.begin() + static_cast<ptrdiff_t>(b) * span;
      std::copy_n(src, span, data.begin() + static_cast<ptrdiff_t>(dst));
    }
  }

  const auto copy_out_end = std::chrono::steady_clock::now();
  const auto stage_end = copy_out_end;

  auto to_ms = [](auto delta) {
    return std::chrono::duration<double, std::milli>(delta).count();
  };

  stats_.copy_in_ms += to_ms(copy_in_end - copy_in_start);
  stats_.dpu_ms += to_ms(dpu_end - dpu_start);
  stats_.copy_out_ms += to_ms(copy_out_end - copy_out_start);
  stats_.total_ms += to_ms(stage_end - stage_start);
  stats_.stages += 1;

  return true;
}
#endif
