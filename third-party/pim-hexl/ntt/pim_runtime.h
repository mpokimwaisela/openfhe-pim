#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

#ifdef NTT_USE_PIM
extern "C" {
#include <dpu.h>
#include <dpu_log.h>
}
#endif

struct PIMExecutionStats {
  double total_ms = 0.0;
  double copy_in_ms = 0.0;
  double copy_out_ms = 0.0;
  double dpu_ms = 0.0;
  uint64_t stages = 0;
};

struct PIMConfig {
  bool enable = false;
  uint32_t dpus = 0;
  const char *binary_path = nullptr;
};

class PIMRuntime {
public:
  explicit PIMRuntime(const PIMConfig &cfg);
  ~PIMRuntime();

  PIMRuntime(const PIMRuntime &) = delete;
  PIMRuntime &operator=(const PIMRuntime &) = delete;

  bool execute_stage_on_pim(std::vector<u64> &data, int len, u64 step,
                            u64 modulus, int total_size) const;

  void shutdown() const;

  void reset_stats() const;

  const PIMExecutionStats &stats() const { return stats_; }

  bool enabled() const { return enabled_; }

private:
  bool enabled_;
  mutable PIMExecutionStats stats_;

#ifdef NTT_USE_PIM
  bool ensure_initialized() const;

  struct Assignment {
    struct dpu_set_t dpu;
    uint32_t start_block;
    uint32_t blocks;
    std::vector<u64> buffer;
  };

  bool run_stage(std::vector<u64> &data, int len, u64 step, u64 modulus,
                int total_size) const;

  uint32_t requested_dpus_ = 2;
  std::string binary_path_;
  mutable bool initialized_ = false;
  mutable struct dpu_set_t dpu_set_;
  mutable uint32_t actual_dpus_ = 0;
#endif
};
