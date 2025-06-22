// ============================
// pim_ntt_fixed.hpp
// Host-side NTT orchestrator
// ============================
#pragma once
#include "pim.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <utility>
#include <vector>

// Build and replicate twiddles (size N per DPU)
// This code is correct as it tries to replicate the twiddles
// across all DPUs, which is necessary for the distributed NTT
// and INTT to work correctly. Each DPU will have the same
// To understand the DPU access look at the Vector class, each DPU is a shard
namespace pim {
inline std::pair<Vector<uint64_t>, Vector<uint64_t>>
replicated_twiddles(uint32_t N, uint64_t mod) {
  int D = PIMManager::instance().num_dpus();
  if (D == 0)
    throw std::runtime_error("PIMManager not initialised");

  uint64_t omega = find_root(N, mod);

  Vector<uint64_t> W(N * D), W_inv(N * D);

  for (int i = 0; i < D; ++i) {
    W[i * D] = 0;
    W_inv[i * D] = 0;
  }

  for (uint32_t k = 1; k < N; ++k) {

    uint64_t a = mul_mod_u64(W[k - 1], omega, mod);
    uint64_t b = mul_mod_u64(W_inv[k - 1], pow_mod_u64(omega, N - 1, mod), mod);

    for (int d = 0; d < D; ++d) {
      W[k * D] = a;
      W_inv[k * D] = b;
    }
  }

  W.commit();
  W_inv.commit();
  return {std::move(W), std::move(W_inv)};
}

static void bit_reverse(Vector<uint64_t> &vec) {
  uint32_t N = vec.size();
  uint32_t logN = 31u - __builtin_clz(N);
  for (uint32_t i = 0; i < N; ++i) {
    uint32_t j = bitrev(i, logN);
    if (j > i)
      std::swap(vec[i], vec[j]);
  }
}

// Launch a single NTT/INTT stage on all DPUs
static void launch_stage(Vector<uint64_t> &data,
                         const Vector<uint64_t> &W, uint64_t mod,
                         uint32_t span, bool inverse, bool last) {
  uint32_t step = data.size() / (2 * span);
  auto args = ArgsBuilder{}
                  .A(data.shards()[0].blk.off, data.size())
                  .B(W.shards()[0].blk.off, W.size())
                  .kernel(pimop_t::NTT_STAGE)
                  .mod(mod)
                  .scalar(0ULL) // do final scaling on host
                  .mod_factor(span)
                  .in_factor(step)
                  .out_factor((inverse ? 1u : 0u) | (last ? 2u : 0u))
                  .build();

  run_kernel(args, std::tie(data, const_cast<Vector<uint64_t> &>(W)),
             std::tie(data));
}

enum class NTTDir { FORWARD, INVERSE };

inline void distributed_ntt(Vector<uint64_t> &vec,
                            const Vector<uint64_t> &W, uint64_t mod,
                            NTTDir dir) {
  const uint32_t N = vec.size();
  const uint32_t D = PIMManager::instance().num_dpus();
  if (N % D)
    throw std::runtime_error("N must be multiple of DPUs");

  const uint32_t L = N / D;
  const uint32_t logN = ilog2(N);
  const uint32_t logL = ilog2(L);
  const bool inverse = (dir == NTTDir::INVERSE);

  if (!inverse) {
    bit_reverse(vec);
    vec.commit();
  }

  uint32_t span = 1;
  for (uint32_t s = 0; s < logL; ++s) {
    bool last = (s + 1 == logN);
    launch_stage(vec, W, mod, span, inverse, last);
    span <<= 1;
  }

  for (uint32_t s = logL; s < logN; ++s) {
    uint32_t partner_bit = span / L;
    bool last = (s + 1 == logN);

    for (uint32_t d = 0; d < D; ++d) {
      uint32_t p = d ^ partner_bit;
      if (d > p)
        continue;
      uint32_t startA = d * L + span;
      uint32_t startB = p * L + span;
      for (uint32_t i = 0; i < span; ++i) {
        uint32_t idxA = startA + i;
        uint32_t idxB = startB + i;
        if (idxA < N && idxB < N)
          std::swap(vec[startA + i], vec[startB + i]);
      }
    }
    launch_stage(vec, W, mod, span, inverse, last);
    span <<= 1;
  }

  if (inverse) {
    bit_reverse(vec);
    vec.commit();
    uint64_t invN = inverse_mod_u64(N, mod);
    for (uint32_t i = 0; i < N; ++i)
      vec[i] = mul_mod_u64(vec[i], invN, mod);
  } else {
    vec.commit();
  }
}

} // namespace pim
