#include "ntt.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <utility>
#include <vector>

namespace {

template <typename Func> double measure_ms(Func &&fn) {
  auto start = std::chrono::steady_clock::now();
  std::forward<Func>(fn)();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// u64 mulmod(u64 a, u64 b, u64 m) {
// #if defined(__SIZEOF_INT128__)
//   return static_cast<u64>((__uint128_t)a * (__uint128_t)b % m);
// #else
//   u64 res = 0;
//   a %= m;
//   while (b) {
//     if (b & 1)
//       res = (res + a) % m;
//     a = (a * 2) % m;
//     b >>= 1;
//   }
//   return res;
// #endif
// }

// u64 powmod(u64 a, u64 e, u64 m) {
//   u64 r = 1 % m;
//   a %= m;
//   while (e) {
//     if (e & 1)
//       r = mulmod(r, a, m);
//     a = mulmod(a, a, m);
//     e >>= 1;
//   }
//   return r;
// }

bool is_prime(u64 n) {
  if (n < 2)
    return false;
  for (u64 p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL}) {
    if (n % p == 0)
      return n == p;
  }
  u64 d = n - 1;
  int s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  const u64 witnesses[] = {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL};
  for (u64 a : witnesses) {
    if (a % n == 0)
      continue;
    u64 x = modpow(a, d, n);
    if (x == 1 || x == n - 1)
      continue;
    bool composite = true;
    for (int r = 1; r < s; ++r) {
      x = modmul(x, x, n);
      if (x == n - 1) {
        composite = false;
        break;
      }
    }
    if (composite)
      return false;
  }
  return true;
}

// u64 find_ntt_prime(u64 n) {
//   const u64 step = 2ULL * n;
//   const u64 max_modulus = (1ULL << 60) - 1;
//   const u64 max_k = (max_modulus - 1) / step;
//   for (u64 k = max_k; k > 0; --k) {
//     u64 candidate = step * k + 1;
//     if (candidate > max_modulus)
//       continue;
//     if (is_prime(candidate))
//       return candidate;
//   }
//   throw std::runtime_error("Failed to find NTT prime for given n");
// }

} // namespace

int main() {
  struct TestPair {
    int n;
    u64 p;
  };

  const std::vector<TestPair> tests = {
      // {1024, 12289ULL},
      // {1024, 1152921504606830593ULL},
      // {2048, 12289ULL},
      // {2048, 1152921504606830593ULL},
      // {4096, 40961ULL},
      // {4096, 1152921504606830593ULL},
      {8192, 65537ULL},
      {8192, 1152921504606830593ULL},
      // {16384, 65537ULL},
      // {16384, 1152921504606748673ULL},
      // {32768, 65537ULL},
      // {32768, 1152921504606584833ULL},
      // {65536, 786433ULL},
      // {65536, 1152921504606584833ULL},
  };

  const int naive_limit = 2048;

  PIMConfig pim_cfg{};
#ifdef NTT_USE_PIM
  pim_cfg.enable = true;
  pim_cfg.dpus = 256;
  pim_cfg.binary_path = "dpu.bin";
#endif

  bool all_pass = true;

  std::cout << std::fixed << std::setprecision(3);

  for (size_t idx = 0; idx < tests.size(); ++idx) {
    int n = tests[idx].n;
    u64 p = tests[idx].p;

    if (!is_prime(p)) {
      std::cout << "[WARN] Skipping composite modulus p=" << p
                << " for n=" << n << "\n";
      all_pass = false;
      continue;
    }

    std::cout << "\n=== Testing n=" << n << " p=" << p << " ===\n";

    std::vector<u64> a, b;
    sample_vectors(n, p, a, b,
                   static_cast<uint64_t>(42 + idx)); // deterministic seed

    u64 g = primitive_root(p);
    if (g == static_cast<u64>(-1)) {
      std::cout << "[FAIL] primitive root search failed for p=" << p << "\n";
      all_pass = false;
      continue;
    }

    u64 psi = modpow(g, (p - 1) / (2 * n), p);
    NTT plan_cpu(n, p, psi);

    // CPU roundtrip check
    auto roundtrip = a;
    plan_cpu.forward(roundtrip);
    plan_cpu.inverse(roundtrip);
    std::ostringstream label_rt;
    label_rt << "cpu roundtrip n=" << n << " p=" << p;
    bool cpu_roundtrip_ok =
        check_correctness(a, roundtrip, label_rt.str());
    all_pass = all_pass && cpu_roundtrip_ok;

    // CPU polynomial multiplication
    double cpu_forward_ms =
        measure_ms([&] {
          auto tmp = a;
          plan_cpu.forward(tmp);
        });

    std::vector<u64> cpu_result;
    double cpu_total_ms = measure_ms([&] {
      auto va = a;
      auto vb = b;
      plan_cpu.forward(va);
      plan_cpu.forward(vb);
      auto pointwise = hadamard(va, vb, p);
      plan_cpu.inverse(pointwise);
      cpu_result = std::move(pointwise);
    });

    std::cout << "CPU forward NTT: " << cpu_forward_ms << " ms\n";
    std::cout << "CPU polynomial multiply: " << cpu_total_ms << " ms\n";

    if (n <= naive_limit) {
      auto F = psi_matrix(plan_cpu);
      auto Finv = psi_matrix(plan_cpu, true);
      auto v_ntt = matvec(F, a, p);
      auto w_ntt = matvec(F, b, p);
      auto res = hadamard(v_ntt, w_ntt, p);
      auto naive = matvec(Finv, res, p);
      std::ostringstream label_naive;
      label_naive << "cpu vs naive n=" << n << " p=" << p;
      bool cpu_naive_ok =
          check_correctness(naive, cpu_result, label_naive.str());
      all_pass = all_pass && cpu_naive_ok;
    } else {
      std::cout << "[INFO] Naive verification skipped (n=" << n << ")\n";
    }

#ifdef NTT_USE_PIM
    if (pim_cfg.enable) {
      NTT plan_pim(n, p, psi, pim_cfg);

      auto pim_roundtrip = a;
      plan_pim.forward(pim_roundtrip);
      plan_pim.inverse(pim_roundtrip);
      std::ostringstream label_pim_rt;
      label_pim_rt << "pim roundtrip n=" << n << " p=" << p;
      bool pim_roundtrip_ok =
          check_correctness(a, pim_roundtrip, label_pim_rt.str());
      all_pass = all_pass && pim_roundtrip_ok;

      plan_pim.reset_pim_stats();
      double pim_forward_ms =
          measure_ms([&] {
            auto tmp = a;
            plan_pim.forward(tmp);
          });
      auto pim_forward_stats = plan_pim.pim_stats();

      plan_pim.reset_pim_stats();
      std::vector<u64> pim_result;
      double pim_total_ms = measure_ms([&] {
        auto va = a;
        auto vb = b;
        plan_pim.forward(va);
        plan_pim.forward(vb);
        auto pointwise = hadamard(va, vb, p);
        plan_pim.inverse(pointwise);
        pim_result = std::move(pointwise);
      });
      auto pim_total_stats = plan_pim.pim_stats();

      std::ostringstream label_pim_cmp;
      label_pim_cmp << "pim vs cpu n=" << n << " p=" << p;
      bool pim_cpu_ok =
          check_correctness(cpu_result, pim_result, label_pim_cmp.str());
      all_pass = all_pass && pim_cpu_ok;

      std::cout << "PIM forward NTT: " << pim_forward_ms << " ms\n";
      std::cout << "  PIM breakdown forward: total=" << pim_forward_stats.total_ms
                << " ms copy_to=" << pim_forward_stats.copy_in_ms
                << " ms dpu=" << pim_forward_stats.dpu_ms
                << " ms copy_from=" << pim_forward_stats.copy_out_ms
                << " ms stages=" << pim_forward_stats.stages << "\n";
      std::cout << "PIM polynomial multiply: " << pim_total_ms << " ms\n";
      std::cout << "  PIM breakdown total: total=" << pim_total_stats.total_ms
                << " ms copy_to=" << pim_total_stats.copy_in_ms
                << " ms dpu=" << pim_total_stats.dpu_ms
                << " ms copy_from=" << pim_total_stats.copy_out_ms
                << " ms stages=" << pim_total_stats.stages << "\n";
    } else {
      std::cout << "[INFO] PIM disabled at runtime.\n";
    }
#else
    std::cout << "[INFO] PIM support disabled at compile time.\n";
#endif
  }

  if (!all_pass) {
    std::cout << "\nAt least one configuration failed.\n";
    return 1;
  }

  std::cout << "\nAll configurations passed.\n";
  return 0;
}
