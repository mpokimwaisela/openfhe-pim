#include "pim.hpp"
#include <iostream>
#include <random>

int main() {
  pim::Init(1, "main.dpu");
  constexpr uint32_t N = 1024;
  constexpr uint64_t mod = 12289;

  auto W = pim::replicated_twiddles(N, mod);

  std::vector<uint64_t> orig(N);
  std::mt19937_64 rng(42);
  for (auto &x : orig)
    x = rng() % mod;

  pim::Vector<uint64_t> vec(N);
  for (uint32_t i = 0; i < N; ++i)
    vec[i] = orig[i];

  distributed_ntt(vec, W.first, mod, pim::NTTDir::FORWARD);
  distributed_ntt(vec, W.second, mod, pim::NTTDir::INVERSE);

  bool ok = true;
  for (uint32_t i = 0; i < N; ++i)
    if (vec[i] != orig[i]) {
      ok = false;
      break;
    }

  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}