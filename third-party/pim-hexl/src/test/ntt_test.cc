#include "pim.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>


// -----------------------------------------------------------------------------
// Forward NTT (Cooley–Tukey, DIT)
// -----------------------------------------------------------------------------
void cpu_ntt(std::vector<uint64_t> &a, const std::vector<uint64_t> &w,
             uint64_t mod) {
  const uint32_t N = static_cast<uint32_t>(a.size());
  const uint32_t logN = ilog2(N);

  for (uint32_t i = 0; i < N; ++i) {
    uint32_t j = bitrev(i, logN);
    if (j > i)
      std::swap(a[i], a[j]);
  }

  for (uint32_t len = 2; len <= N; len <<= 1) {
    uint32_t step = N / len;
    for (uint32_t i = 0; i < N; i += len) {
      for (uint32_t j = 0; j < len / 2; ++j) {
        uint64_t u = a[i + j];
        uint64_t v = mul_mod_u64(a[i + j + len / 2], w[step * j], mod);
        a[i + j] = add_mod_u64(u, v, mod);
        a[i + j + len / 2] = sub_mod_u64(u, v, mod);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Inverse NTT (Gentleman–Sande, DIF) – uses w_inv
// -----------------------------------------------------------------------------
void cpu_intt(std::vector<uint64_t> &a, const std::vector<uint64_t> &w_inv,
              uint64_t mod) {
  const uint32_t N = static_cast<uint32_t>(a.size());
  const uint32_t logN = ilog2(N);

  for (uint32_t len = N; len > 1; len >>= 1) {
    uint32_t step = N / len;
    for (uint32_t i = 0; i < N; i += len) {
      for (uint32_t j = 0; j < len / 2; ++j) {
        uint64_t u = a[i + j];
        uint64_t v = a[i + j + len / 2];
        a[i + j] = add_mod_u64(u, v, mod);
        a[i + j + len / 2] = mul_mod_u64(sub_mod_u64(u, v, mod), w_inv[step * j], mod);
      }
    }
  }

  for (uint32_t i = 0; i < N; ++i) { // bit-reverse back
    uint32_t j = bitrev(i, logN);
    if (j > i)
      std::swap(a[i], a[j]);
  }

  uint64_t invN = inverse_mod_u64(N, mod);
  for (auto &x : a)
    x = mul_mod_u64(x, invN, mod);
}

// -----------------------------------------------------------------------------
// Quick round-trip test
// -----------------------------------------------------------------------------
int main() {
  constexpr uint32_t N   = 4096;
    constexpr uint64_t mod = 12289;    // 2·N | (mod-1)

  // Generate twiddles
  uint64_t omega = find_root(N, mod);
  std::vector<uint64_t> w(N), w_inv(N);
  w[0] = w_inv[0] = 1;
  for (uint32_t k = 1; k < N; ++k) {
    w[k] = mul_mod_u64(w[k - 1], omega, mod);
    w_inv[k] = mul_mod_u64(w_inv[k - 1], pow_mod_u64(omega, N - 1, mod), mod);
  }

  // Generate random polynomials a(x), b(x)
  std::vector<uint64_t> a(N), b(N);
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
  for (uint32_t i = 0; i < N / 2; ++i) { // Only degree N/2 - 1
    a[i] = dist(rng);
    b[i] = dist(rng);
  }

  // Save reference result (naive convolution mod p)
  std::vector<uint64_t> expected(N, 0);
  for (uint32_t i = 0; i < N / 2; ++i)
    for (uint32_t j = 0; j < N / 2; ++j)
      expected[i + j] = add_mod_u64(expected[i + j], mul_mod_u64(a[i], b[j], mod), mod);

  // NTT both
  cpu_ntt(a, w, mod);
  cpu_ntt(b, w, mod);

  // Pointwise multiply
  std::vector<uint64_t> c(N);
  for (uint32_t i = 0; i < N; ++i)
    c[i] = mul_mod_u64(a[i], b[i], mod);

  // Inverse NTT
  cpu_intt(c, w_inv, mod);

  // Compare
  bool ok = true;
  for (uint32_t i = 0; i < N; ++i) {
    if (c[i] != expected[i]) {
      std::cout << "Mismatch at " << i << ": got " << c[i] << ", expected "
                << expected[i] << "\n";
      ok = false;
    }
  }

  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
