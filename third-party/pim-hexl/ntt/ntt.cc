#include "ntt.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

vector<u64> factorize_distinct(u64 x) {
  vector<u64> factors;
  for (u64 d = 2; d * d <= x; ++d) {
    if (x % d == 0) {
      factors.push_back(d);
      while (x % d == 0)
        x /= d;
    }
  }
  if (x > 1)
    factors.push_back(x);
  return factors;
}

u64 primitive_root(u64 p) {
  u64 phi = p - 1;
  auto fac = factorize_distinct(phi);
  for (u64 g = 2; g < p; ++g) {
    bool ok = true;
    for (u64 q : fac) {
      if (modpow(g, phi / q, p) == 1) {
        ok = false;
        break;
      }
    }
    if (ok)
      return g;
  }
  return -1;
}

NTT::NTT(int size, u64 modulus, u64 psi_value, const PIMConfig &pim_cfg)
    : n(size), p(modulus), psi(psi_value), psi_inv(modinv(psi_value, modulus)),
      omega(modmul(psi_value, psi_value, modulus)),
      omega_inv(modmul(psi_inv, psi_inv, modulus)),
      ninv(modinv(size % modulus, modulus)), pim_runtime_(pim_cfg) {}

void NTT::forward(vector<u64> &a) const {
  twist(a, psi);
  bit_reverse(a);
  butterflies(a, omega);
}

void NTT::inverse(vector<u64> &a) const {
  bit_reverse(a);
  butterflies(a, omega_inv);
  twist(a, psi_inv, ninv);
}

void NTT::reset_pim_stats() const { pim_runtime_.reset_stats(); }

const PIMExecutionStats &NTT::pim_stats() const { return pim_runtime_.stats(); }

void NTT::twist(vector<u64> &a, u64 root, u64 scale) const {
  u64 power = 1;
  for (auto &v : a) {
    v = modmul(v, modmul(scale, power, p), p);
    power = modmul(power, root, p);
  }
}

void NTT::bit_reverse(vector<u64> &a) const {
  int size = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < size; ++i) {
    int bit = size >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      swap(a[i], a[j]);
  }
}

void NTT::butterflies(vector<u64> &a, u64 base) const {
  for (int len = 1; len < n; len <<= 1) {
    u64 step = modpow(base, n / (2 * len), p);
    if (pim_runtime_.execute_stage_on_pim(a, len, step, p, n))
      continue;
    // cout << "Executing NTT stage on CPU with len=" << len << " step=" << step << " modulus=" << p << "\n";
    for (int i = 0; i < n; i += 2 * len) {
      u64 w = 1;
      for (int j = 0; j < len; ++j) {
        u64 u = a[i + j];
        u64 v = modmul(a[i + j + len], w, p);
        u64 sum = modadd(u, v, p);
        u64 diff = modsub(u, v, p);
        a[i + j] = sum;
        a[i + j + len] = diff;
        w = modmul(w, step, p);
      }
    }
  }
}

vector<vector<u64>> psi_matrix(const NTT &plan, bool inverse) {
  vector<vector<u64>> M(plan.n, vector<u64>(plan.n));
  const u64 base = inverse ? plan.psi_inv : plan.psi;
  const int order = 2 * plan.n;

  if (!inverse) {
    for (int i = 0; i < plan.n; ++i) {
      u64 step = modpow(base, (2 * i + 1) % order, plan.p);
      u64 value = 1;
      for (int j = 0; j < plan.n; ++j) {
        M[i][j] = value;
        value = modmul(value, step, plan.p);
      }
    }
  } else {
    for (int i = 0; i < plan.n; ++i) {
      u64 start = modpow(base, i % order, plan.p);
      u64 step = modpow(base, (2 * i) % order, plan.p);
      u64 value = start;
      for (int j = 0; j < plan.n; ++j) {
        M[i][j] = modmul(plan.ninv, value, plan.p);
        value = modmul(value, step, plan.p);
      }
    }
  }
  return M;
}

void print_mat(const vector<vector<u64>> &M, const string &name) {
  cout << name << " =\n";
  for (const auto &row : M) {
    for (int j = 0; j < static_cast<int>(row.size()); ++j) {
      cout << row[j] << (j + 1 < static_cast<int>(row.size()) ? ' ' : '\n');
    }
  }
  cout << '\n';
}

void print_vec(const string &label, const vector<u64> &v) {
  cout << label;
  for (auto x : v)
    cout << ' ' << x;
  cout << '\n';
}

vector<u64> matvec(const vector<vector<u64>> &M, const vector<u64> &x, u64 p) {
  int n = static_cast<int>(M.size());
  vector<u64> y(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      u64 prod = modmul(M[i][j], x[j], p);
      y[i] = modadd(y[i], prod, p);
    }
  }
  return y;
}

vector<u64> hadamard(const vector<u64> &a, const vector<u64> &b, u64 p) {
  vector<u64> c(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    c[i] = modmul(a[i], b[i], p);
  return c;
}

u64 centered_mod(u64 x, u64 p) { return (x >= p / 2) ? x - p : x; }

bool check_correctness(const vector<u64> &naive, const vector<u64> &efficient,
                       const string &test_name) {
  // ANSI color codes: bold green for PASS, bold red for FAIL
  const string bold_green = "\033[1;32m";
  const string bold_red = "\033[1;31m";
  const string reset = "\033[0m";
  
  if (naive.size() != efficient.size()) {
    cout << bold_red << "[FAIL]" << reset << " " << test_name << ": Size mismatch\n";
    return false;
  }

  for (size_t i = 0; i < naive.size(); ++i) {
    if (naive[i] != efficient[i]) {
      cout << bold_red << "[FAIL]" << reset << " " << test_name << " at index " << i
           << ": naive=" << naive[i] << ", efficient=" << efficient[i] << "\n";
      return false;
    }
  }

  cout << bold_green << "[PASS]" << reset << " " << test_name << "\n";
  return true;
}

// Sample two vectors v and w of length n with values uniform in [0, p-1].
// If seed == 0, the random_device will be used to seed the generator.
void sample_vectors(int n, u64 p, vector<u64> &v, vector<u64> &w,
                    uint64_t seed) {
  v.assign(n, 0);
  w.assign(n, 0);
  std::mt19937_64 rng;
  if (seed == 0) {
    std::random_device rd;
    rng.seed(rd());
  } else {
    rng.seed(seed);
  }
  std::uniform_int_distribution<u64> dist(0, p - 1);
  for (int i = 0; i < n; ++i) {
    v[i] = dist(rng);
    w[i] = dist(rng);
  }
}
