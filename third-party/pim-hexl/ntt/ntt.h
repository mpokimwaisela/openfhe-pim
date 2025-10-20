#pragma once

#include "common.h"
#include "pim_runtime.h"

#include <cstdint>
#include <string>
#include <vector>

std::vector<u64> factorize_distinct(u64 x);
u64 primitive_root(u64 p);

class NTT {
public:
  NTT(int size, u64 modulus, u64 psi_value, const PIMConfig &pim_cfg = {});

  void forward(std::vector<u64> &a) const;
  void inverse(std::vector<u64> &a) const;

  void reset_pim_stats() const;
  const PIMExecutionStats &pim_stats() const;

  const int n;
  const u64 p;
  const u64 psi;
  const u64 psi_inv;
  const u64 omega;
  const u64 omega_inv;
  const u64 ninv;

private:
  void twist(std::vector<u64> &a, u64 root, u64 scale = 1) const;
  void bit_reverse(std::vector<u64> &a) const;
  void butterflies(std::vector<u64> &a, u64 base) const;

  mutable PIMRuntime pim_runtime_;
};

std::vector<std::vector<u64>> psi_matrix(const NTT &plan,
                                         bool inverse = false);

void print_mat(const std::vector<std::vector<u64>> &M, const std::string &name);
void print_vec(const std::string &label, const std::vector<u64> &v);

std::vector<u64> matvec(const std::vector<std::vector<u64>> &M,
                        const std::vector<u64> &x, u64 p);
std::vector<u64> hadamard(const std::vector<u64> &a,
                          const std::vector<u64> &b, u64 p);
u64 centered_mod(u64 x, u64 p);

bool check_correctness(const std::vector<u64> &naive,
                       const std::vector<u64> &efficient,
                       const std::string &test_name);

void sample_vectors(int n, u64 p, std::vector<u64> &v, std::vector<u64> &w,
                    uint64_t seed = 0);
