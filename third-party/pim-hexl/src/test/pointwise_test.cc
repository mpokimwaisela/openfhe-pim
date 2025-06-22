// ─── test_all_kernels.cpp ─────────────────────────────────────
#include "pim.hpp"
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/* ── CPU reference helpers ──────────────────────────────────── */
static uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t m) {
  return (a >= b) ? a - b : a + m - b;
}
static uint64_t mul_mod_exact(uint64_t a, uint64_t b, uint64_t m) {
#ifdef __SIZEOF_INT128__
  return static_cast<unsigned __int128>(a) * b % m;
#else
  const uint64_t M = 0xFFFFFFFFull;
  uint64_t a_hi = a >> 32, a_lo = a & M;
  uint64_t b_hi = b >> 32, b_lo = b & M;
  __uint128_t full =
      (__uint128_t(a_hi) * b_hi << 64) +
      (__uint128_t(a_hi) * b_lo + __uint128_t(a_lo) * b_hi << 32) +
      (__uint128_t(a_lo) * b_lo);
  return uint64_t(full % m);
#endif
}
static bool cmp_ok(uint64_t v, uint64_t bound, cmp_t c) {
  switch (c) {
  case CMP_EQ:
    return v == bound;
  case CMP_NE:
    return v != bound;
  case CMP_LT:
    return v < bound;
  case CMP_LE:
    return v <= bound;
  case CMP_NLT:
    return v >= bound;
  case CMP_NLE:
    return v > bound;
  case CMP_TRUE:
    return true;
  default:
    return false;
  }
}

/* ── Build args helper ─────────────────────────────────────── */
template <class Buf>
dpu_arguments_t make_args(pimop_t op, uint64_t mod, uint64_t scalar, cmp_t cmp,
                          uint64_t bound, uint32_t in_f, uint32_t out_f, Buf &A,
                          Buf &B, Buf &C) {
  size_t elems = A.shards()[0].host.size();
  auto bldr = ArgsBuilder{}
                  .A(A.shards()[0].blk.off, elems)
                  .C(C.shards()[0].blk.off, elems)
                  .kernel(op)
                  .mod(mod)
                  .scalar(scalar)
                  .cmp(cmp)
                  .bound(bound)
                  .in_factor(in_f)
                  .out_factor(out_f);
  if (op == MOD_ADD || op == MOD_SUB || op == MOD_MUL ||
      (op == FMA_MOD && B.shards().size() > 0 /*with addend*/)) {
    bldr.B(B.shards()[0].blk.off, elems);
  }
  return bldr.build();
}

/* ── Arithmetic tests ──────────────────────────────────────── */
void test_arith(pimop_t op, pim::Vector<uint64_t> &A, pim::Vector<uint64_t> &B,
                pim::Vector<uint64_t> &C, uint64_t mod, uint64_t scalar = 0) {
  std::mt19937_64 rng(1000 + op);
  std::uniform_int_distribution<uint64_t> d(0, mod - 1);
  for (size_t i = 0; i < A.size(); ++i) {
    A[i] = d(rng);
    B[i] = d(rng);
    C[i] = 0;
  }
  auto args = make_args(op, mod, scalar, CMP_TRUE, 0, 1, 1, A, B, C);
  run_kernel(args, std::tie(A, B), std::tie(C));

  bool ok = true;
  for (size_t i = 0; i < A.size(); ++i) {
    uint64_t exp = 0;
    switch (op) {
    case MOD_ADD:
      exp = (A[i] + B[i]) % mod;
      break;
    case MOD_ADD_SCALAR:
      exp = (A[i] + scalar) % mod;
      break;
    case MOD_SUB:
      exp = sub_mod(A[i], B[i], mod);
      break;
    case MOD_SUB_SCALAR:
      exp = sub_mod(A[i], scalar, mod);
      break;
    case MOD_MUL:
      exp = mul_mod_exact(A[i], B[i], mod);
      break;
    default:
      break;
    }
    if (C[i] != exp) {
      ok = false;
      break;
    }
  }
  std::cout << std::left << std::setw(18)
            << (op == MOD_ADD          ? "MOD_ADD"
                : op == MOD_ADD_SCALAR ? "MOD_ADD_SCALAR"
                : op == MOD_SUB        ? "MOD_SUB"
                : op == MOD_SUB_SCALAR ? "MOD_SUB_SCALAR"
                                       : "MOD_MUL")
            << (ok ? "[OK]\n" : "[FAIL]\n");
  if (!ok)
    std::exit(1);
}

/* ── Compare tests ─────────────────────────────────────────── */
void test_cmp(pimop_t op, cmp_t code, pim::Vector<uint64_t> &A,
              pim::Vector<uint64_t> &B, pim::Vector<uint64_t> &C, uint64_t mod,
              uint64_t diff, uint64_t bound) {
  std::mt19937_64 rng(2000 + op + code);
  std::uniform_int_distribution<uint64_t> d(0, 4 * bound);
  for (size_t i = 0; i < A.size(); ++i) {
    A[i] = d(rng);
    C[i] = 0;
  }
  auto args = make_args(op, mod, diff, code, bound, 1, 1, A, B, C);
  run_kernel(args, std::tie(A), std::tie(C));

  bool ok = true;
  for (size_t i = 0; i < A.size(); ++i) {
    uint64_t v = (op == CMP_SUB_MOD ? A[i] % mod : A[i]);
    uint64_t exp = v;
    if (op == CMP_ADD) {
      if (cmp_ok(v, bound, code))
        exp = v + diff;
    } else {
      if (cmp_ok(v, bound, code))
        exp = sub_mod(v, diff, mod);
    }
    if (C[i] != exp) {
      ok = false;
      break;
    }
  }
  std::cout << std::left << std::setw(12)
            << (op == CMP_ADD ? "CMP_ADD" : "CMP_SUB_MOD")
            << " cmp=" << int(code) << (ok ? " [OK]\n" : " [FAIL]\n");
  if (!ok)
    std::exit(1);
}

/* ── FMA-MOD end-to-end check ────────────────────────────────── */
/* ── single FMA-MOD test (with or without addend) ───────────── */
void test_fma(pim::Vector<uint64_t> &A,
              pim::Vector<uint64_t> &B, // will be ignored if no addend
              pim::Vector<uint64_t> &C, uint64_t mod, uint64_t scalar,
              bool with_addend, uint32_t mod_factor) {
  /* Intel reference vectors */
  const std::vector<uint64_t> vec_a{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<uint64_t> vec_b =
      with_addend ? std::vector<uint64_t>{9, 10, 11, 12, 13, 14, 15, 16}
                  : std::vector<uint64_t>(8, 0);

  for (size_t i = 0; i < 8; ++i) {
    A[i] = vec_a[i];
    B[i] = vec_b[i];
    C[i] = 0;
  }

  /* ---------------- Build argument block ------------------ */
  ArgsBuilder ab;
  ab.A(A.shards()[0].blk.off, 8)
      .C(C.shards()[0].blk.off, 8)
      .kernel(FMA_MOD)
      .mod(mod)
      .scalar(scalar)
      .mod_factor(mod_factor);

  if (with_addend) // only when an addend is used
    ab.B(B.shards()[0].blk.off, 8);

  auto args = ab.build();

  /* ---------------- Launch kernel ------------------------- */
  if (with_addend)
    run_kernel(args, std::tie(A, B), std::tie(C)); // A,B → C
  else
    run_kernel(args, std::tie(A), std::tie(C)); // A   → C

  /* ---------------- Reference check ----------------------- */
  bool ok = true;
  for (size_t i = 0; i < 8; ++i) {
    uint64_t prod = mul_mod_exact(vec_a[i] % mod, scalar, mod);
    uint64_t exp = with_addend ? (prod + vec_b[i]) % mod : prod;
    if (C[i] != exp) {
      ok = false;
      break;
    }
  }

  std::cout << std::left << std::setw(16)
            << (with_addend ? "FMA_MOD(+add)" : "FMA_MOD(no-add)")
            << (ok ? "[OK]\n" : "[FAIL]\n");
  if (!ok)
    std::exit(1);
}
/* ── REDUCE_MOD tests ──────────────────────────────────────── */
void test_reduce(const std::vector<uint64_t> &in,
                 const std::vector<uint64_t> &exp, uint64_t mod, uint32_t in_f,
                 uint32_t out_f) {
  size_t K = in.size();
  pim::Vector<uint64_t> A(K), B(K), C(K);
  for (size_t i = 0; i < K; ++i) {
    A[i] = in[i];
    C[i] = 0;
  }
  auto args = make_args(MOD_REDUCE, mod, 0, CMP_TRUE, 0, in_f, out_f, A, B, C);
  run_kernel(args, std::tie(A), std::tie(C));

  bool ok = true;
  for (size_t i = 0; i < K; ++i) {
    if (C[i] != exp[i]) {
      ok = false;
      break;
    }
  }
  std::cout << "REDUCE f=" << in_f << "->" << out_f
            << (ok ? " [OK]\n" : " [FAIL]\n");
  if (!ok)
    std::exit(1);
}

/* ── main: drive every test ───────────────────────────────── */
int main() {
  const size_t N = 1024, D = 2;
  pim::Init(D, "main.dpu");

  pim::Vector<uint64_t> A(N), B(N), C(N);

  // Arithmetic
  test_arith(MOD_ADD, A, B, C, 257);
  test_arith(MOD_ADD_SCALAR, A, B, C, 257, 5);
  test_arith(MOD_SUB, A, B, C, 263);
  test_arith(MOD_SUB_SCALAR, A, B, C, 263, 5);
  test_arith(MOD_MUL, A, B, C, 269);

  // Compare (all 8 codes)
  for (auto c : {CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_NLT, CMP_NLE, CMP_TRUE,
                 CMP_FALSE}) {
    test_cmp(CMP_ADD, c, A, B, C, /*mod*/ 0, /*diff*/ 3, /*bound*/ 7);
    test_cmp(CMP_SUB_MOD, c, A, B, C, /*mod*/ 211, /*diff*/ 2, /*bound*/ 5);
  }

  // FMA_MOD
  test_fma(A, B, C, 769, 1, true, /*mod_factor=*/1);
  test_fma(A, B, C, 769, 1, false, /*mod_factor=*/1);

  // REDUCE_MOD
  test_reduce({0, 450, 735, 900, 1350, 1459}, {0, 450, 735, 900, 1350, 1459},
              /*mod*/ 750, /*in_f*/ 2, /*out_f*/ 2);

  test_reduce({2, 4, 1600, 2500}, {2, 4, 100, 250},
              /*mod*/ 750, /*in_f*/ 4, /*out_f*/ 1);

  test_reduce({2, 4, 1600, 2500}, {2, 4, 100, 250},
              /*mod*/ 750, /*in_f*/ 750, /*out_f*/ 1);

  test_reduce({0, 450, 735, 900, 1350, 1459}, {0, 450, 5, 170, 620, 729},
              /*mod*/ 730, /*in_f*/ 2, /*out_f*/ 1);

  test_reduce({1, 730, 1000, 1460, 2100, 2919}, {1, 730, 1000, 0, 640, 1459},
              /*mod*/ 730, /*in_f*/ 4, /*out_f*/ 2);

  std::cout << "All kernels passed!\n";
  return 0;
}
