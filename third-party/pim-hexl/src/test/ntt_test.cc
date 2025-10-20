#include <bits/stdc++.h>
#include <iostream>
using namespace std;
using i64 = long long;
using u128 = unsigned __int128;

i64 modmul(i64 a, i64 b, i64 m) { return (i64)((u128)a * b % m); }
i64 modpow(i64 a, long long e, i64 m) {
  i64 r = 1 % m;
  a %= m;
  while (e) {
    if (e & 1)
      r = modmul(r, a, m);
    a = modmul(a, a, m);
    e >>= 1;
  }
  return r;
}
i64 modinv(i64 a, i64 p) { return modpow(a, p - 2, p); } // p must be prime

vector<i64> factorize_distinct(i64 x) {
  vector<i64> f;
  for (i64 d = 2; d * d <= x; ++d) {
    if (x % d == 0) {
      f.push_back(d);
      while (x % d == 0)
        x /= d;
    }
  }
  if (x > 1)
    f.push_back(x);
  return f;
}
i64 primitive_root(i64 p) {
  i64 phi = p - 1;
  auto fac = factorize_distinct(phi);
  for (i64 g = 2; g < p; ++g) {
    bool ok = true;
    for (i64 q : fac) {
      if (modpow(g, phi / q, p) == 1) {
        ok = false;
        break;
      }
    }
    if (ok) return g;
  }
  return -1;
}

void print_mat(const vector<vector<i64>> &M, const int n, const string &name) {
  cout << name << " =\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << M[i][j] << (j + 1 < n ? ' ' : '\n');
    }
  }
  cout << "\n";
}

vector<i64> matvec(const vector<vector<i64>> &M, const vector<i64> &x, i64 p, int n) {
  vector<i64> y(n);
  for (int i = 0; i < n; ++i) {
    __int128 acc = 0;
    for (int j = 0; j < n; ++j) {
      acc += (__int128)M[i][j] * x[j];
    }
    y[i] = (i64)(acc % p);
  }
  return y;
}

vector<i64> element_wise_mul(const vector<i64> &a, const vector<i64> &b, i64 p, int n) {
  vector<i64> c(n);
  for (int i = 0; i < n; ++i) {
    c[i] = modmul(a[i], b[i], p);
  }
  return c;
}

// in-place Cooley–Tukey algorithm
void ntt(vector<i64>& a, i64 psi, i64 p) {
    int n = a.size();
    i64 omega = modmul(psi, psi, p);           // omega = psi^2 (primitive n-th root of unity mod p)
    // Input twist: multiply each element by psi^j
    i64 psi_power = 1;
    for (int j = 0; j < n; ++j) {
        a[j] = modmul(a[j], psi_power, p);
        psi_power = modmul(psi_power, psi, p);
    }
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    // Cooley–Tukey NTT iterations (butterfly merges)
    for (int len = 1; len < n; len <<= 1) {
        i64 root_step = modpow(omega, n / (2 * len), p);
        for (int i = 0; i < n; i += 2 * len) {
            i64 w = 1;
            for (int j = 0; j < len; ++j) {
                i64 u = a[i + j];
                i64 v = modmul(a[i + j + len], w, p);
                i64 sum = u + v;
                if (sum >= p) sum -= p;
                i64 diff = u - v;
                if (diff < 0) diff += p;
                a[i + j] = sum;
                a[i + j + len] = diff;
                w = modmul(w, root_step, p);
            }
        }
    }
}

// Efficient inverse NTT (inverse transform) - in-place
void intt(vector<i64>& a, i64 psi, i64 p) {
    int n = a.size();
    i64 psi_inv = modinv(psi, p);              // psi_inv = psi^{-1} mod p
    i64 omega_inv = modmul(psi_inv, psi_inv, p); // omega_inv = (psi^2)^{-1} (inverse n-th root)
    // Bit-reversal permutation (same procedure as forward)
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    // Cooley–Tukey inverse NTT iterations
    for (int len = 1; len < n; len <<= 1) {
        i64 root_step = modpow(omega_inv, n / (2 * len), p);
        for (int i = 0; i < n; i += 2 * len) {
            i64 w = 1;
            for (int j = 0; j < len; ++j) {
                i64 u = a[i + j];
                i64 v = modmul(a[i + j + len], w, p);
                i64 sum = u + v;
                if (sum >= p) sum -= p;
                i64 diff = u - v;
                if (diff < 0) diff += p;
                a[i + j] = sum;
                a[i + j + len] = diff;
                w = modmul(w, root_step, p);
            }
        }
    }
    // Divide by n and apply output twist (multiply by psi^{-j})
    i64 ninv = modinv(n % p, p);  // modular inverse of n
    i64 psi_inv_power = 1;
    for (int j = 0; j < n; ++j) {
        a[j] = modmul(a[j], ninv, p);
        a[j] = modmul(a[j], psi_inv_power, p);
        psi_inv_power = modmul(psi_inv_power, psi_inv, p);
    }
}

int main() {
  i64 p = 7681;
  int n = 4;
  vector<i64> v = {1, 2, 3, 4};
  vector<i64> w = {5, 6, 7, 8};

  i64 g = primitive_root(p);
  if (g < 0) {
    cerr << "Failed to find primitive root.\n";
    return 1;
  }

  i64 psi = modpow(g, (p - 1) / (2 * n), p);  // primitive 2n-th root of unity mod p
  i64 ninv = modinv(n % p, p);
  i64 psi_inv = modinv(psi, p);

  cout << "p=" << p << " n=" << n << "\n";
  cout << "primitive root g=" << g << "\n";
  cout << "psi=" << psi << "  (order 2n)\n";
  cout << "psi_inv=" << psi_inv << "\n";
  cout << "ninv=" << ninv << "\n\n";
  cout << "omega=psi^2=" << modmul(psi, psi, p) << "  (order n)\n";

  // Build the psi-twisted NTT matrix F and its inverse Finv
  vector<vector<i64>> F(n, vector<i64>(n));
  vector<vector<i64>> Finv(n, vector<i64>(n));
  const int modOrder = 2 * n;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int e = ((2 * i) % modOrder * j + j) % modOrder;  // exponent = 2*i*j + j
      F[i][j] = modpow(psi, e, p);
      Finv[j][i] = modmul(ninv, modpow(psi_inv, e, p), p);
    }
  }

  print_mat(F, n, "Forward psi-matrix  F[i][j] = psi^{2*i*j + j}");
  print_mat(Finv, n, "Inverse  psi-matrix Finv[i][j] = (1/n)*psi^{-(2*i*j + j)}");

  // NTT via matrix multiplication (verification)
  auto v_ntt = matvec(F, v, p, n);
  auto w_ntt = matvec(F, w, p, n);
  auto res = element_wise_mul(v_ntt, w_ntt, p, n);
  auto back = matvec(Finv, res, p, n);

  auto to_signed = [&](i64 x) {
    return (x >= p/2) ? x - p : x;
  };

  cout << "v ="; for (auto x : v) cout << ' ' << x; cout << '\n';
  cout << "w ="; for (auto x : w) cout << ' ' << x; cout << '\n';
  cout << "NTT(v) ="; for (auto x : v_ntt) cout << ' ' << x; cout << '\n';
  cout << "NTT(w) ="; for (auto x : w_ntt) cout << ' ' << x; cout << '\n';
  cout << "NTT(v)*NTT(w) ="; for (auto x : res) cout << ' ' << x; cout << '\n';
  cout << "Inverse NTT ="; for (auto x : back) cout << ' ' << x; cout << '\n';
  cout << "Inverse NTT (signed) ="; for (auto x : back) cout << ' ' << to_signed(x); cout << '\n';

  // NTT via efficient algorithm (using the new ntt() and intt() functions)
  vector<i64> v2 = v;
  vector<i64> w2 = w;
  ntt(v2, psi, p);
  ntt(w2, psi, p);
  auto res2 = element_wise_mul(v2, w2, p, n);
  intt(res2, psi, p);

  cout << "\nUsing efficient NTT algorithms:\n";
  cout << "NTT2(v) ="; for (auto x : v2) cout << ' ' << x; cout << '\n';
  cout << "NTT2(w) ="; for (auto x : w2) cout << ' ' << x; cout << '\n';
  cout << "NTT2(v)*NTT2(w) ="; 
  for (auto x : element_wise_mul(v2, w2, p, n)) cout << ' ' << x; cout << '\n';
  cout << "Inverse NTT2 ="; for (auto x : res2) cout << ' ' << x; cout << '\n';
  cout << "Inverse NTT2 (signed) ="; 
  for (auto x : res2) cout << ' ' << to_signed(x); cout << '\n';

  return 0;
}
