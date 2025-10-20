# NTT/PIM

This directory contains a Cooley–Tukey NTT implementation that can run either purely on the CPU or offload stages to UPMEM DPUs. The `main.cc` enumerates several `(n, p)` pairs (small and 60‑bit primes) to validate round-trip correctness, compare against a dense O(n²) reference when feasible, and time a forward NTT plus the full polynomial multiply.

## Build

- **ALL**: `make` produces all binaries
- **CPU only** (default): `make cpu` produces `ntt_cpu`.
- **CPU + PIM** (requires UPMEM SDK): `make pim` builds `ntt_pim` and the `dpu.bin` kernel. 
Make sure `dpu-upmem-dpurte-clang` and `dpu-pkg-config` are on PATH.

## Run

```bash
./ntt_cpu # runs cpu only variant
# or with PIM enabled
./ntt_pim # runs cpu and pim variants
```

For each `(n, p)` pair you’ll see:

- Round-trip checks (`[PASS]/[FAIL]`).
- Optional dense-matrix verification for small `n`.
- Timings for a single forward NTT and the polynomial multiply.
- (PIM build) Host vs. DPU copy/compute breakdowns.

A failure indicates the chosen modulus does not admit the required 2n‑th root of unity or there’s a regression in the CPU/PIM kernels. Timings allow quick comparison between small and 60‑bit primes.***
