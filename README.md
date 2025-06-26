# OpenFHE UPMEM-PIM Acceleration
This repository contains code that intergrate UPMEM-PIM.

## Installation

**Make sure that you have the UPMEM TOOLCHAIN installed !!**

Run:

```bash
mkdir -p build  
cd build  
cmake ..  -DWITH_PIM_HEXL=ON
make -j
```

## Benchmark

To run the benchmark:

```bash
./bin/benchmark/poly-benchmark-8k
```

To run the CPU variant just recompile with `cmake ..  -DWITH_PIM_HEXL=OFF`

## PIM Code

The backend PIM implementation is located in the `third-party/pim-hexl` directory.

## Known Issues

There are several operational tasks needed to complete the backend implementation:
- Memory accesses for the pim vector abstraction needs to be safe.
- Handling PIM parallel threads access
