.DEFAULT_GOAL := all

DPU_CC ?= dpu-upmem-dpurte-clang
DPU_PKG ?= dpu-pkg-config
CXX ?= g++

DPU_SRC := dpu.c
NTT_SRCS := ntt.cc pim_runtime.cc main.cc

DPU_BIN := dpu.bin
NTT_PIM_BIN := ntt_pim
NTT_CPU_BIN := ntt_cpu

CXXFLAGS := -std=c++17 -g
PIM_CFLAGS := $(shell $(DPU_PKG) --cflags dpu 2>/dev/null || echo "")
PIM_LIBS := $(shell $(DPU_PKG) --libs dpu 2>/dev/null || echo "")
PIM_DEFS := -DNTT_USE_PIM
DPU_FLAGS := -O2 -DNR_TASKLETS=16

.PHONY: all pim cpu dpu clean help

help:
	@echo "Makefile targets:"
	@echo "  make         (default) build both PIM and CPU binaries"
	@echo "  make pim     build with PIM flags (also builds DPU binary)"
	@echo "  make cpu     build CPU-only binary (ntt_cpu)"
	@echo "  make dpu     build the DPU binary using $(DPU_CC)"
	@echo "  make clean   remove built artifacts"

all: pim cpu

pim: dpu $(NTT_PIM_BIN)

dpu: $(DPU_BIN)

$(DPU_BIN): $(DPU_SRC)
	@echo "Compiling DPU binary: $(DPU_CC) $(DPU_FLAGS) -o $@ $<"
	$(DPU_CC) $(DPU_FLAGS) -o $@ $<
	@echo "DPU binary: $@"

PIM_AVAILABLE := $(strip $(PIM_CFLAGS) $(PIM_LIBS))

ifneq ($(PIM_AVAILABLE),)
$(NTT_PIM_BIN): $(NTT_SRCS)
	@echo "Compiling PIM-enabled ntt with cflags: $(PIM_CFLAGS) libs: $(PIM_LIBS)"
	$(CXX) $(CXXFLAGS) $(PIM_CFLAGS) $(PIM_DEFS) $(NTT_SRCS) -o $(NTT_PIM_BIN) $(PIM_LIBS)
else
$(NTT_PIM_BIN): $(NTT_SRCS)
	@echo "Warning: PIM pkg-config did not return flags; compiling with -DNTT_USE_PIM only"
	$(CXX) $(CXXFLAGS) $(PIM_DEFS) $(NTT_SRCS) -o $(NTT_PIM_BIN)
endif

cpu: $(NTT_CPU_BIN)

$(NTT_CPU_BIN): $(NTT_SRCS)
	@echo "Compiling CPU-only ntt: $(CXX) $(CXXFLAGS) $^ -o $@"
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(DPU_BIN) $(NTT_PIM_BIN) $(NTT_CPU_BIN)
