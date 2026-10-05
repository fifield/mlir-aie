# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

M ?= 2048
N ?= 3072
K ?= 1024
OUT ?= i8
SHIFT ?= 12
ITERS ?= 1
COLS ?= 3
ABCAST ?= 1
# Overwrite tile output with cycle timestamps for debugging.
TIMING ?= 0
# Two W buffers overlap loading and replay, subject to BD and memory limits.
WBUF ?= 2
# A DMA channels per core; 2 loads the slice halves in parallel.
ASPLIT ?= 2
# C path: 1 reassembles wide runs; 0 joins row pairs onto two shim channels.
C_REASM ?= 1
C_RUN ?= 1024
# Trace buffer bytes for column 0's cores, memtile and shim; 0 disables tracing.
TRACE_SIZE ?= 0
trace_on := $(if $(filter-out 0,$(TRACE_SIZE)),1,0)
# C tiles per core buffer; int32 at K=1024 fits CT=1 in L1.
CT ?= $(if $(filter i8,$(OUT)),4,1)

# Cache device artifacts by configuration; kernel objects use only kernel knobs.
cfg := build_$(M)x$(N)x$(K)_$(OUT)_s$(SHIFT)_it$(ITERS)_c$(COLS)_ab$(ABCAST)_ct$(CT)_wb$(WBUF)_as$(ASPLIT)_cr$(C_REASM)$(if $(filter-out 1024,$(C_RUN)),_run$(C_RUN))$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr$(TRACE_SIZE))
kcfg := kernel_K$(K)_$(OUT)_s$(SHIFT)_ct$(CT)_as$(ASPLIT)$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr)
out_i8 := $(if $(filter i8,$(OUT)),1,0)
out_bits := $(if $(filter i8,$(OUT)),8,32)

# A failed generator run must not leave an empty .mlir that looks up to date.
.DELETE_ON_ERROR:

all: $(cfg)/gemm_int8.elf test.exe

test.exe: ${srcdir}/test.cpp
	clang++ -std=c++17 -O3 -I/opt/xilinx/xrt/include \
		-L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil -lpthread $< -o $@

$(kcfg)/kernel.o: ${srcdir}/kernel.cc
	mkdir -p $(kcfg)
	cd $(kcfg) && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include \
		-DK_DIM=$(K) -DOUT_I8=$(out_i8) -DOUT_SHIFT=$(SHIFT) -DCT=$(CT) -DTIMING=$(TIMING) -DTRACE=$(trace_on) -DA_SPLIT=$(ASPLIT) \
		-c $(abspath $<) -o kernel.o

$(cfg)/gemm_int8.mlir: ${srcdir}/gemm_int8_placed.py
	mkdir -p $(cfg)
	python3 $< -M $(M) -N $(N) -K $(K) --out $(OUT) --iters $(ITERS) --cols $(COLS) --a-bcast $(ABCAST) --ct $(CT) --wbuf $(WBUF) --a-split $(ASPLIT) --c-reasm $(C_REASM) --c-run $(C_RUN) --trace-size $(TRACE_SIZE) > $@

$(cfg)/gemm_int8.elf: $(cfg)/gemm_int8.mlir $(kcfg)/kernel.o
	mkdir -p $(cfg)/project
	cp $(kcfg)/kernel.o $(cfg)/project/kernel.o
	# Lower FIFOs before tracing to avoid counting C FIFOs as trace channels.
	cd $(cfg) && $(if $(filter 1,$(trace_on)),aie-opt --aie-objectFifo-stateful-transform \
		gemm_int8.mlir -o gemm_int8_in.mlir,cp gemm_int8.mlir gemm_int8_in.mlir)
	cd $(cfg) && aiecc -v -j4 --tmpdir=project --xchesscc --xbridge --dump-intermediates \
		--get-full-elf --full-elf-name=gemm_int8.elf gemm_int8_in.mlir \
		> aiecc.log 2>&1 || (grep -i error aiecc.log | head -20; false)

run: all
	./test.exe $(cfg)/gemm_int8.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) $(ITERS) $(COLS) $(if $(filter 1,$(TIMING)),timing) $(if $(filter 1,$(trace_on)),trace=$(TRACE_SIZE))

TRACE_PY := $(srcdir)/../../../python/utils/trace

# Run with tracing, then parse, summarize and render a timeline.
# Example: make run_trace TRACE_SIZE=4194304 ITERS=2
run_trace: all
	@test "$(trace_on)" = 1 || (echo "set TRACE_SIZE=<bytes>"; false)
	cd $(cfg) && ../test.exe gemm_int8.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) \
		$(ITERS) $(COLS) trace=$(TRACE_SIZE)
	cd $(cfg) && $(TRACE_PY)/parse.py --input trace.txt \
		--mlir project/input_physical.mlir --output trace.json
	cd $(cfg) && $(TRACE_PY)/get_trace_summary.py --input trace.json
	cd $(cfg) && python3 $(srcdir)/analyze_trace.py trace.json --clk-ghz 1.2 \
		--timeline trace_timeline.png

clean:
	rm -rf build_* kernel_K* test.exe

.PHONY: all run run_trace clean
