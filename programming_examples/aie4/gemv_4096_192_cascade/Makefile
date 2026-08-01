##===- Makefile -----------------------------------------------------------===##
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Iteration counts:
#   ITERS=1         -> default elf, used for correctness + per-call latency
#   BENCH_ITERS=100 -> batched elf, used to measure steady-state device perf
#                      by amortizing the XRT host launch overhead
ITERS       ?= 1
BENCH_ITERS ?= 100
TRACE_SIZE  ?= 8192

# Design geometry (rebuild after changing -- `make clean` first):
#   COLS    = number of compute columns (1, 2 or 3); N = COLS * 64
#   STREAMS = per-column shim weight-stream MM2S channels (2 or 4)
#   K       = contraction length (multiple of 256; <= ~7680 for L1 budget)
#   BUF_DEPTH = shim->memtile weight stream FIFO depth (lives in memtile SRAM)
#   AW_DEPTH  = memtile->core AW_lo FIFO depth (lives in core L1)
#   WHI_DEPTH = memtile->core W_hi  FIFO depth (lives in core L1)
COLS        ?= 3
STREAMS     ?= 2
K           ?= 4096
BUF_DEPTH   ?= 2
AW_DEPTH    ?= 1
WHI_DEPTH   ?= 1

DEPTH_FLAGS := --bufdepth $(BUF_DEPTH) --awdepth $(AW_DEPTH) --whidepth $(WHI_DEPTH)

all: gemv_4096_192_cascade.elf gemv_4096_192_cascade_bench.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -DN_COLS=$(COLS) -DK_TOTAL=$(K) \
		-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p project
	cd project && xchesscc_wrapper aie4 -DK_TOTAL=$(K) -I $(AIETOOLS_DIR)/include -c $(abspath $<) -o kernel.o

gemv_placed.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -c $(COLS) -s $(STREAMS) -k $(K) -i $(ITERS) $(DEPTH_FLAGS) > $@

gemv_placed_bench.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -c $(COLS) -s $(STREAMS) -k $(K) -i $(BENCH_ITERS) $(DEPTH_FLAGS) > $@

gemv_placed_trace.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -c $(COLS) -s $(STREAMS) -k $(K) -i 1 $(DEPTH_FLAGS) --trace --trace_size $(TRACE_SIZE) > $@

gemv_4096_192_cascade.elf: gemv_placed.mlir project/kernel.o
	aiecc -v --tmpdir=project --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

gemv_4096_192_cascade_bench.elf: gemv_placed_bench.mlir project/kernel.o
	aiecc -v --tmpdir=project_bench --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

gemv_4096_192_cascade_trace.elf: gemv_placed_trace.mlir project/kernel.o
	aiecc -v --tmpdir=project_trace --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

# Run correctness + per-call benchmark (single iter per call).
run: test.exe gemv_4096_192_cascade.elf
	./test.exe gemv_4096_192_cascade.elf 1

# Run batched benchmark to amortize XRT overhead.
run_bench: test.exe gemv_4096_192_cascade_bench.elf
	./test.exe gemv_4096_192_cascade_bench.elf $(BENCH_ITERS)

# Run with event trace, parse, and render PNG timeline.
run_trace: test.exe gemv_4096_192_cascade_trace.elf
	./test.exe gemv_4096_192_cascade_trace.elf 1 $(TRACE_SIZE)
	@if [ -f trace.bin ]; then \
		echo ""; \
		echo "Parsing trace output..."; \
		$(srcdir)/../../../python/utils/trace/parse.py \
			--input trace.bin \
			--mlir project_trace/input_physical.mlir \
			--output trace.json; \
		if [ -f trace.json ]; then \
			$(srcdir)/../../../python/utils/trace/get_trace_summary.py \
				--input trace.json 2>/dev/null || true; \
			echo ""; \
			echo "Rendering PNG timeline..."; \
			python3 $(srcdir)/../event_trace/visualize_trace.py \
				-i trace.json \
				-o trace_timeline.png \
				-t "GEMV 4096x192 cascade trace" 2>/dev/null || true; \
			if [ -f trace_timeline.png ]; then \
				echo "Trace PNG: $(srcdir)/trace_timeline.png"; \
			fi \
		fi; \
	fi

clean_trace:
	rm -f trace.bin trace.json trace_timeline.png

clean: clean_trace
	rm -rf project project_bench project_trace \
		gemv_placed.mlir gemv_placed_bench.mlir gemv_placed_trace.mlir \
		gemv_4096_192_cascade.elf gemv_4096_192_cascade_bench.elf \
		gemv_4096_192_cascade_trace.elf test.exe

.PHONY: all run run_bench run_trace clean clean_trace
