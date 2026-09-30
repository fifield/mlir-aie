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
TRACE_SIZE  ?= 32768

# Design geometry (rebuild after changing -- `make clean` first):
#   N       = total output width; padded up to a multiple of 64 (COLS,
#             below, is derived from it). npu3 caps at N=192 (3 columns).
#   ROWS    = active cascade rows per column (1..4); ROWS=1 skips the
#             cascade entirely (uses the no-cascade "solo" kernel path)
#   STREAMS = per-column shim weight-stream MM2S channels (2 or 4)
#   K       = contraction length (multiple of ROWS*64; subject to L1 budget)
#   KCHUNK  = K-tiles per chunk (must divide K_TILES_PER_ROW = K/ROWS/64);
#             0 = whole row (no chunking, default -- see README's
#             "DMA batching and K-chunking" section)
#   BUF_DEPTH = shim->memtile weight stream FIFO depth (lives in memtile SRAM)
#   AW_DEPTH  = memtile->core AW_lo FIFO depth (lives in core L1, per chunk)
#   WHI_DEPTH = memtile->core W_hi  FIFO depth (lives in core L1, per chunk)
N           ?= 192
ROWS        ?= 4
STREAMS     ?= 2
K           ?= 4096
KCHUNK      ?= 0
BUF_DEPTH   ?= 2
AW_DEPTH    ?= 1
WHI_DEPTH   ?= 1

# Derive COLS from N for both host and device. Override command-line COLS
# to prevent inconsistent geometry.
override COLS := $(shell echo $$(( ($(N) + 63) / 64 )))

DEPTH_FLAGS := --bufdepth $(BUF_DEPTH) --awdepth $(AW_DEPTH) --whidepth $(WHI_DEPTH)
KERNEL_DEFS := -DK_TOTAL=$(K) -DN_ROWS=$(ROWS) -DK_CHUNK_TILES_RAW=$(KCHUNK)

all: gemv_4096_192_cascade.elf gemv_4096_192_cascade_bench.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -DN_COLS=$(COLS) -DN_VALID=$(N) $(KERNEL_DEFS) \
		-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p project
	cd project && xchesscc_wrapper aie4 $(KERNEL_DEFS) -I $(AIETOOLS_DIR)/include -c $(abspath $<) -o kernel.o

gemv_placed.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -n $(N) -r $(ROWS) -s $(STREAMS) -k $(K) -i $(ITERS) --kchunk $(KCHUNK) $(DEPTH_FLAGS) > $@

gemv_placed_bench.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -n $(N) -r $(ROWS) -s $(STREAMS) -k $(K) -i $(BENCH_ITERS) --kchunk $(KCHUNK) $(DEPTH_FLAGS) > $@

gemv_placed_trace.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -n $(N) -r $(ROWS) -s $(STREAMS) -k $(K) -i 1 --kchunk $(KCHUNK) $(DEPTH_FLAGS) --trace --trace_size $(TRACE_SIZE) > $@

# Compile --vector compute IR through the AIE4 vector pipeline and Peano.
gemv_vector.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -n $(N) -r $(ROWS) -s $(STREAMS) -k $(K) -i $(ITERS) --kchunk $(KCHUNK) $(DEPTH_FLAGS) --vector > $@

gemv_vector_bench.mlir: ${srcdir}/gemv_placed.py
	python $< -d npu3 -n $(N) -r $(ROWS) -s $(STREAMS) -k $(K) -i $(BENCH_ITERS) --kchunk $(KCHUNK) $(DEPTH_FLAGS) --vector > $@

gemv_4096_192_cascade_vector.elf: gemv_vector.mlir
	aiecc -v --tmpdir=project_vector --enable-aie4-vectorization \
		--get-full-elf --full-elf-name=$@ $<

gemv_4096_192_cascade_vector_bench.elf: gemv_vector_bench.mlir
	aiecc -v --tmpdir=project_vector_bench --enable-aie4-vectorization \
		--get-full-elf --full-elf-name=$@ $<

vector: gemv_4096_192_cascade_vector.elf gemv_4096_192_cascade_vector_bench.elf test.exe

gemv_4096_192_cascade.elf: gemv_placed.mlir project/kernel.o
	aiecc -v --tmpdir=project --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

gemv_4096_192_cascade_bench.elf: gemv_placed_bench.mlir project/kernel.o
	aiecc -v --tmpdir=project_bench --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

gemv_4096_192_cascade_trace.elf: gemv_placed_trace.mlir project/kernel.o
	aiecc -v --dump-intermediates --tmpdir=project_trace --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

# Run correctness + per-call benchmark (single iter per call).
run: test.exe gemv_4096_192_cascade.elf
	./test.exe gemv_4096_192_cascade.elf 1

run_vector: test.exe gemv_4096_192_cascade_vector.elf
	./test.exe gemv_4096_192_cascade_vector.elf 1

run_vector_full_range: test.exe gemv_4096_192_cascade_vector.elf
	./test.exe --full-range gemv_4096_192_cascade_vector.elf 1

run_full_range: test.exe gemv_4096_192_cascade.elf
	./test.exe --full-range gemv_4096_192_cascade.elf 1

run_vector_bench: test.exe gemv_4096_192_cascade_vector_bench.elf
	./test.exe gemv_4096_192_cascade_vector_bench.elf $(BENCH_ITERS)

# Run batched benchmark to amortize XRT overhead.
run_bench: test.exe gemv_4096_192_cascade_bench.elf
	./test.exe gemv_4096_192_cascade_bench.elf $(BENCH_ITERS)

# Run with event trace, parse, and render PNG timeline.
run_trace: test.exe gemv_4096_192_cascade_trace.elf
	rm -f trace.bin trace.json trace_timeline.png
	./test.exe gemv_4096_192_cascade_trace.elf 1 $(TRACE_SIZE)
	test -s trace.bin
	@echo ""
	@echo "Parsing trace output..."
	$(srcdir)/../../../python/utils/trace/parse.py \
		--input trace.bin \
		--mlir project_trace/input_physical.mlir \
		--output trace.json
	test -s trace.json
	$(srcdir)/../../../python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Rendering PNG timeline..."
	python3 $(srcdir)/../event_trace/visualize_trace.py \
		-i trace.json \
		-o trace_timeline.png \
		-t "GEMV 4096x192 cascade trace"
	test -s trace_timeline.png
	@echo "Trace PNG: $(srcdir)/trace_timeline.png"

clean_trace:
	rm -f trace.bin trace.json trace_timeline.png

clean: clean_trace
	rm -rf project project_bench project_trace \
		gemv_placed.mlir gemv_placed_bench.mlir gemv_placed_trace.mlir \
		gemv_4096_192_cascade.elf gemv_4096_192_cascade_bench.elf \
		gemv_4096_192_cascade_trace.elf test.exe \
		project_vector project_vector_bench gemv_vector.mlir \
		gemv_vector_bench.mlir gemv_4096_192_cascade_vector.elf \
		gemv_4096_192_cascade_vector_bench.elf

.PHONY: all vector run run_bench run_full_range run_vector run_vector_bench \
	run_vector_full_range run_trace clean clean_trace
