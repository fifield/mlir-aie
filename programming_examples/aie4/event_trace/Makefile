##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Accept either AIETOOLS_DIR or AIETOOLS from the environment.
ifeq ($(strip $(AIETOOLS_DIR)),)
AIETOOLS_DIR := $(AIETOOLS)
endif

TRACE_PY := $(srcdir)/../../../python/utils/trace

trace_size = 8192

all: event_trace.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

project/scale.o: ${srcdir}/vector_scalar_mul.cc
	mkdir -p project
	cd project && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include -c $(abspath $<) -o scale.o

event_trace.elf: ${srcdir}/aie_trace.mlir project/scale.o
	aiecc -v --dump-intermediates --tmpdir=project --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

# Base design without trace — for verifying data path independently
base.elf: ${srcdir}/base.mlir project/scale.o
	aiecc -v --dump-intermediates --tmpdir=project_base --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

run: test.exe event_trace.elf
	rm -f trace.bin trace.json trace_timeline.png
	./test.exe event_trace.elf
	test -s trace.bin
	@echo ""
	@echo "Parsing trace output..."
	$(TRACE_PY)/parse.py --input ./trace.bin --mlir project/input_physical.mlir --output trace.json
	test -s trace.json
	$(TRACE_PY)/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "AIE4 Event Trace"
	test -s trace_timeline.png
	@echo "Trace visualization saved to trace_timeline.png"

# Base design without trace — for verifying the data path independently
run_base: test.exe base.elf
	./test.exe base.elf

clean_trace:
	rm -rf trace.bin trace.json trace_timeline.png

clean: clean_trace
	rm -rf project project_base *.elf test.exe

.PHONY: all run run_base clean clean_trace
