##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
#
# Makefile for MLIR auto-trace example
#
# Uses the aie_events_proposal build for all tooling (aie-opt, aiecc.py)
# since it includes the trace lowering passes.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Use aie_events_proposal for makefile-common and all build tools
EVENTS_PROPOSAL_DIR ?= /work/acdc/aie_events_proposal
EVENTS_PROPOSAL_BUILD ?= /work/acdc/build/aie_events_proposal

include ${EVENTS_PROPOSAL_DIR}/programming_examples/makefile-common

# Override tools to use events_proposal build
AIE_OPT  := ${EVENTS_PROPOSAL_BUILD}/bin/aie-opt
AIECC    := ${EVENTS_PROPOSAL_BUILD}/bin/aiecc.py

devicename ?= npu
targetname = mlir_trace_example

# Source files shared with the sibling pctrace example
KERNEL_SRC ?= ${srcdir}/vector_scalar_mul.cc
HOST_SRC   ?= ${srcdir}/test.cpp
CMAKE_SRC  ?= ${srcdir}/CMakeLists.txt
VIS_SCRIPT ?= ${srcdir}/visualize_trace.py

# MLIR source with declarative aie.trace syntax
MLIR_SRC   ?= ${srcdir}/aie_new_trace_flows.mlir

in1_size = 16384 # in bytes
in2_size = 4     # in bytes, should always be 4 (1x int32)
out_size = 16384 # in bytes, should always be equal to in1_size
trace_size = 1048576

all: build/final_new.xclbin

build/insts_new.bin: build/final_new.xclbin

# Compile the AIE kernel
build/scale.o: ${KERNEL_SRC}
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}

# Step 1: Lower declarative aie.trace ops to standard NPU ops
#   --aie-insert-trace-flows: auto-generates packet_flows, writebd, address_patch,
#     and channel config from the aie.trace blocks (the whole point of this test)
build/aie_lowered.mlir: ${MLIR_SRC}
	mkdir -p ${@D}
	${AIE_OPT} --aie-insert-trace-flows --aie-trace-to-config \
		--aie-trace-pack-reg-writes --aie-inline-trace-config \
		$< -o $@

# Step 2: Build xclbin from the lowered MLIR
build/final_new.xclbin: build/aie_lowered.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && ${AIECC} --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_new.bin \
			$(realpath $<)

# Build the host executable
${targetname}.exe: ${HOST_SRC}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake $(dir ${CMAKE_SRC}) -DTARGET_NAME=${targetname} \
		-DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

# Run the design (no trace processing)
run: ${targetname}.exe build/final_new.xclbin build/insts_new.bin
	${powershell} ./$< -x build/final_new.xclbin -i build/insts_new.bin -k MLIR_AIE

# Run with trace capture and analysis
trace: ${targetname}.exe build/final_new.xclbin build/insts_new.bin
	@echo "Running with declarative auto-trace..."
	${powershell} ./$< -x build/final_new.xclbin -i build/insts_new.bin -k MLIR_AIE -t ${trace_size}
	${EVENTS_PROPOSAL_DIR}/python/utils/trace/parse.py --input trace.txt \
		--mlir build/aie_lowered.mlir.prj/input_physical_with_elfs.mlir \
		--output trace.json
	${EVENTS_PROPOSAL_DIR}/python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${VIS_SCRIPT} -i trace.json -o trace_timeline.png -t "Auto Trace"
	@echo "Trace visualization saved to trace_timeline.png"

# Run the Python test with trace
run_trace_py: build/final_new.xclbin build/insts_new.bin
	@echo "Running Python test with auto-trace..."
	python3 ${srcdir}/../mlir_pctrace_example/test.py \
		--xclbin build/final_new.xclbin --instr build/insts_new.bin \
		--kernel MLIR_AIE --verbosity 1 --trace-sz ${trace_size} --trace-file trace.txt
	${EVENTS_PROPOSAL_DIR}/python/utils/trace/parse.py --input trace.txt \
		--mlir build/aie_lowered.mlir.prj/input_physical_with_elfs.mlir \
		--output trace.json
	${EVENTS_PROPOSAL_DIR}/python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${VIS_SCRIPT} -i trace.json -o trace_timeline.png -t "Auto Trace (Python)"
	@echo "Trace visualization saved to trace_timeline.png"

# Visualize existing trace data
visualize: trace.json
	python3 ${VIS_SCRIPT} -i trace.json -o trace_timeline.png -t "Auto Trace"
	@echo "Trace visualization saved to trace_timeline.png"

clean_trace:
	rm -rf tmpTrace trace.txt trace_timeline.png trace.json trace_new.json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all run trace run_trace_py visualize clean clean_trace
