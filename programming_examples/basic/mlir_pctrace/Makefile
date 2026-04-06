##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
#
# MLIR Event-PC-Trace example: vector-scalar-multiply with event-PC trace
# mode capture using the declarative trace API (aie_new_trace.mlir).
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= npu1
targetname = mlir_pctrace

in1_size = 16384 # in bytes
in2_size = 4 # in bytes, should always be 4 (1x int32)
out_size = 16384 # in bytes, should always be equal to in1_size
trace_size = 1048576

all: build/final.xclbin

build/insts.bin: build/final.xclbin

# Compile the AIE kernel
build/scale.o: ${srcdir}/vector_scalar_mul.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu1)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

# Preprocess the MLIR: substitute NPUDEVICE with the target device string
build/aie_new_trace.mlir: ${srcdir}/aie_new_trace.mlir
	mkdir -p ${@D}
	sed 's/NPUDEVICE/${devicename}_1col/g' $< > $@

# Build xclbin from the declarative event-PC trace MLIR
build/final.xclbin: build/aie_new_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			--no-xchesscc --no-xbridge --dump-intermediates \
			--aie-generate-npu-insts --npu-insts-name=insts.bin aie_new_trace.mlir

# Build the host executable
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} \
		-DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

# C++ host run with event-PC trace
run_trace: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	python3 ${srcdir}/../../../python/utils/trace/parse_trace.py \
		--input trace.txt --output trace.json --mode eventpc

# Python host run with event-PC trace
run_trace_py: build/final.xclbin build/insts.bin
	python3 ${srcdir}/test.py --xclbin build/final.xclbin --instr build/insts.bin \
		--kernel MLIR_AIE --verbosity 1 --trace-sz ${trace_size} --trace-file trace.txt
	python3 ${srcdir}/../../../python/utils/trace/parse_trace.py \
		--input trace.txt --output trace.json --mode eventpc

clean_trace:
	rm -rf tmpTrace trace.txt trace*.json trace_timeline.png

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all run_trace run_trace_py clean clean_trace
