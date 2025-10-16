#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

targetname = repncsp_bf16
mlirFileName = aie

# Design parameters - can be overridden
HEIGHT ?= 8
WIDTH ?= 8
IN_CHANNELS ?= 16
OUT_CHANNELS ?= 16
CSP_EXPAND ?= 0.5

# Device (only npu2 supported for now)
devicename = npu2

# Kernel compilation
VPATH := ${srcdir}

all: build/repncsp_bf16.o build/final.xclbin

build/repncsp_bf16.o: repncsp_bf16.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c ${srcdir}/$< -o ${@F}

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< $(devicename) $(HEIGHT) $(WIDTH) $(IN_CHANNELS) $(OUT_CHANNELS) $(CSP_EXPAND) > $@

build/final.xclbin: build/aie.mlir build/repncsp_bf16.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

# Run test (CPU reference only for now)
test: 
	${powershell} python3 ${srcdir}/test.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -e $(CSP_EXPAND)

# Run on hardware (when xclbin is built)
run: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -e $(CSP_EXPAND) -x build/final.xclbin -i build/insts.bin

clean:
	rm -rf build *.elf* *.lst *.bif aie.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin *.log aie_partition.json *.bin BOOT.BIN _x

.PHONY: all test run clean
