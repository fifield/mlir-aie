#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

targetname = conv_bf16
mlirFileName = aie

# Design parameters - can be overridden
HEIGHT ?= 8
WIDTH ?= 8
IN_CHANNELS ?= 8
OUT_CHANNELS ?= 8
KERNEL_SIZE ?= 3
STRIDE ?= 1
PADDING ?= 1

# Tiling parameters (for large images)
TILE_H ?= 32
TILE_W ?= 32
OUT_CHAN_BLOCK ?= 8

# Device (only npu2 supported for now)
devicename = npu2

# Kernel compilation
VPATH := ${srcdir}

all: build/conv_bf16.o build/final.xclbin

build/conv_bf16.o: conv_bf16.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c ${srcdir}/$< -o ${@F}

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< $(devicename) $(HEIGHT) $(WIDTH) $(IN_CHANNELS) $(OUT_CHANNELS) $(KERNEL_SIZE) $(STRIDE) $(PADDING) > $@

build/final.xclbin: build/aie.mlir build/conv_bf16.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

# Tiled version for large images (640x640)
build/aie_tiled.mlir: ${srcdir}/aie2_tiled.py
	mkdir -p ${@D}
	python3 $< $(devicename) $(HEIGHT) $(WIDTH) $(IN_CHANNELS) $(OUT_CHANNELS) $(TILE_H) $(TILE_W) $(OUT_CHAN_BLOCK) > $@

build/final_tiled.xclbin: build/aie_tiled.mlir build/conv_bf16.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts_tiled.bin $(<:%=../%)

# Run test (CPU reference only for now)
test: 
	${powershell} python3 ${srcdir}/test.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -k $(KERNEL_SIZE) -s $(STRIDE) -p $(PADDING)

# Test model dimensions (Conv0 and Conv1)
test-model-dims:
	${powershell} python3 ${srcdir}/test_model_dimensions.py --config all

test-conv0:
	${powershell} python3 ${srcdir}/test_model_dimensions.py --config conv0

test-conv1:
	${powershell} python3 ${srcdir}/test_model_dimensions.py --config conv1

# Run on hardware (when xclbin is built)
run: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -k $(KERNEL_SIZE) -s $(STRIDE) -p $(PADDING) -x build/final.xclbin -i build/insts.bin

# Test tiled version (CPU reference)
test-tiled:
	${powershell} python3 ${srcdir}/test_tiled.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -th $(TILE_H) -tw $(TILE_W) -ob $(OUT_CHAN_BLOCK)

# Build tiled xclbin
build-tiled: build/final_tiled.xclbin

# Run tiled version on hardware
run-tiled: build/final_tiled.xclbin
	${powershell} python3 ${srcdir}/test_tiled.py -ht $(HEIGHT) -wd $(WIDTH) -ic $(IN_CHANNELS) -oc $(OUT_CHANNELS) -th $(TILE_H) -tw $(TILE_W) -ob $(OUT_CHAN_BLOCK) -x build/final_tiled.xclbin -i build/insts_tiled.bin

# Quick test: Conv0 640×640 (3→32)
test-conv0-640:
	make clean
	make test-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=3 OUT_CHANNELS=32 TILE_H=32 TILE_W=32 OUT_CHAN_BLOCK=8

# Quick test: Conv1 640×640 (32→64) - will need multi-core implementation
test-conv1-640:
	make clean
	make test-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=32 OUT_CHANNELS=64 TILE_H=20 TILE_W=20 OUT_CHAN_BLOCK=16

# Build and run Conv0 640×640 on hardware
run-tiled-conv0-640:
	make clean
	make build-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=3 OUT_CHANNELS=32 TILE_H=32 TILE_W=32 OUT_CHAN_BLOCK=8
	make run-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=3 OUT_CHANNELS=32 TILE_H=32 TILE_W=32 OUT_CHAN_BLOCK=8

# Build and run Conv1 640×640 on hardware
run-tiled-conv1-640:
	make clean
	make build-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=32 OUT_CHANNELS=64 TILE_H=20 TILE_W=20 OUT_CHAN_BLOCK=16
	make run-tiled HEIGHT=640 WIDTH=640 IN_CHANNELS=32 OUT_CHANNELS=64 TILE_H=20 TILE_W=20 OUT_CHAN_BLOCK=16

clean:
	rm -rf build *.elf* *.lst *.bif aie.mlir.prj aie_tiled.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin insts_tiled.bin *.log aie_partition.json *.bin BOOT.BIN _x

.PHONY: all test run clean test-model-dims test-conv0 test-conv1 \
        test-tiled build-tiled run-tiled test-conv0-640 test-conv1-640 \
        run-tiled-conv0-640 run-tiled-conv1-640
