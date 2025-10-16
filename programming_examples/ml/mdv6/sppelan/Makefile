# sppelan/Makefile
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

# Default test parameters
HEIGHT ?= 8
WIDTH ?= 8
IN_CHANNELS ?= 16
OUT_CHANNELS ?= 16

# Derived parameters
NECK_CHANNELS := $(shell echo $$(($(IN_CHANNELS) / 2)))

targetname = sppelan_bf16

VPATH := ${srcdir}

.PHONY: all
all: build/${targetname}.o build/final.xclbin

build/${targetname}.o: ${targetname}.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c ${srcdir}/$< -o ${@F}

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< npu2 ${HEIGHT} ${WIDTH} ${IN_CHANNELS} ${OUT_CHANNELS} > $@

build/final.xclbin: build/aie.mlir build/${targetname}.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${sourcedir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

.PHONY: test
test:
	python3 test.py --height ${HEIGHT} --width ${WIDTH} \
		--in-channels ${IN_CHANNELS} --out-channels ${OUT_CHANNELS}

.PHONY: run
run: build/final.xclbin
	python3 test.py --height ${HEIGHT} --width ${WIDTH} \
		--in-channels ${IN_CHANNELS} --out-channels ${OUT_CHANNELS} --use-aie

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
