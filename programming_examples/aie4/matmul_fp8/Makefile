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

targetname := matmul_fp8

all: $(targetname).elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil -lm $< -o $@

project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p project
	cd project && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include -c $(abspath $<) -o kernel.o

$(targetname)_placed.mlir: ${srcdir}/$(targetname)_placed.py
	python $< -d npu3 > $@

$(targetname).elf: $(targetname)_placed.mlir project/kernel.o
	aiecc -v --tmpdir=project --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

run: test.exe $(targetname).elf
	./test.exe $(targetname).elf

clean:
	rm -rf project $(targetname)_placed.mlir $(targetname).elf test.exe

.PHONY: all run clean
