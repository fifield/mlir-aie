#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: msc_fp32.elf test.exe msc_bf16.elf test_bf16.exe msc_lhs_product.elf test_lhs_product.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

# The flag is what routes the vector arithmetic to the AIE4 instructions
# these tests measure.
msc_fp32.elf: ${srcdir}/msc_fp32.mlir
	aiecc -v --enable-aie4-vectorization --tmpdir=project \
			--get-full-elf --full-elf-name=$@ $<

msc_lhs_product.elf: ${srcdir}/msc_lhs_product.mlir
	aiecc -v --enable-aie4-vectorization --tmpdir=project_lhs \
			--get-full-elf --full-elf-name=$@ $<

msc_bf16.elf: ${srcdir}/msc_bf16.mlir
	aiecc -v --enable-aie4-vectorization --tmpdir=project_bf16 \
			--get-full-elf --full-elf-name=$@ $<

run: all
	./test.exe
	./test_bf16.exe
	./test_lhs_product.exe

clean:
	rm -rf project project_bf16 project_lhs msc_fp32.elf msc_bf16.elf msc_lhs_product.elf test.exe test_bf16.exe test_lhs_product.exe
