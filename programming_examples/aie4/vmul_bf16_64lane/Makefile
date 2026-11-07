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
include ${srcdir}/../../makefile-common

targetname := vmul_bf16_64lane

# Kernel compiler: chess (default, kernel.cc via xchesscc) or peano
# (kernel.cc via Peano clang++; aiecc then compiles/links the core with Peano).
KERNEL_CC ?= chess
ifeq ($(filter $(KERNEL_CC),chess peano),)
$(error KERNEL_CC must be chess or peano)
endif

all: $(targetname).elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil -lm $< -o $@

ifeq ($(KERNEL_CC),peano)
project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p project
	cd project && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP4_FLAGS) -c $(abspath $<) -o kernel.o
core_flags := --peano $(PEANO_INSTALL_DIR)
else
project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p project
	cd project && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include -c $(abspath $<) -o kernel.o
core_flags := --xchesscc --xbridge
endif

$(targetname)_placed.mlir: ${srcdir}/$(targetname)_placed.py
	python $< -d npu3 > $@

$(targetname).elf: $(targetname)_placed.mlir project/kernel.o
	aiecc -v --tmpdir=project $(core_flags) \
		--get-full-elf --full-elf-name=$@ $<

run: test.exe $(targetname).elf
	./test.exe $(targetname).elf

clean:
	rm -rf project $(targetname)_placed.mlir $(targetname).elf test.exe

.PHONY: all run clean
