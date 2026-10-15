##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: memtile_recv.elf shim_upper_recv.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -Wall -Wextra -lstdc++ -I/opt/xilinx/xrt/include \
		-L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

# The designs have no aie.core, so no core compiler is involved.
%.elf: ${srcdir}/%.mlir
	aiecc --tmpdir=project_$* --get-full-elf --full-elf-name=$@ $<

run: all
	./test.exe memtile_recv.elf 256
	./test.exe shim_upper_recv.elf 264

clean:
	rm -rf project_* *.elf test.exe
