##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Total dispatches (A and B alternating) and host timeout (ms) per wait.
N          ?= 32
TIMEOUT_MS ?= 5000

all: design_a.elf design_b.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

# Every AIE4 configuration section starts with the reset of the partition,
# including the stream switch clear. Each design has its own device name (stream_a,
# stream_b), so its kernel is stream_<x>:sequence and the two ELFs can share a
# hardware context.
design_%.elf: ${srcdir}/design_%.mlir
	aiecc -v --tmpdir=project_$* --device-name=stream_$* \
		--get-full-elf --full-elf-name=$@ $<

run: all
	for m in seq runlist startall; do \
		./test.exe --mode $$m --n $(N) --timeout-ms $(TIMEOUT_MS) || exit 1; \
	done

clean:
	rm -rf project_a project_b \
		design_a.elf design_b.elf \
		test.exe
