# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
KT ?= 8
CALLS ?= 4
# Key artifacts by both knobs to match the host's runtime buffer sizes.
cfg := build_kt$(KT)_calls$(CALLS)
.DELETE_ON_ERROR:
all: $(cfg)/kbench2.elf test.exe
test.exe: ${srcdir}/test.cpp
	clang++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@
$(cfg)/project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p $(cfg)/project
	cd $(cfg)/project && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include -DKT=$(KT) -c $(abspath $<) -o kernel.o
$(cfg)/kbench2.mlir: ${srcdir}/kbench2_placed.py
	mkdir -p $(cfg)
	python3 $< $(KT) $(CALLS) > $@
$(cfg)/kbench2.elf: $(cfg)/kbench2.mlir $(cfg)/project/kernel.o
	cd $(cfg) && aiecc -v --tmpdir=project --xchesscc --xbridge --get-full-elf --full-elf-name=kbench2.elf kbench2.mlir > aiecc.log 2>&1 || (grep -i error aiecc.log | head; false)
run: all
	./test.exe $(cfg)/kbench2.elf $(KT) $(CALLS)
clean:
	rm -rf build_kt* project kbench2.mlir kbench2.elf test.exe

.PHONY: all run clean
