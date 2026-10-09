# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
KT ?= 8
CALLS ?= 4
# Key artifacts by both knobs to match the host's runtime buffer sizes.
# KERNEL_CC=peano builds kernel_peano.cc with Peano instead of kernel.cc with chess.
KERNEL_CC ?= chess
PEANO ?= $(PEANO_INSTALL_DIR)
ifeq ($(filter $(KERNEL_CC),chess peano),)
$(error KERNEL_CC must be chess or peano)
endif
ksuf := $(if $(filter peano,$(KERNEL_CC)),_peano)
cfg := build_kt$(KT)_calls$(CALLS)$(ksuf)
.DELETE_ON_ERROR:
all: $(cfg)/kbench2.elf test.exe
test.exe: ${srcdir}/test.cpp
	clang++ -std=c++17 -O2 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@
ifeq ($(KERNEL_CC),peano)
$(if $(PEANO),,$(error PEANO_INSTALL_DIR not set))
$(cfg)/project/kernel.o: ${srcdir}/kernel_peano.cc
	mkdir -p $(cfg)/project
	$(PEANO)/bin/clang++ --target=aie4-none-unknown-elf -O2 -std=c++20 \
		-mllvm -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load \
		-DKT=$(KT) $(PEANO_CFLAGS) -c $< -o $@
core_flags := --peano $(PEANO)
else
$(cfg)/project/kernel.o: ${srcdir}/kernel.cc
	mkdir -p $(cfg)/project
	cd $(cfg)/project && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include -DKT=$(KT) -c $(abspath $<) -o kernel.o
core_flags := --xchesscc --xbridge
endif
$(cfg)/kbench2.mlir: ${srcdir}/kbench2_placed.py
	mkdir -p $(cfg)
	python3 $< $(KT) $(CALLS) > $@
$(cfg)/kbench2.elf: $(cfg)/kbench2.mlir $(cfg)/project/kernel.o
	cd $(cfg) && aiecc -v --tmpdir=project $(core_flags) --get-full-elf --full-elf-name=kbench2.elf kbench2.mlir > aiecc.log 2>&1 || (grep -i error aiecc.log | head; false)
run: all
	./test.exe $(cfg)/kbench2.elf $(KT) $(CALLS)
clean:
	rm -rf build_kt* project kbench2.mlir kbench2.elf test.exe

.PHONY: all run clean
