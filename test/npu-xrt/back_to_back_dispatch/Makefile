##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Number of dispatches and host timeout (ms) per wait. Bounded waits turn a
# hung dispatch into a reported timeout instead of a hung test.
N          ?= 200
TIMEOUT_MS ?= 5000

# back_to_back_dispatch.elf           out = in + 1
# back_to_back_dispatch_outfirst.elf  same, core acquires its output first
all: back_to_back_dispatch.elf back_to_back_dispatch_outfirst.elf test.exe

# Builds for the measurements and manual checks in README.md:
# back_to_back_dispatch_expand.elf  with --expand-load-pdis
# back_to_back_dispatch_b.elf       entry main_b, out = in + 2 (other PDI ids)
extra: back_to_back_dispatch_expand.elf back_to_back_dispatch_b.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

back_to_back_dispatch.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< > $@

back_to_back_dispatch_outfirst.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< --output-first > $@

back_to_back_dispatch_b.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< --name main_b --k 2 > $@

back_to_back_dispatch.elf: back_to_back_dispatch.mlir
	aiecc --tmpdir=project \
		--get-full-elf --full-elf-name=$@ $<

back_to_back_dispatch_outfirst.elf: back_to_back_dispatch_outfirst.mlir
	aiecc --tmpdir=project_outfirst \
		--get-full-elf --full-elf-name=$@ $<

back_to_back_dispatch_expand.elf: back_to_back_dispatch.mlir
	aiecc --tmpdir=project_expand --expand-load-pdis \
		--get-full-elf --full-elf-name=$@ $<

back_to_back_dispatch_b.elf: back_to_back_dispatch_b.mlir
	aiecc --tmpdir=project_b --device-name=main_b \
		--get-full-elf --full-elf-name=$@ $<

run: all
	for m in seq runlist startall; do \
		./test.exe --elf back_to_back_dispatch.elf --mode $$m --n $(N) \
			--timeout-ms $(TIMEOUT_MS) || exit 1; \
	done
	./test.exe --elf back_to_back_dispatch_outfirst.elf --mode seq --n $(N) \
		--timeout-ms $(TIMEOUT_MS)

run-expand: back_to_back_dispatch_expand.elf test.exe
	./test.exe --elf back_to_back_dispatch_expand.elf --mode seq --n 8 \
		--timeout-ms $(TIMEOUT_MS)

# Expected to FAIL until the configuration resets the array before
# configuring it: every dispatch reloads the other ELF's configuration over
# live memtile channels. Bounded to 8 runs; run it last and
# check the device afterwards (see README.md).
run-alternate: back_to_back_dispatch.elf back_to_back_dispatch_b.elf test.exe
	./test.exe --elf back_to_back_dispatch.elf --alt-elf back_to_back_dispatch_b.elf \
		--mode alternate --n 8 --timeout-ms $(TIMEOUT_MS)

clean:
	rm -rf project project_* back_to_back_dispatch*.mlir \
		back_to_back_dispatch*.elf test.exe
