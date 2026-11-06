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

# Builds that reload a configuration on every dispatch (see README.md):
# back_to_back_dispatch_expand.elf  with --expand-load-pdis
# back_to_back_dispatch_b.elf       entry main_b, out = in + 2 (other PDI ids)
# back_to_back_dispatch_aba.elf     A (in + 1), B (in + 2), A again in one
#                                   program, on the same tiles and locks
extra: back_to_back_dispatch_expand.elf back_to_back_dispatch_b.elf \
	back_to_back_dispatch_aba.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

back_to_back_dispatch.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< > $@

back_to_back_dispatch_outfirst.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< --output-first > $@

back_to_back_dispatch_b.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< --name main_b --k 2 > $@

back_to_back_dispatch_aba.mlir: ${srcdir}/back_to_back_dispatch_placed.py
	python $< --aba > $@

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

back_to_back_dispatch_aba.elf: back_to_back_dispatch_aba.mlir
	aiecc --tmpdir=project_aba \
		--get-full-elf --full-elf-name=$@ $<

run: all
	for m in seq runlist startall; do \
		./test.exe --elf back_to_back_dispatch.elf --mode $$m --n $(N) \
			--timeout-ms $(TIMEOUT_MS) || exit 1; \
	done
	./test.exe --elf back_to_back_dispatch_outfirst.elf --mode seq --n $(N) \
		--timeout-ms $(TIMEOUT_MS)

# Each of these reconfigures the array on every dispatch, over the memtile
# channels the previous dispatch left running; the reset at the head of the
# configuration is what makes them exact. Without it they fail
# and would hang after 16 dispatches, so they stay bounded to 8 runs.
run-extra: run-expand run-aba run-alternate

run-expand: back_to_back_dispatch_expand.elf test.exe
	./test.exe --elf back_to_back_dispatch_expand.elf --mode seq --n 8 \
		--timeout-ms $(TIMEOUT_MS)

# Every dispatch loads the other ELF's configuration.
run-alternate: back_to_back_dispatch.elf back_to_back_dispatch_b.elf test.exe
	./test.exe --elf back_to_back_dispatch.elf --alt-elf back_to_back_dispatch_b.elf \
		--mode alternate --n 8 --timeout-ms $(TIMEOUT_MS)

# A, B, A in one program: within a dispatch both transitions reload; the next
# dispatch's first load of A is skipped (same id as the last one loaded).
run-aba: back_to_back_dispatch_aba.elf test.exe
	./test.exe --elf back_to_back_dispatch_aba.elf --mode aba --n 8 \
		--timeout-ms $(TIMEOUT_MS)

clean:
	rm -rf project project_* back_to_back_dispatch*.mlir \
		back_to_back_dispatch*.elf test.exe
