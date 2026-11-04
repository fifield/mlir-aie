##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Three separately compiled ELFs (see README.md):
#   a.elf       entry main_a       section cfga  core: out = in + 1
#   b.elf       entry main_b       section cfgb  core: out = in + 2
#   a_copy.elf  entry main_a_copy  section cfga  core: out = in + 1 (a.elf's
#               configuration again, under another entry name)
ELFS := a.elf b.elf a_copy.elf

all: ids test.exe

.PHONY: all ids run clean

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -Wall -Wextra -lstdc++ -I/opt/xilinx/xrt/include \
		-L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

a.mlir: ${srcdir}/pdi_id_reload_placed.py
	python $< --entry main_a --cfg cfga --k 1 > $@

b.mlir: ${srcdir}/pdi_id_reload_placed.py
	python $< --entry main_b --cfg cfgb --k 2 > $@

a_copy.mlir: ${srcdir}/pdi_id_reload_placed.py
	python $< --entry main_a_copy --cfg cfga --k 1 > $@

%.elf: %.mlir
	aiecc --tmpdir=project_$* --get-full-elf \
		--device-name=main_$* --full-elf-name=$@ $<

# Validate the same-ID and different-ID cases before dispatching.
ids: $(ELFS)
	python ${srcdir}/check_ids.py

run: all
	./test.exe --order abab a.elf:main_a:1 b.elf:main_b:2
	./test.exe --order abab b.elf:main_b:2 a.elf:main_a:1
	./test.exe --order abab a.elf:main_a:1 a_copy.elf:main_a_copy:1
	./test.exe --order aaaa a.elf:main_a:1

clean:
	rm -rf project_* a.mlir b.mlir a_copy.mlir $(ELFS) test.exe
