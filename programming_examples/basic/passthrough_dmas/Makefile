##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# NPU paths use @iron.jit's compile() directly via --xclbin-path / --insts-path.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = passthrough_dmas
LENGTH ?= 4096
col = 0

use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
# explicitly-placed design: prints MLIR from positional args (LENGTH dev col)
emit_mlir_args = ${LENGTH} ${devicename} ${col}
else
aie_py_src = ${targetname}.py
# @iron.jit design: prints MLIR via --emit-mlir
emit_mlir_args = --emit-mlir ${aieargs}
endif

aieargs = -d ${devicename} -n ${LENGTH}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin
design_elf_target = build/design.elf

.PHONY: all
ifeq (${devicename},npu3)
all: ${design_elf_target}
else
all: ${xclbin_target} ${insts_target}
endif

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

# npu3: emit the resolved MLIR, then aiecc packs a single full ELF.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

${design_elf_target}: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --tmpdir=project \
		--no-xchesscc --no-xbridge \
		--get-full-elf --full-elf-name=$(@F) $(<F)

.PHONY: run
ifeq (${devicename},npu3)
run: ${targetname}.exe ${design_elf_target}
	${powershell} ./$< --full-elf ${design_elf_target} -k main:sequence -l ${LENGTH}
else
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -l ${LENGTH}
endif

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
