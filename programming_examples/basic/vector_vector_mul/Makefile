##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# npu / npu2 use the @iron.jit workflow: `make run` JIT-compiles + runs the
# design directly in python (run_py).
#
# npu3 / AIE4 (medusa) has no xclbin flow yet: emit the resolved MLIR to
# build/aie_npu3.mlir and have aiecc package a single full ELF
# (build/design.elf); `make run` loads it via xrt::elf + xrt::ext::kernel
# ("main:sequence") through test.cpp's --full-elf path.  ``use_placed=1``
# swaps the @iron.jit design for the explicitly-placed one (which prints MLIR
# from positional device/col args).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_vector_mul
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
col = 0

use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
# explicitly-placed design: prints MLIR from positional (device, col) args
emit_mlir_args = ${devicename} ${col}
else
aie_py_src = ${targetname}.py
# @iron.jit design: prints MLIR via --emit-mlir --dev
emit_mlir_args = --emit-mlir --dev ${devicename}
endif

.PHONY: all run run_py clean

ifeq (${devicename},npu3)
all: build/design.elf
else
all:
	@echo "JIT mode: kernel compiles on first 'make run'."
endif

ifeq (${devicename},npu3)
run: ${targetname}.exe build/design.elf
	${powershell} ./$< --full-elf build/design.elf -k main:sequence
else
run: run_py
endif

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src} --dev ${devicename}

# ----------------------------------------------------------------------------
# npu3 path: emit the resolved MLIR, then aiecc packs a single full ELF.
# ----------------------------------------------------------------------------

build/aie_npu3.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

build/design.elf: build/aie_npu3.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --tmpdir=project \
		--no-xchesscc --no-xbridge \
		--get-full-elf --full-elf-name=$(@F) $(<F)

$(eval $(call build_host_exe,${targetname},_build,test.cpp))

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
