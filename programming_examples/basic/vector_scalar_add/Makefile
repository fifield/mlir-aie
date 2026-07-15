##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# npu / npu2 use @iron.jit's compile() directly via --xclbin-path /
# --insts-path / --elf-path.  insts.elf is the ELF-wrapped form the C++
# testbench (xrt::elf + xrt::module flow) loads.
# The same xclbin/insts pair is exercised by two testbenches:
#   * test.cpp          — single run (`make run`)
#   * test_runlist.cpp  — chained runs via xrt::runlist (`make run_runlist`,
#                          NPU2 only — runlist isn't implemented on Phoenix)
#
# npu3 / AIE4 (medusa) has no xclbin flow yet: emit the resolved MLIR and
# have aiecc.py package a single full ELF (build/design.elf); `make run`
# loads it via xrt::elf + xrt::ext::kernel("main:sequence") (test.cpp
# --full-elf path).  ``use_placed=1`` swaps the @iron.jit design for the
# explicitly-placed one (which prints MLIR from a positional device arg).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = vector_scalar_add
runlist_targetname = vector_scalar_add_runlist
runlist_hrx_targetname = vector_scalar_add_runlist_hrx

use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
# explicitly-placed design: prints MLIR from a positional device-name arg
emit_mlir_args = ${devicename}
else
aie_py_src = ${targetname}.py
# @iron.jit design: prints MLIR via --emit-mlir --dev
emit_mlir_args = --emit-mlir --dev ${devicename}
endif

aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin
elf_target = build/insts.elf
design_elf_target = build/design.elf

.PHONY: all
ifeq (${devicename},npu3)
all: ${design_elf_target}
else
all: ${xclbin_target} ${insts_target} ${elf_target}
endif

# npu / npu2: @iron.jit xclbin + insts + elf-wrapped insts.
$(eval $(call jit_xclbin_elf,${xclbin_target},${insts_target},${elf_target},${aie_py_src},${aieargs}))

# npu3: emit the resolved MLIR, then aiecc.py packs a single full ELF.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

${design_elf_target}: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --tmpdir=project --no-compile-host \
		--no-xchesscc --no-xbridge \
		--generate-full-elf --full-elf-name=$(@F) $(<F)

$(eval $(call build_host_exe,${targetname},_build,test.cpp))
$(eval $(call build_host_exe,${runlist_targetname},_build_runlist,test_runlist.cpp,-DTEST_SRC=test_runlist.cpp))
# HRX chain (runlist) testbench. Self-selects the HRX backend (-DUSE_HRX=ON) so
# `make run_runlist_hrx` works without NPU_RUNTIME=hrx.
$(eval $(call build_host_exe,${runlist_hrx_targetname},_build_runlist_hrx,test_runlist_hrx.cpp,-DTEST_SRC=test_runlist_hrx.cpp -DUSE_HRX=ON))

.PHONY: run
ifeq (${devicename},npu3)
run: ${targetname}.exe ${design_elf_target}
	${powershell} ./$< --full-elf ${design_elf_target} -k main:sequence
else
run: ${targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE
endif

.PHONY: run_runlist
run_runlist: ${runlist_targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: run_runlist_hrx
run_runlist_hrx: ${runlist_hrx_targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build _build_runlist _build_runlist_hrx ${targetname}.exe ${runlist_targetname}.exe ${runlist_hrx_targetname}.exe
