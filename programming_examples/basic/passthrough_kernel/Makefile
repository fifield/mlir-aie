##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# npu / npu2: @iron.jit python flow — the design .py JITs the kernel and runs
# it (`make run` / `make trace`).  No C++ host build on these devices.
#
# npu3 / AIE4 (medusa) has no xclbin flow yet: emit the explicitly-placed
# design's MLIR, compile the passThrough kernel with Peano (aie4), and have
# aiecc pack a single full ELF (build/design.elf).  `make run` loads it via
# the C++ testbench's xrt_test_wrapper --full-elf path (xrt::elf +
# xrt::ext::kernel "main:sequence").  npu3 is exercised placed-only
# (use_placed=1); the placed design hand-adds the AIE4 controller_id +
# Control32->Controller32 packet flows.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/eltwise

targetname = passthrough_kernel
in1_size = 4096 # in bytes (also element count, since the kernel is uint8)
out_size = 4096 # in bytes, should always be equal to in1_size
trace_size = 8192

aie_py_src = ${targetname}.py
use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
endif

.PHONY: all run run_py trace trace_py clean clean_trace

ifeq (${devicename},npu3)
# --- npu3 / AIE4 (medusa): full ELF + C++ full-elf host --------------------
all: build/design.elf

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP4_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -os ${out_size} > $@

build/design.elf: build/aie.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --tmpdir=project \
		--get-full-elf --full-elf-name=$(@F) $(<F)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe build/design.elf
	${powershell} ./$< --full-elf build/design.elf -k main:sequence

else
# --- npu / npu2: @iron.jit python flow ------------------------------------
all:
	@echo "JIT mode: kernel compiles on first 'make run' / 'make run_py'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size}

trace: trace_py

trace_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size} -t ${trace_size}
endif

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
