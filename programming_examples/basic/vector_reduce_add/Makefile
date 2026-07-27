##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# npu / npu2 use @iron.jit's compile() directly (jit_xclbin -> xclbin + insts);
# `make run` loads them via test.cpp (xrt xclbin + insts flow).
#
# npu3 / AIE4 (medusa) has no xclbin flow yet: emit the resolved MLIR and
# have aiecc package a single full ELF (build/design.elf); `make run`
# loads it via xrt::elf + xrt::ext::kernel("main:sequence") (test.cpp
# --full-elf path).  ``use_placed=1`` swaps the @iron.jit design for the
# explicitly-placed one (which prints MLIR from positional device/col args).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_reduce_add
col = 0

use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
# explicitly-placed design: prints MLIR from positional device-name + col args
emit_mlir_args = ${devicename} ${col}
else
aie_py_src = ${targetname}.py
# @iron.jit design: prints MLIR via --emit-mlir -d
emit_mlir_args = --emit-mlir -d ${devicename}
endif

aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin
design_elf_target = build/design.elf

.PHONY: all
ifeq (${devicename},npu3)
all: ${design_elf_target}
else
all: ${xclbin_target} ${insts_target}
endif

# npu / npu2: @iron.jit xclbin + insts.
$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

# npu3: compute-kernel object built with the AIE4 peano flags.
ifeq (${devicename},npu3)
VPATH := ${srcdir}/../../../aie_kernels/aie4
else ifeq (${devicename},npu2)
VPATH := ${srcdir}/../../../aie_kernels/aie2
else
VPATH := ${srcdir}/../../../aie_kernels/aie2
endif

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu3)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP4_FLAGS}  -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS}  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS}  -c $< -o ${@F}
endif

# npu3: emit the resolved MLIR, then aiecc packs a single full ELF.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

${design_elf_target}: build/aie.mlir build/reduce_add.cc.o
	mkdir -p ${@D}
	# The placed design references the bare reduce_add.cc.o.  The @iron.jit
	# (non-placed) design instead references the reduce kernel by a
	# content-digest object name with a digest-prefixed symbol (e.g.
	# reduce_add_vector_<hash>.o / <hash>_reduce_add_vector) that the jit
	# toolchain normally produces in-process.  Both link the same
	# aie_kernels/aie4/reduce_add.cc, so materialise whatever link_with name
	# the emitted MLIR asks for from the one compiled object, renaming the
	# symbol to match, then let aiecc pack the full ELF.
	cd ${@D} && obj=$$(grep -o 'link_with = "[^"]*"' aie.mlir | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	  if [ -n "$$obj" ] && [ "$$obj" != "reduce_add.cc.o" ]; then \
	    digest=$$(echo "$$obj" | sed 's/^reduce_add_vector_//; s/\.o$$//'); \
	    cp reduce_add.cc.o "$$obj"; \
	    llvm-objcopy --redefine-sym reduce_add_vector=$${digest}_reduce_add_vector "$$obj"; \
	  fi
	cd ${@D} && aiecc --tmpdir=project \
		--no-xchesscc --no-xbridge \
		--get-full-elf --full-elf-name=$(@F) $(<F)

.PHONY: run
ifeq (${devicename},npu3)
run: ${targetname}.exe ${design_elf_target}
	${powershell} ./$< --full-elf ${design_elf_target} -k main:sequence
else
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE
endif

trace:
	../../../python/utils/trace/parse.py --input trace.txt --mlir $(xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
