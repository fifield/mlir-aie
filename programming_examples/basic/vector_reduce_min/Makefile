##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_reduce_min

col = 0

use_placed ?= 0
ifeq (${use_placed},1)
aie_py_src = ${targetname}_placed.py
# explicitly-placed design: prints MLIR from positional device-name + col args
emit_mlir_args = ${devicename} ${col}
else
aie_py_src = ${targetname}.py
# @iron.jit design: prints MLIR via --emit-mlir -d
emit_mlir_args = --emit-mlir ${aieargs}
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

# npu3 / AIE4 (medusa): compile the compute kernel with the AIE4 flags so
# aiecc finds reduce_min.cc.o in the build/ cwd when packing the full ELF.
ifeq (${devicename},npu3)
VPATH := ${srcdir}/../../../aie_kernels/aie4
else
VPATH := ${srcdir}/../../../aie_kernels/aie2
endif

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu3)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP4_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

# npu3: emit the resolved MLIR, then aiecc packs a single full ELF.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

${design_elf_target}: build/aie.mlir build/reduce_min.cc.o
	mkdir -p ${@D}
	# The placed design references the bare reduce_min.cc.o.  The @iron.jit
	# (non-placed) design instead references the reduce kernel by a
	# content-digest object name with a digest-prefixed symbol (e.g.
	# reduce_min_vector_<hash>.o / <hash>_reduce_min_vector) that the jit
	# toolchain normally produces in-process.  Both link the same
	# aie_kernels/aie4/reduce_min.cc, so materialise whatever link_with name
	# the emitted MLIR asks for from the one compiled object, renaming the
	# symbol to match, then let aiecc pack the full ELF.
	cd ${@D} && obj=$$(grep -o 'link_with = "[^"]*"' aie.mlir | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	  if [ -n "$$obj" ] && [ "$$obj" != "reduce_min.cc.o" ]; then \
	    digest=$$(echo "$$obj" | sed 's/^reduce_min_vector_//; s/\.o$$//'); \
	    cp reduce_min.cc.o "$$obj"; \
	    llvm-objcopy --redefine-sym reduce_min_vector=$${digest}_reduce_min_vector "$$obj"; \
	  fi
	cd ${@D} && aiecc --tmpdir=project \
		--no-xchesscc --no-xbridge \
		--get-full-elf --full-elf-name=$(@F) $(<F)

$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
ifeq (${devicename},npu3)
run: ${targetname}.exe ${design_elf_target}
	${powershell} ./$< --full-elf ${design_elf_target} -k main:sequence
else
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE
endif

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
