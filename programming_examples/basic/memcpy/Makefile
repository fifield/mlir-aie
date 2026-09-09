##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# npu / npu2: @iron.jit drives aiecc directly via --xclbin-path /
# --insts-path / --elf-path; ExternalFunction(source_file=passThrough.cc)
# lets iron build the .o into the same dir, no separate Makefile rule needed.
#
# npu3 / AIE4 (medusa) has no xclbin flow yet: emit the resolved MLIR and
# have aiecc package a single full ELF (build/design.elf); `make run`
# loads it via xrt::elf + xrt::ext::kernel("main:sequence") (test.cpp
# --full-elf path).  The passthrough compute kernel is compiled to
# build/passThrough.cc.o with PEANOWRAP4_FLAGS for aiecc to pick up.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/generic

SHELL := /bin/bash

targetname = memcpy
length ?= 16384
cols ?= 1
chans ?= 1
bypass ?= True

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -l ${length} -co ${cols} -ch ${chans} -b ${bypass}
emit_mlir_args = --emit-mlir ${aieargs}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin
elf_target    = build/insts.elf
design_elf_target = build/design.elf

.PHONY: all
ifeq (${devicename},npu3)
all: ${design_elf_target}
else
all: ${xclbin_target} ${insts_target} ${elf_target}
endif

# npu / npu2: @iron.jit xclbin + insts + elf-wrapped insts.
$(eval $(call jit_xclbin_elf,${xclbin_target},${insts_target},${elf_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

# npu3: compute kernel object, compiled for AIE4.  BIT_WIDTH must match
# memcpy.py's xfr_dtype (np.int32); iron.jit passes the same flag itself.
build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP4_FLAGS} -DBIT_WIDTH=32 -c $< -o ${@F}

# npu3: emit the resolved MLIR, then aiecc packs a single full ELF.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${emit_mlir_args} > $@

${design_elf_target}: build/aie.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	# The @iron.jit (non-placed, non-bypass) design references the passthrough
	# kernel by a content-digest object name with a digest-prefixed symbol
	# (e.g. passThroughLine_<hash>.o / <hash>_passThroughLine) that the jit
	# toolchain normally produces in-process.  --emit-mlir does not compile
	# kernels and aiecc --get-full-elf only STAGES pre-built .o files,
	# so materialise whatever link_with name the emitted MLIR asks for from the
	# one compiled object, renaming the symbol to match, then let aiecc pack the
	# full ELF.  No-op for the bypass path (no kernel) and the bare name.
	cd ${@D} && obj=$$(grep -o 'link_with = "[^"]*"' aie.mlir | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	  if [ -n "$$obj" ] && [ "$$obj" != "passThrough.cc.o" ]; then \
	    digest=$$(echo "$$obj" | sed 's/^passThroughLine_//; s/\.o$$//'); \
	    cp passThrough.cc.o "$$obj"; \
	    llvm-objcopy --redefine-sym passThroughLine=$${digest}_passThroughLine "$$obj"; \
	  fi
	cd ${@D} && aiecc --tmpdir=project \
		--get-full-elf --full-elf-name=$(@F) $(<F)

.PHONY: run
ifeq (${devicename},npu3)
run: ${targetname}.exe ${design_elf_target}
	${powershell} ./$< --full-elf ${design_elf_target} -k main:sequence -l ${length}
else
run: ${targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE -l ${length}
endif

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
