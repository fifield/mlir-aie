##===- Makefile -----------------------------------------------------------===##
#
<<<<<<< HEAD
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
=======
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
>>>>>>> 630b33b6d54 ([programming_examples] vector_reduce_add: add npu3 (medusa) full-ELF support)
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_reduce_add
<<<<<<< HEAD
=======
devicename ?= $(if $(filter 1,$(NPU3)),npu3,$(if $(filter 1,$(NPU2)),npu2,npu))
col = 0
>>>>>>> 630b33b6d54 ([programming_examples] vector_reduce_add: add npu3 (medusa) full-ELF support)

aie_py_src = ${targetname}.py
aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin

<<<<<<< HEAD
.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE
=======
ifeq (${devicename},npu3)
all: build/design.elf
else
all: build/final.xclbin build/insts.bin
endif

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

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} > $@

build/final.xclbin: build/aie.mlir build/reduce_add.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

# AIE4/medusa: single full ELF instead of xclbin + insts.
build/design.elf: build/aie.mlir build/reduce_add.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --tmpdir=project --no-compile-host \
		--no-xchesscc --no-xbridge \
		--generate-full-elf --full-elf-name=${@F} $(<F)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

ifeq (${devicename},npu3)
run: ${targetname}.exe build/design.elf
	${powershell} ./$< --full-elf build/design.elf -k main:sequence
else
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
endif
>>>>>>> 630b33b6d54 ([programming_examples] vector_reduce_add: add npu3 (medusa) full-ELF support)

trace:
	../../../python/utils/trace/parse.py --input trace.txt --mlir $(xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
