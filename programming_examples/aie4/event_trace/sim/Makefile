##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

MLIR_AIE_INSTALL_DIR ?= /scratch/jefff/pacifica-sim/install
MLIR_AIE_BUILD_DIR ?= /scratch/jefff/pacifica-sim/build
CERT_ELF ?= ${MLIR_AIE_INSTALL_DIR}/cert/cert_medusa.elf
TEST_LIB  = ${MLIR_AIE_BUILD_DIR}/runtime_lib/x86_64/test_lib
CFLAGS    = -I${TEST_LIB}/include -L${TEST_LIB}/lib -ltest_utils -lelf

all: run_trace

project/project0/cert_medusa.elf:
	mkdir -p project/project0
	cp ${CERT_ELF} project/project0/cert_medusa.elf

project/transaction.bin: ${srcdir}/base.mlir project/project0/cert_medusa.elf
	cd project && aiecc --tmpdir=project0 --xchesscc --xbridge \
		--aie-generate-txn --txn-name=transaction.bin \
		${srcdir}/base.mlir

project/scale.o: ${srcdir}/../vector_scalar_mul.cc
	mkdir -p project/
	cd project/ && xchesscc_wrapper aie4 -I ${AIETOOLS_DIR}/include -c $< -o scale.o

project/mesim.sh: ${srcdir}/aie_trace.mlir ${srcdir}/test.cpp project/scale.o project/transaction.bin
	cd project && aiecc --tmpdir=project1 --xchesscc --xbridge \
		--no-compile-host --generate-full-elf --full-elf-name=aie4.elf \
		--aiesim ${srcdir}/aie_trace.mlir \
		${CFLAGS} ${srcdir}/test.cpp
	ln -sf project1/mesim.sh project/mesim.sh

run_trace: project/mesim.sh
	project/mesim.sh
	@if [ -f trace.bin ]; then \
		echo ""; \
		echo "Converting trace.bin to hex text..."; \
		python3 -c "import struct, sys; data=open('trace.bin','rb').read(); words=struct.unpack('<'+'I'*(len(data)//4),data); [print(f'{w:08x}') for w in words]" > trace.txt; \
		echo "Parsing trace output..."; \
		${srcdir}/../../../../python/utils/trace/parse.py --input trace.txt --mlir project/project1/main_physical_with_elfs.mlir --output trace.json 2>/dev/null || true; \
		if [ -f trace.json ]; then \
			${srcdir}/../../../../python/utils/trace/get_trace_summary.py --input trace.json 2>/dev/null || true; \
			echo ""; \
			echo "Generating trace visualization..."; \
			python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "AIE4 Event Trace" 2>/dev/null || true; \
		fi; \
	fi

clean:
	rm -rf project trace.bin trace.txt trace.json trace_timeline.png \
		diag_report.log mesim_output performance.txt stats_0.toml xsc_report.log

.PHONY: all run_trace clean
