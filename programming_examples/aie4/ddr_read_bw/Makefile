# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# assumes MLIR_AIE_DIR has been set by utils/pacifica_setup.sh or similar process
CFLAGS=-I${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/include \
		-L${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/lib \
		-ltest_utils

.PHONY:
all: project/mesim.sh

project/mesim.sh: aie4.mlir test.cpp
	aiecc.py --aiesim --tmpdir=project --xchesscc --xbridge \
			--aie-generate-txn --txn-name=transaction.bin \
			aie4.mlir ${CFLAGS} test.cpp

aie4.mlir: aie4.py
	python3 $< > $@

clean:
	rm -rf project aie4.mlir *.txt *.log *.bin *.toml mesim_output mesim.vcd