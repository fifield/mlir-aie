# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run with L/COLS/CH/REPS; sweep.sh fits two REPS values to remove launch overhead.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

L ?= 256
COLS ?= 3
CH ?= 2
REPS ?= 8
BURST ?= 0
WB ?= 3072
# Concurrent DDR read bytes per written byte.
RD ?= 0
# rows per shim task (gemm issues one task per 256-row M-block)
BR ?= 256

cfg := build_L$(L)_c$(COLS)_ch$(CH)_r$(REPS)_b$(BURST)_wb$(WB)_rd$(RD)_br$(BR)
geo = $(shell python3 $(srcdir)/ddr_write_placed.py --L $(L) --cols $(COLS) --wb $(WB) --block-rows $(BR) --geometry)

all: $(cfg)/dw.elf

$(cfg)/dw.mlir: $(srcdir)/ddr_write_placed.py
	mkdir -p $(cfg)
	python3 $< --L $(L) --cols $(COLS) --ch $(CH) --reps $(REPS) --burst $(BURST) --wb $(WB) --rd $(RD) --block-rows $(BR) > $@

$(cfg)/dw.elf: $(cfg)/dw.mlir
	cd $(cfg) && aiecc -v --tmpdir=project --get-full-elf --full-elf-name=dw.elf \
		dw.mlir > aiecc.log 2>&1 || (grep -i error aiecc.log | head; false)

run: all
	python3 $(srcdir)/host.py --elf $(cfg)/dw.elf --L $(L) --cols $(COLS) \
		--rd $(RD) --wb $(word 1,$(geo)) --rows $(word 2,$(geo))

clean:
	rm -rf build_*

.PHONY: all run clean
