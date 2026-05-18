##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# dispatch_micro: micro-benchmarks comparing NPU configuration / dispatch
# mechanisms (xclbin, load_pdi firmware, load_pdi expanded, ctrl packets).
#
# Per-build-key directory layout (all under build/<key>/):
#   aie.mlir         generated placed-IRON
#   passThrough.cc.o compiled core kernel
#   aie.xclbin       xclbin (mechanisms: baseline, ctrlpkt)
#   insts.bin        NPU instruction stream
#   aie.elf          full ELF with PDIs (mechanisms: load_pdi_fw, load_pdi_expanded)
#   ctrlpkt.bin      control-packet binary (mechanism: ctrlpkt)
#   ctrlpkt_dma_seq.bin
#   sizes.json       artifact byte sizes + aiecc command
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/generic

# Mechanism / sweep params (override via env or `make MECH=... TILES=...`).
MECH         ?= baseline
DEVICE       ?= npu2_1col
TILES        ?= 1
ROWS_PER_COL ?= 1
BDS          ?= 2
TOPOLOGY     ?= linear
AB           ?= 0
N_CONFIGS    ?= 0

# N_CONFIGS takes precedence over AB. Build key reflects multi-config too.
KEY := $(MECH)_$(DEVICE)_t$(TILES)_r$(ROWS_PER_COL)_b$(BDS)_$(TOPOLOGY)$(if $(filter-out 0,$(N_CONFIGS)),_n$(N_CONFIGS),$(if $(filter 1,$(AB)),_ab,))
BUILD := build/$(KEY)

# Per-mechanism aiecc flags. xclbin family uses --aie-generate-xclbin; ELF
# family uses --generate-full-elf. ctrlpkt adds --aie-generate-ctrlpkt to the
# xclbin family. --expand-load-pdis only differs for load_pdi_expanded.
AIECC_COMMON := --no-xchesscc --no-xbridge

ifeq ($(MECH),baseline)
  AIECC_FLAGS := $(AIECC_COMMON) --aie-generate-xclbin --aie-generate-npu-insts \
                 --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.bin
  ARTIFACTS   := $(BUILD)/aie.xclbin $(BUILD)/insts.bin
else ifeq ($(MECH),ctrlpkt)
  AIECC_FLAGS := $(AIECC_COMMON) --aie-generate-xclbin --aie-generate-ctrlpkt \
                 --no-compile-host --xclbin-name=aie.xclbin \
                 --ctrlpkt-name=ctrlpkt.bin --ctrlpkt-dma-seq-name=ctrlpkt_dma_seq.bin
  ARTIFACTS   := $(BUILD)/aie.xclbin $(BUILD)/ctrlpkt.bin $(BUILD)/ctrlpkt_dma_seq.bin
else ifeq ($(MECH),load_pdi_fw)
  AIECC_FLAGS := $(AIECC_COMMON) --generate-full-elf
  ARTIFACTS   := $(BUILD)/aie.elf
else ifeq ($(MECH),load_pdi_expanded)
  AIECC_FLAGS := $(AIECC_COMMON) --generate-full-elf --expand-load-pdis
  ARTIFACTS   := $(BUILD)/aie.elf
else
  $(error Unknown MECH=$(MECH); choose baseline | ctrlpkt | load_pdi_fw | load_pdi_expanded)
endif

GEN_FLAGS := --mechanism=$(MECH) --device=$(DEVICE) \
             --tiles=$(TILES) --rows-per-col=$(ROWS_PER_COL) \
             --bds=$(BDS) --topology=$(TOPOLOGY) \
             $(if $(filter-out 0,$(N_CONFIGS)),--n-configs=$(N_CONFIGS),$(if $(filter 1,$(AB)),--ab,))

.PHONY: all bench smoke clean clean-all

all: $(BUILD)/sizes.json

# Generate MLIR.
$(BUILD)/aie.mlir: $(srcdir)/generate.py
	mkdir -p $(@D)
	python3 $< $(GEN_FLAGS) > $@

# Compile the core kernel for npu2 (Strix → aie2p). Single device family
# supported for now; extend if npu1 builds are needed.
$(BUILD)/passThrough.cc.o: passThrough.cc
	mkdir -p $(@D)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) \
		-DBIT_WIDTH=8 -c $< -o $(@F)

# Build the per-mechanism artifact set.
$(BUILD)/.compiled: $(BUILD)/aie.mlir $(BUILD)/passThrough.cc.o
	mkdir -p $(@D)
	cd $(@D) && aiecc $(AIECC_FLAGS) aie.mlir
	@touch $@

# Materialize the human-named artifact files as alias deps of .compiled.
$(ARTIFACTS): $(BUILD)/.compiled
	@test -f $@ || { echo "Missing artifact: $@"; exit 1; }

# Emit sizes.json after the build. Also runs verify_artifact.py for content
# sanity (opcode counts, ctrlpkt parity); a failure here aborts the build.
$(BUILD)/sizes.json: $(ARTIFACTS) $(srcdir)/verify_artifact.py
	@python3 $(srcdir)/verify_artifact.py --build-dir $(BUILD) \
		--mechanism $(MECH) --bds $(BDS) --tiles $(TILES) \
		> $@

bench: $(srcdir)/bench.cpp $(srcdir)/bench_runner.h $(srcdir)/json_writer.h
	rm -rf _bench && mkdir -p _bench
	cd _bench && cmake -DCMAKE_BUILD_TYPE=Release $(srcdir)
	cmake --build _bench --config Release
	cp _bench/bench $(srcdir)/bench

# Smoke slice: one representative combination per supported mechanism.
# `ctrlpkt` needs the column-control-overlay pass (see test/npu-xrt/
# ctrl_packet_reconfig_elf/run.lit) — wired up in v2; skipped here.
smoke:
	$(MAKE) MECH=baseline           DEVICE=npu2_1col TILES=1 BDS=2 TOPOLOGY=linear
	$(MAKE) MECH=load_pdi_fw        DEVICE=npu2_1col TILES=1 BDS=2 TOPOLOGY=linear
	$(MAKE) MECH=load_pdi_expanded  DEVICE=npu2_1col TILES=1 BDS=2 TOPOLOGY=linear
	$(MAKE) bench

clean:
	rm -rf $(BUILD)

clean-all:
	rm -rf build _bench bench
