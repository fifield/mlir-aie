# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CHANNELS ?= 12
CHANNEL_ORDER ?= roundrobin
BYTES_PER_CHANNEL ?= 16MiB
SINK_BYTES ?= 4KiB
ITERATIONS ?= 8
PATTERN ?= linear
STRIDE_ELEMENTS ?= 16
BLOCK_BYTES ?= 4KiB
GAP_BYTES ?= 0
BURST_LENGTH ?= 0
SAMPLES ?= 5
WARMUPS ?= 1

.PHONY: all run clean force

GEN_ARGS = \
	--channels $(CHANNELS) \
	--channel-order $(CHANNEL_ORDER) \
	--bytes-per-channel $(BYTES_PER_CHANNEL) \
	--sink-bytes $(SINK_BYTES) \
	--iterations $(ITERATIONS) \
	--pattern $(PATTERN) \
	--stride-elements $(STRIDE_ELEMENTS) \
	--block-bytes $(BLOCK_BYTES) \
	--gap-bytes $(GAP_BYTES) \
	--burst-length $(BURST_LENGTH)

HOST_ARGS = \
	--channels $(CHANNELS) \
	--channel-order $(CHANNEL_ORDER) \
	--bytes-per-channel $(BYTES_PER_CHANNEL) \
	--sink-bytes $(SINK_BYTES) \
	--iterations $(ITERATIONS) \
	--pattern $(PATTERN) \
	--stride-elements $(STRIDE_ELEMENTS) \
	--block-bytes $(BLOCK_BYTES) \
	--gap-bytes $(GAP_BYTES) \
	--samples $(SAMPLES) \
	--warmups $(WARMUPS)

all: shim_dma_bw.elf

shim_dma_bw.mlir: ${srcdir}/shim_dma_bw_placed.py ${srcdir}/shim_dma_bw_config.py force
	python ${srcdir}/shim_dma_bw_placed.py $(GEN_ARGS) > $@

shim_dma_bw.elf: shim_dma_bw.mlir
	aiecc -v --tmpdir=project --xchesscc --xbridge \
		--get-full-elf --full-elf-name=$@ $<

run: shim_dma_bw.elf
	python ${srcdir}/host.py --elf shim_dma_bw.elf $(HOST_ARGS)

clean:
	rm -rf project shim_dma_bw.mlir shim_dma_bw.elf
