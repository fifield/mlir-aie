##===- Makefile -----------------------------------------------------------===##
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# vadd_trace_route_test — 3-column vector add with configurable trace tile.
#
# Configurable parameters (set on make command line):
#   ROW=2   compute row to trace (2-5)
#   COL=0   compute column to trace (0-2)
#
# Quick start:
#   make               — build with defaults (ROW=2, COL=0)
#   make run ROW=3     — exercise a multi-hop trace path
#   make diff-pathfinder ROW2=2 ROW3=3 — diff pathfinder output between rows
#
# Investigation targets:
#   make check-pathfinder       — show packet-switch switch entries for current ROW
#   make show-trace-flow        — show what aie-insert-trace-flows generates
#
##===----------------------------------------------------------------------===##

srcdir    := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MLIR_DIR  := $(srcdir)/../../..
PARSE_PY  := $(MLIR_DIR)/python/utils/trace/parse.py

# Env: source env.sh before running make -- it puts aiecc/aie-opt (Peano-based
# build, no xchesscc license needed) on PATH.
AIE_OPT   := aie-opt

# Configurable trace target
ROW ?= 2
COL ?= 0

# For diff-pathfinder
ROW2 ?= 2
ROW3 ?= 3

TRACE_SIZE = 8192

all: vadd_trace.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang++ -std=c++17 \
		-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

# ── MLIR generation ──────────────────────────────────────────────────────────

vadd_trace.mlir: ${srcdir}/gen_vadd_trace.py
	python3 $< --row $(ROW) --col $(COL) --output $@
	@echo "Generated $@ (ROW=$(ROW) COL=$(COL))"

# Keep static variants for reference / quick diff
vadd_trace_row2.mlir: ${srcdir}/gen_vadd_trace.py
	python3 $< --row 2 --col 0 --output $@

vadd_trace_row3.mlir: ${srcdir}/gen_vadd_trace.py
	python3 $< --row 3 --col 0 --output $@

# ── ELF build ────────────────────────────────────────────────────────────────

vadd_trace.elf: vadd_trace.mlir
	aiecc --dump-intermediates --tmpdir=project \
		--get-full-elf --full-elf-name=$@ $<

# ── Run ──────────────────────────────────────────────────────────────────────

run: test.exe vadd_trace.elf
	rm -f trace.bin trace.json
	./test.exe vadd_trace.elf
	test -s trace.bin
	@echo ""
	@echo "Parsing trace..."
	$(PARSE_PY) --input trace.bin --mlir project/input_physical.mlir \
		--output trace.json
	test -s trace.json

# ── Investigation: pathfinder analysis ───────────────────────────────────────
#
# These targets place the IRON design, insert trace flows, and inspect/diff
# the stream switch configuration that the pathfinder generates.
#
# Check that the selected core's trace reaches the shim DMA. The compiler
# assigns the trace packet ID and chooses where to cross to the 512-bit network.

TRACE_INSERT_PASSES = builtin.module(aie.device(aie-place-tiles,aie-insert-trace-flows))
PATHFINDER_PASSES   = builtin.module(aie.device(aie-create-pathfinder-flows))

# Generate pathfinder output for current ROW/COL
pathfinder_row$(ROW).mlir: vadd_trace.mlir
	$(AIE_OPT) --pass-pipeline='$(TRACE_INSERT_PASSES)' $< | \
		$(AIE_OPT) --pass-pipeline='$(PATHFINDER_PASSES)' - > $@
	@echo "Wrote $@"

# Show trace-flow insertion result (what packet_flow does aie-insert-trace-flows emit?)
show-trace-flow: vadd_trace.mlir
	@echo "=== aie.packet_flow emitted by aie-insert-trace-flows for ROW=$(ROW) COL=$(COL) ==="
	$(AIE_OPT) --pass-pipeline='$(TRACE_INSERT_PASSES)' $< 2>/dev/null \
		| grep -B1 -A2 "aie.packet_source<.*Trace"

# Show pathfinder switch entries for current ROW/COL
check-pathfinder: vadd_trace.mlir
	@echo "=== Packet-switch switch entries for ROW=$(ROW) COL=$(COL) ==="
	$(AIE_OPT) --pass-pipeline='$(TRACE_INSERT_PASSES)' $< 2>/dev/null | \
	$(AIE_OPT) --pass-pipeline='$(PATHFINDER_PASSES)' - 2>/dev/null | \
		grep -n "switchbox\|packet_rules\|masterset.*DMA\|masterset.*South\|masterset.*North" \
		| grep -v "objectfifo\|#aie\|controller\|runtime\|core("
	@echo ""
	@echo "(Expect a connected trace route from the selected core to the shim DMA)"

# Diff pathfinder output between ROW2 and ROW3 (or any two rows)
diff-pathfinder:
	@python3 $(srcdir)/gen_vadd_trace.py --row $(ROW2) --col $(COL) | \
		$(AIE_OPT) --pass-pipeline='$(TRACE_INSERT_PASSES)' - 2>/dev/null | \
		$(AIE_OPT) --pass-pipeline='$(PATHFINDER_PASSES)' - 2>/dev/null \
		> /tmp/_pfinder_a.mlir
	@python3 $(srcdir)/gen_vadd_trace.py --row $(ROW3) --col $(COL) | \
		$(AIE_OPT) --pass-pipeline='$(TRACE_INSERT_PASSES)' - 2>/dev/null | \
		$(AIE_OPT) --pass-pipeline='$(PATHFINDER_PASSES)' - 2>/dev/null \
		> /tmp/_pfinder_b.mlir
	@echo "=== Pathfinder diff: ROW=$(ROW2) vs ROW=$(ROW3) (COL=$(COL)) ==="
	@diff --color=auto -u /tmp/_pfinder_a.mlir /tmp/_pfinder_b.mlir | head -80 || true

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf project *.elf test.exe vadd_trace.mlir
	rm -f pathfinder_row*.mlir trace.bin trace.json

.PHONY: all run show-trace-flow check-pathfinder diff-pathfinder clean vadd_trace.mlir
