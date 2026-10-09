# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

M ?= 2048
N ?= 3072
K ?= 1024
OUT ?= i8
SHIFT ?= 12
ITERS ?= 1
COLS ?= 3
ABCAST ?= 1
# Overwrite tile output with cycle timestamps for debugging.
TIMING ?= 0
# Two W buffers overlap loading and replay, subject to BD and memory limits.
WBUF ?= 2
# A DMA channels per core; 2 loads the slice halves in parallel.
ASPLIT ?= 2
# Kernel compiler: chess (default, kernel.cc via xchesscc) or peano
# (kernel_peano.cc via Peano clang++; cores are then compiled/linked by Peano).
KERNEL_CC ?= chess
PEANO ?= $(PEANO_INSTALL_DIR)
ifeq ($(filter $(KERNEL_CC),chess peano),)
$(error KERNEL_CC must be chess or peano)
endif
# Core kernel: extern (default; the KERNEL_CC kernel.o is linked) or aievec
# (the generator emits the core body in flat AIEVec; Peano compiles it; no
# kernel.o).
KERNEL ?= extern
ifeq ($(filter $(KERNEL),extern aievec),)
$(error KERNEL must be extern or aievec)
endif
ifeq ($(KERNEL),aievec)
ifeq ($(origin KERNEL_CC),command line)
$(error KERNEL_CC does not apply to KERNEL=aievec)
endif
ifeq ($(TIMING),1)
$(error TIMING is not supported by KERNEL=aievec; use TRACE_SIZE instead)
endif
# Extra Peano llc flags for the aievec core (aiecc --peano-llc-arg). The
# post-pipeliner software-pipelines the K-loop at II = 65 (64 MMACs + 1),
# issuing each commit before the previous K-tile's last MMACs; its default
# maximum II (64) is one short, hence -aie-postpipeliner-maxii. Non-default
# flags get their own build directories.
AIEVEC_LLC_ARGS_DEFAULT := -aie-force-postpipeliner -aie-postpipeliner-maxii=80
AIEVEC_LLC_ARGS ?= $(AIEVEC_LLC_ARGS_DEFAULT)
ksuf := _aievec$(if $(filter-out $(AIEVEC_LLC_ARGS_DEFAULT),$(AIEVEC_LLC_ARGS))$(filter-out $(AIEVEC_LLC_ARGS),$(AIEVEC_LLC_ARGS_DEFAULT)),_llc$(shell echo '$(AIEVEC_LLC_ARGS)' | md5sum | cut -c1-6))
else
ksuf := $(if $(filter peano,$(KERNEL_CC)),_peano)
endif
# C path: 1 reassembles wide runs; 0 joins row pairs onto two shim channels.
C_REASM ?= 1
C_RUN ?= 1024
# Trace buffer bytes for column 0's cores, memtile and shim; 0 disables tracing.
TRACE_SIZE ?= 0
trace_on := $(if $(filter-out 0,$(TRACE_SIZE)),1,0)
# C tiles per core buffer; int32 at K=1024 fits CT=1 in L1.
CT ?= $(if $(filter i8,$(OUT)),4,1)

# Cache device artifacts by configuration; kernel objects use only kernel knobs.
cfg := build_$(M)x$(N)x$(K)_$(OUT)_s$(SHIFT)_it$(ITERS)_c$(COLS)_ab$(ABCAST)_ct$(CT)_wb$(WBUF)_as$(ASPLIT)_cr$(C_REASM)$(if $(filter-out 1024,$(C_RUN)),_run$(C_RUN))$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr$(TRACE_SIZE))$(ksuf)
kcfg := kernel_K$(K)_$(OUT)_s$(SHIFT)_ct$(CT)_as$(ASPLIT)$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr)$(ksuf)
out_i8 := $(if $(filter i8,$(OUT)),1,0)
out_bits := $(if $(filter i8,$(OUT)),8,32)

# A failed generator run must not leave an empty .mlir that looks up to date.
.DELETE_ON_ERROR:

all: $(cfg)/gemm_int8.elf test.exe

test.exe: ${srcdir}/test.cpp
	clang++ -std=c++17 -O3 -I/opt/xilinx/xrt/include \
		-L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil -lpthread $< -o $@

ifeq ($(KERNEL),aievec)
$(if $(PEANO),,$(error PEANO_INSTALL_DIR not set))
core_flags := --peano $(PEANO) $(foreach a,$(AIEVEC_LLC_ARGS),--peano-llc-arg=$(a))
kobj :=
else
kobj := $(kcfg)/kernel.o
endif

kdefs := -DK_DIM=$(K) -DOUT_I8=$(out_i8) -DOUT_SHIFT=$(SHIFT) -DCT=$(CT) -DTIMING=$(TIMING) -DTRACE=$(trace_on) -DA_SPLIT=$(ASPLIT)
ifeq ($(KERNEL),aievec)
else ifeq ($(KERNEL_CC),peano)
$(if $(PEANO),,$(error PEANO_INSTALL_DIR not set))
$(kcfg)/kernel.o: ${srcdir}/kernel_peano.cc
	mkdir -p $(kcfg)
	$(PEANO)/bin/clang++ --target=aie4-none-unknown-elf -O2 -std=c++20 \
		-mllvm -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load \
		$(kdefs) -c $< -o $@
core_flags := --peano $(PEANO)
else
$(kcfg)/kernel.o: ${srcdir}/kernel.cc
	mkdir -p $(kcfg)
	cd $(kcfg) && xchesscc_wrapper aie4 -I $(AIETOOLS_DIR)/include \
		$(kdefs) \
		-c $(abspath $<) -o kernel.o
core_flags := --xchesscc --xbridge
endif

# KERNEL=aievec also depends on the AIEVec helper builders.
aievec_py := $(if $(filter aievec,$(KERNEL)),$(srcdir)/../../../python/helpers/dialects/aievec_aie4.py)

$(cfg)/gemm_int8.mlir: ${srcdir}/gemm_int8_placed.py $(aievec_py)
	mkdir -p $(cfg)
	python3 $< -M $(M) -N $(N) -K $(K) --out $(OUT) --iters $(ITERS) --cols $(COLS) --a-bcast $(ABCAST) --ct $(CT) --wbuf $(WBUF) --a-split $(ASPLIT) --c-reasm $(C_REASM) --c-run $(C_RUN) --trace-size $(TRACE_SIZE)$(if $(filter aievec,$(KERNEL)), --kernel aievec --shift $(SHIFT)) > $@

$(cfg)/gemm_int8.elf: $(cfg)/gemm_int8.mlir $(kobj)
	mkdir -p $(cfg)/project
	$(if $(kobj),cp $(kobj) $(cfg)/project/kernel.o)
	# Lower FIFOs before tracing to avoid counting C FIFOs as trace channels.
	cd $(cfg) && $(if $(filter 1,$(trace_on)),aie-opt --aie-objectFifo-stateful-transform \
		gemm_int8.mlir -o gemm_int8_in.mlir,cp gemm_int8.mlir gemm_int8_in.mlir)
	cd $(cfg) && aiecc -v --tmpdir=project $(core_flags) --dump-intermediates \
		--get-full-elf --full-elf-name=gemm_int8.elf gemm_int8_in.mlir \
		> aiecc.log 2>&1 || (grep -i error aiecc.log | head -20; false)

run: all
	./test.exe $(cfg)/gemm_int8.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) $(ITERS) $(COLS) $(if $(filter 1,$(TIMING)),timing) $(if $(filter 1,$(trace_on)),trace=$(TRACE_SIZE))

# Static gate for KERNEL=aievec: .text budget, exact instruction counts and
# the zb/za timing check on each extracted K-loop (check_aievec_core.py).
# One tile body per core when a strip is a single output tile (NG*CT = N /
# (COLS*64) = 1), else two. K = 128 inlines the one-iteration K-loop: no
# K-loop to check, and Peano spills accumulators (hardware covers timing).
tile_copies := $(if $(filter $(N),$(shell expr $(COLS) \* 64)),1,2)
kt2 := $(filter 128,$(K))
ifeq ($(KERNEL),aievec)
check_aievec: $(cfg)/gemm_int8.elf
	python3 $(srcdir)/check_aievec_core.py $(cfg)/project --out $(OUT) --max-text 16384 --schedule \
		--tile-copies $(tile_copies) --expect-kloops $(if $(kt2),0,$(tile_copies)) $(if $(kt2),--allow-spills) \
		$(if $(strip $(AIEVEC_LLC_ARGS)),--allow-pipelined)
else
check_aievec:
	@echo "check_aievec needs KERNEL=aievec"; false
endif

TRACE_PY := $(srcdir)/../../../python/utils/trace

# Run with tracing, then parse, summarize and render a timeline.
# Example: make run_trace TRACE_SIZE=4194304 ITERS=2
run_trace: all
	@test "$(trace_on)" = 1 || (echo "set TRACE_SIZE=<bytes>"; false)
	cd $(cfg) && ../test.exe gemm_int8.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) \
		$(ITERS) $(COLS) trace=$(TRACE_SIZE)
	cd $(cfg) && $(TRACE_PY)/parse.py --input trace.txt \
		--mlir project/input_physical.mlir --output trace.json
	cd $(cfg) && $(TRACE_PY)/get_trace_summary.py --input trace.json
	cd $(cfg) && python3 $(srcdir)/analyze_trace.py trace.json --clk-ghz 1.2 \
		--timeline trace_timeline.png

clean:
	rm -rf build_* kernel_K* test.exe

.PHONY: all run run_trace check_aievec clean
