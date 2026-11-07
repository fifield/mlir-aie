# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

M ?= 2048
N ?= 3072
# A/W element type; the first output listed is the default.
#   i8, i8u8 (int8 A x uint8 W), u8i8, i4:  OUT=i8 or i32
#   u8 (uint8 x uint8):                      OUT=u8 or i32
#   fp8 (E4M3):                              OUT=fp8, bf16 or f32
#   bf16:                                    OUT=bf16 or f32
# KERNEL=aievec produces i8, u8, i32 or f32 only.
DTYPE ?= i8
# K is in elements. Keep A at <= 1024 bytes per row: bf16 is 2 bytes per
# element and i4 half a byte.
K ?= $(if $(filter bf16,$(DTYPE)),512,$(if $(filter i4,$(DTYPE)),2048,1024))
# KERNEL=aievec has no f32 narrowing op, so its float types default to f32.
OUT ?= $(if $(filter fp8,$(DTYPE)),$(if $(filter aievec,$(KERNEL)),f32,fp8),$(if $(filter bf16,$(DTYPE)),$(if $(filter aievec,$(KERNEL)),f32,bf16),$(if $(filter u8,$(DTYPE)),u8,i8)))
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
ifeq ($(KERNEL),aievec)
ifneq ($(filter fp8 bf16,$(DTYPE)),)
ifneq ($(OUT),f32)
$(error KERNEL=aievec produces OUT=f32 for DTYPE=$(DTYPE): no f32 narrowing op)
endif
endif
endif
# C path: 1 reassembles wide runs; 0 joins row pairs onto two shim channels.
C_REASM ?= 1
C_RUN ?= 1024
# Trace buffer bytes for column 0's cores, memtile and shim; 0 disables tracing.
TRACE_SIZE ?= 0
trace_on := $(if $(filter-out 0,$(TRACE_SIZE)),1,0)
# C tiles per core buffer; 4-byte C at 1 KiB of A per row fits CT=1 in L1.
CT ?= $(if $(filter i8 u8 fp8,$(OUT)),4,$(if $(filter bf16,$(OUT)),2,1))
# Non-int8 types get their own build directories; int8 names are unchanged.
dt := $(if $(filter-out i8,$(DTYPE)),_$(DTYPE))

# Cache device artifacts by configuration; kernel objects use only kernel knobs.
cfg := build_$(M)x$(N)x$(K)_$(OUT)$(dt)_s$(SHIFT)_it$(ITERS)_c$(COLS)_ab$(ABCAST)_ct$(CT)_wb$(WBUF)_as$(ASPLIT)_cr$(C_REASM)$(if $(filter-out 1024,$(C_RUN)),_run$(C_RUN))$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr$(TRACE_SIZE))$(ksuf)
kcfg := kernel_K$(K)_$(OUT)$(dt)_s$(SHIFT)_ct$(CT)_as$(ASPLIT)$(if $(filter 1,$(TIMING)),_tim)$(if $(filter 1,$(trace_on)),_tr)$(ksuf)
out_i8 := $(if $(filter i8,$(OUT)),1,0)
out_bits := $(if $(filter i8 u8 fp8,$(OUT)),8,$(if $(filter bf16,$(OUT)),16,32))
out_u8 := $(if $(filter u8,$(OUT)),1,0)
out_bf16 := $(if $(filter bf16,$(OUT)),1,0)
out_fp8 := $(if $(filter fp8,$(OUT)),1,0)
dtype_def := $(strip $(if $(filter fp8,$(DTYPE)),-DDTYPE_FP8) \
	$(if $(filter bf16,$(DTYPE)),-DDTYPE_BF16) \
	$(if $(filter i4,$(DTYPE)),-DDTYPE_I4) \
	$(if $(filter u8 u8i8,$(DTYPE)),-DA_UNS=1) \
	$(if $(filter u8 i8u8,$(DTYPE)),-DW_UNS=1))

# A failed generator run must not leave an empty .mlir that looks up to date.
.DELETE_ON_ERROR:

all: $(cfg)/gemm.elf test.exe

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

kdefs := -DK_DIM=$(K) $(dtype_def) -DOUT_I8=$(out_i8) -DOUT_U8=$(out_u8) -DOUT_BF16=$(out_bf16) -DOUT_FP8=$(out_fp8) -DOUT_SHIFT=$(SHIFT) -DCT=$(CT) -DTIMING=$(TIMING) -DTRACE=$(trace_on) -DA_SPLIT=$(ASPLIT)
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

$(cfg)/gemm.mlir: ${srcdir}/gemm_placed.py $(aievec_py)
	mkdir -p $(cfg)
	python3 $< -M $(M) -N $(N) -K $(K) --dtype $(DTYPE) --out $(OUT) --iters $(ITERS) --cols $(COLS) --a-bcast $(ABCAST) --ct $(CT) --wbuf $(WBUF) --a-split $(ASPLIT) --c-reasm $(C_REASM) --c-run $(C_RUN) --trace-size $(TRACE_SIZE)$(if $(filter aievec,$(KERNEL)), --kernel aievec --shift $(SHIFT)) > $@

$(cfg)/gemm.elf: $(cfg)/gemm.mlir $(kobj)
	mkdir -p $(cfg)/project
	$(if $(kobj),cp $(kobj) $(cfg)/project/kernel.o)
	# Lower FIFOs before tracing to avoid counting C FIFOs as trace channels.
	cd $(cfg) && $(if $(filter 1,$(trace_on)),aie-opt --aie-objectFifo-stateful-transform \
		gemm.mlir -o gemm_in.mlir,cp gemm.mlir gemm_in.mlir)
	cd $(cfg) && aiecc -v --tmpdir=project $(core_flags) --dump-intermediates \
		--get-full-elf --full-elf-name=gemm.elf gemm_in.mlir \
		> aiecc.log 2>&1 || (grep -i -m 20 'error:' aiecc.log || tail -20 aiecc.log; false)

run: all
	./test.exe $(cfg)/gemm.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) $(ITERS) $(COLS) dtype=$(DTYPE) $(if $(filter 1,$(TIMING)),timing) $(if $(filter 1,$(trace_on)),trace=$(TRACE_SIZE))

# Static gate for KERNEL=aievec: .text budget, exact instruction counts and
# the zb/za timing check on each extracted K-loop (check_aievec_core.py).
# One tile body per core when a strip is a single output tile (NG*CT = N /
# (COLS*64) = 1), else two. K = 128 inlines the one-iteration K-loop: no
# K-loop to check, and Peano spills accumulators (hardware covers timing).
tile_copies := $(if $(filter $(N),$(shell expr $(COLS) \* 64)),1,2)
# K is in elements; the DMA and core structure follow the A row length in bytes.
kbytes := $(if $(filter bf16,$(DTYPE)),$(shell expr $(K) \* 2),$(if $(filter i4,$(DTYPE)),$(shell expr $(K) / 2),$(K)))
kt := $(shell expr $(kbytes) / 64)
# No K-loop survives at two K-tiles, nor at three for float types (the
# one-iteration loop is inlined).
kt2 := $(if $(filter 2,$(kt)),1,$(if $(and $(filter 3,$(kt)),$(filter fp8 bf16,$(DTYPE))),1))
ifeq ($(KERNEL),aievec)
check_aievec: $(cfg)/gemm.elf
	python3 $(srcdir)/check_aievec_core.py $(cfg)/project --out $(OUT) --max-text 16384 --schedule --kt $(kt) $(if $(filter fp8 bf16,$(DTYPE)),--float) \
		--tile-copies $(tile_copies) --expect-kloops $(if $(kt2),0,$(tile_copies)) $(if $(or $(kt2),$(filter fp8 bf16,$(DTYPE))),--allow-spills) \
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
	cd $(cfg) && ../test.exe gemm.elf $(M) $(N) $(K) $(out_bits) $(SHIFT) \
		$(ITERS) $(COLS) dtype=$(DTYPE) trace=$(TRACE_SIZE)
	cd $(cfg) && $(TRACE_PY)/parse.py --input trace.txt \
		--mlir project/input_physical.mlir --output trace.json
	cd $(cfg) && $(TRACE_PY)/get_trace_summary.py --input trace.json
	cd $(cfg) && python3 $(srcdir)/analyze_trace.py trace.json --clk-ghz 1.2 \
		--timeline trace_timeline.png

clean:
	rm -rf build_* kernel_K* test.exe

.PHONY: all run run_trace check_aievec clean
