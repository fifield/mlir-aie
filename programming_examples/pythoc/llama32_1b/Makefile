# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B BF16/AWQ Inference on NPU2 (MLIR-AIE port)
#
# Low-level duplicate of ../llama32_1b. The kernels are compiled by aiecc
# directly (no aircc in the steady state). Each unique kernel's cached IR
# lives at <build>/decode_kernel_cache/<name>.npu.air.mlir and can be
# hand-edited; rerun `make compile` after editing to rebuild the ELF.
#
# Quick start:
#   make compile        - Compile all kernels (~3 min, cached to disk)
#   make run            - Run inference (NPU prefill + NPU decode)
#   make profile        - Run with per-kernel profiling breakdown

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Build directory
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Workaround: pythoc-tree llvm-aie (commit 55604435) crashes in
# InterBlockScheduling::emitLoopRemarks on the RoPE-K core of rms_gemv_rope.
# The AIR-tree pip llvm-aie (commit 5ed1593) doesn't crash. If
# PYTHOC_LLAMA_PEANO is set we use that instead. Default to the known-good
# pip install if it exists.
PYTHOC_LLAMA_PEANO ?= /home/jfifield/npu-dev-air/venv/lib/python3.12/site-packages/llvm-aie
ifneq ($(wildcard $(PYTHOC_LLAMA_PEANO)/bin/clang++),)
  override PEANO_INSTALL_DIR := $(PYTHOC_LLAMA_PEANO)
  export PEANO_INSTALL_DIR
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct

# WEIGHTS=hf (default)        - load real Meta weights from HuggingFace
# WEIGHTS=synthetic           - deterministic random weights (no HF, for CI)
WEIGHTS ?= hf
QUANT ?= bf16
AWQ_WEIGHTS ?=
# AWQ_DECODE_EXPERIMENTAL default flips with QUANT: NPU AWQ decode is the
# default for --quant awq now that Phase 6 lands the PythoC awq_mv kernel
# + placed-IRON o_gemv_ffn_awq builder. Set =0 to fall back to the legacy
# CPU-dequant -> BF16-kernels path.
ifeq ($(QUANT),awq)
  AWQ_DECODE_EXPERIMENTAL ?= 1
else
  AWQ_DECODE_EXPERIMENTAL ?= 0
endif

# HF repo (or local dir) to load weights+tokenizer from. Defaults to the
# unsloth non-gated mirror so `make run` and `make hf-gate` work out of
# the box without HF auth. Override with HF_MODEL_ID=meta-llama/...
HF_MODEL_ID ?= unsloth/Llama-3.2-1B-Instruct

# Quote Make variables for safe shell forwarding. PROMPT is intentionally left
# inside double quotes in recipes so normal make run PROMPT="..." remains tidy.
AWQ_WEIGHTS_ABS := $(shell python3 -c 'import os,sys; p=sys.argv[1]; print(os.path.abspath(p) if p else "")' '$(AWQ_WEIGHTS)')
AWQ_WEIGHTS_Q := $(shell python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' '$(AWQ_WEIGHTS_ABS)')
HF_MODEL_ID_Q := $(shell python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' '$(HF_MODEL_ID)')

ifeq ($(WEIGHTS),synthetic)
  WEIGHTS_FLAG := --synthetic-weights
else
  WEIGHTS_FLAG :=
endif
ifeq ($(QUANT),awq)
  QUANT_FLAG := --quant awq --awq-weights $(AWQ_WEIGHTS_Q)
else
  QUANT_FLAG := --quant bf16
endif
ifneq ($(HF_MODEL_ID),)
ifeq ($(QUANT),awq)
  # --hf-model-id is rejected when --quant awq (tokenizer is loaded from
  # the repacked AWQ dir).  Suppress the flag so `make profile-awq` etc.
  # work with the default HF_MODEL_ID still set.
  HF_MODEL_FLAG :=
else
  HF_MODEL_FLAG := --hf-model-id $(HF_MODEL_ID_Q)
endif
else
  HF_MODEL_FLAG :=
endif
ifeq ($(AWQ_DECODE_EXPERIMENTAL),1)
  AWQ_DECODE_EXPERIMENTAL_FLAG := --awq-decode-experimental
else
  AWQ_DECODE_EXPERIMENTAL_FLAG :=
endif

PREFILL_MANIFEST := $(BUILD_DIR)/prefill_kernel_cache/manifest.json
DECODE_MANIFEST  := $(BUILD_DIR)/decode_kernel_cache/manifest.json

.PHONY: help compile ensure-cache run profile verify chat trace check-awq-weights run-awq run-awq-experimental profile-awq verify-awq test-awq-gemv test-awq-gemv-npu snapshot hf-gate clean

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Inference on NPU2 (MLIR-AIE port)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           With CPU reference verification"
	@echo "  make run-awq          Run AIE/NPU pipeline with repacked AWQ weights (NPU AWQ decode is default for QUANT=awq)"
	@echo "  make run-awq-experimental  [alias] same as run-awq with AWQ_DECODE_EXPERIMENTAL=1 (default for QUANT=awq)"
	@echo "  make profile-awq      Profile AIE/NPU pipeline with repacked AWQ weights"
	@echo "  make verify-awq       Verify AIE/NPU pipeline with repacked AWQ weights"
	@echo ""
	@echo "Phase metrics:"
	@echo "  make snapshot         Capture a per-phase JSON snapshot + diff vs latest"
	@echo "  make hf-gate          Real-HF-weights answer correctness gate"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens (instruct model stops early on EOT)"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  WEIGHTS=hf|synthetic  Weight source (default: hf)"
	@echo "  QUANT=bf16|awq        Weight format (default: bf16)"
	@echo "  AWQ_WEIGHTS=/path     Repacked AWQ model dir for QUANT=awq"
	@echo "  HF_MODEL_ID=repo|dir  Override BF16 HF repo/local dir"
	@echo "  AWQ_DECODE_EXPERIMENTAL=0|1  NPU AWQ decode projections (defaults to 1 for QUANT=awq, 0 otherwise)"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=50"
	@echo "  make run MODEL=base PROMPT=\"The capital of France is\" N_TOKENS=200"
	@echo "  make profile PROMPT=\"How does photosynthesis work?\""
	@echo "  make run-awq AWQ_WEIGHTS=./awq_repacked PROMPT=\"Hello\" N_TOKENS=5"
	@echo "  make verify N_TOKENS=10"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

compile: | $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py --compile-only $(QUANT_FLAG)

ensure-cache: | $(BUILD_DIR)
	@if [ ! -f "$(PREFILL_MANIFEST)" ] || [ ! -f "$(DECODE_MANIFEST)" ]; then \
		echo "Kernel cache missing; running make compile first..."; \
		$(MAKE) compile; \
	fi

run: ensure-cache
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG) $(QUANT_FLAG) $(AWQ_DECODE_EXPERIMENTAL_FLAG)

profile: ensure-cache
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG) $(QUANT_FLAG) $(AWQ_DECODE_EXPERIMENTAL_FLAG)

verify: ensure-cache
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --verify --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG) $(QUANT_FLAG) $(AWQ_DECODE_EXPERIMENTAL_FLAG)

check-awq-weights:
	@if [ -z "$(AWQ_WEIGHTS)" ]; then \
		echo "ERROR: AWQ_WEIGHTS is required for AWQ targets"; \
		exit 2; \
	fi
	@if [ ! -d "$(AWQ_WEIGHTS)" ]; then \
		echo "ERROR: AWQ_WEIGHTS must point to an existing repacked AWQ directory: $(AWQ_WEIGHTS)"; \
		exit 2; \
	fi

run-awq: check-awq-weights
	$(MAKE) run QUANT=awq WEIGHTS=hf

run-awq-experimental: check-awq-weights
	$(MAKE) run QUANT=awq WEIGHTS=hf AWQ_DECODE_EXPERIMENTAL=1

profile-awq: check-awq-weights
	$(MAKE) profile QUANT=awq WEIGHTS=hf

verify-awq: check-awq-weights
	$(MAKE) verify QUANT=awq WEIGHTS=hf

test-awq-gemv:
	python3 -m pytest tools/test_awq_gemv*.py -v

test-awq-gemv-npu:
	RUN_AWQ_NPU_SMOKE=1 python3 -m pytest tools/test_awq_gemv_npu_smoke.py -v

chat: ensure-cache
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG) $(QUANT_FLAG) $(AWQ_DECODE_EXPERIMENTAL_FLAG)

# AIE event-trace capture: instrument exactly one decode kernel with
# aie.utils.trace.configure_trace/start_trace before aiecc, then run inference
# and dump trace bytes to <BUILD>/decode_kernel_cache/trace/.
# Required: KERNEL, SUBDEVICE, COL, ROW. Example:
#   make trace KERNEL=rms_gemv_rope SUBDEVICE=v_matvec_bf16_0 COL=0 ROW=2 \
#       N_TOKENS=2 PROMPT=hi
TRACE_SIZE ?= 8388608
TRACE_TARGET := $(KERNEL):$(SUBDEVICE):$(COL):$(ROW)

trace: ensure-cache
	@if [ -z "$(KERNEL)" ] || [ -z "$(SUBDEVICE)" ] || [ -z "$(COL)" ] || [ -z "$(ROW)" ]; then \
	    echo "ERROR: set KERNEL, SUBDEVICE, COL, ROW. e.g. make trace KERNEL=rms_gemv_rope SUBDEVICE=v_matvec_bf16_0 COL=0 ROW=2"; \
	    exit 2; \
	fi
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL) \
		--trace $(TRACE_TARGET) --trace-size $(TRACE_SIZE) \
		$(WEIGHTS_FLAG) $(HF_MODEL_FLAG) $(QUANT_FLAG) $(AWQ_DECODE_EXPERIMENTAL_FLAG)

# ---------------------------------------------------------------------------
# Per-phase metrics harness (tests/test_phase_snapshot.py).
# Captures a JSON snapshot of compile times, per-token decode latency, K/V
# corr, logits corr, and the generated token sequence. Then diffs against
# the most recent prior snapshot under tests/snapshots/.
SNAPSHOT_N_TOKENS ?= 10
SNAPSHOT_PROMPT   ?= What is the capital of France?
snapshot: ensure-cache
	python3 $(srcdir)/tests/test_phase_snapshot.py capture \
		--n-tokens $(SNAPSHOT_N_TOKENS) --prompt "$(SNAPSHOT_PROMPT)"

# Gold-standard answer-level correctness gate (real HF weights).
# Auto-skips with a clear message if the HF cache isn't populated.
# The gate reads HF_MODEL_ID from the environment (default
# unsloth/Llama-3.2-1B-Instruct) so it works without HF auth.
hf-gate: ensure-cache
	HF_MODEL_ID=$(HF_MODEL_ID_Q) HF_GATE_QUANT=$(QUANT) HF_GATE_AWQ_WEIGHTS=$(AWQ_WEIGHTS_Q) python3 -m pytest $(srcdir)/tests/test_hf_answer_gate.py -v -s

all: compile profile

clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	@echo "Build directory removed. Run 'make compile' to rebuild."
