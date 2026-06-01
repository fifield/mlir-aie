# Top-level Makefile for the MDV6 PythoC programming examples.
#
# Run `make help` for a list of targets.
#
# Source env.sh first:  source /home/jfifield/npu-dev-pythoc/env.sh

DST := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

KERNELS_DIR := $(DST)/kernels
CONV_DIR    := $(DST)/conv
GEMM_DIR    := $(DST)/gemm_conv1x1

KERNELS_BUILD := $(KERNELS_DIR)/build
CONV_BUILD    := $(CONV_DIR)/build
GEMM_BUILD    := $(GEMM_DIR)/build

# Single-core per-layer build dirs (created on demand by each <layer>_pythoc.py).
SC_BUILD_DIRS := $(wildcard $(DST)/*/*_pythoc_build)

.PHONY: help all build kernels mc gemm \
        clean clean-mc clean-gemm clean-kernels clean-sc clean-pycache distclean \
        profile profile1 verify-kernels

help:
	@echo "MDV6 PythoC programming examples"
	@echo
	@echo "Build targets:"
	@echo "  all / build         Build 4 unified .o + 48 MC xclbins + 18 GEMM xclbins"
	@echo "  kernels             Build only the 4 unified .o files (rep_elan_bf16 kernels)"
	@echo "  mc                  Build only the 48 MC conv xclbins  (~150 MB)"
	@echo "  gemm                Build only the 18 GEMM xclbins      (~ 65 MB)"
	@echo
	@echo "Run targets:"
	@echo "  profile             Run test_full_model_mc.py --profile 3  (cold + 2 warm)"
	@echo "  profile1            Quick smoke: --profile 1"
	@echo "  verify-kernels      Run kernels/verify_one_xclbin.py --kernel all"
	@echo
	@echo "Clean targets:"
	@echo "  clean               Remove all multicore artifacts (kernels + mc + gemm builds)"
	@echo "  clean-kernels       Remove only kernels/build/*.o"
	@echo "  clean-mc            Remove only conv/build/"
	@echo "  clean-gemm          Remove only gemm_conv1x1/build/"
	@echo "  clean-sc            Remove all single-core per-layer *_pythoc_build/ dirs"
	@echo "  clean-pycache       Remove all __pycache__ trees"
	@echo "  distclean           clean + clean-sc + clean-pycache  (full wipe)"
	@echo
	@echo "Source env.sh first:  source $(abspath $(DST)/../../../../env.sh)"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

all: build
build: kernels mc gemm

kernels:
	@echo ">> Building unified PythoC kernels (.o files)"
	cd $(KERNELS_DIR) && python build_kernels.py

mc: kernels
	@echo ">> Building 48 MC conv xclbins"
	cd $(CONV_DIR) && python build_multicore.py

gemm: kernels
	@echo ">> Building 18 GEMM xclbins"
	cd $(GEMM_DIR) && python build_gemm_conv1x1.py

# -----------------------------------------------------------------------------
# Run / verify
# -----------------------------------------------------------------------------

# Wrap NPU runs with flock so concurrent invocations queue cleanly.
NPU_LOCK := flock /tmp/npu-dev.lock

profile:
	@echo ">> test_full_model_mc.py --profile 3"
	cd $(DST) && $(NPU_LOCK) python test_full_model_mc.py --profile 3

profile1:
	@echo ">> test_full_model_mc.py --profile 1"
	cd $(DST) && $(NPU_LOCK) python test_full_model_mc.py --profile 1

verify-kernels:
	@echo ">> kernels/verify_one_xclbin.py --kernel all"
	cd $(KERNELS_DIR) && $(NPU_LOCK) python verify_one_xclbin.py --kernel all

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

clean: clean-kernels clean-mc clean-gemm

clean-kernels:
	@echo ">> rm $(KERNELS_BUILD)"
	rm -rf $(KERNELS_BUILD)
	rm -rf $(KERNELS_DIR)/verify_build

clean-mc:
	@echo ">> rm $(CONV_BUILD)"
	rm -rf $(CONV_BUILD)

clean-gemm:
	@echo ">> rm $(GEMM_BUILD)"
	rm -rf $(GEMM_BUILD)

clean-sc:
	@echo ">> rm $(DST)/*/*_pythoc_build"
	rm -rf $(DST)/*/*_pythoc_build

clean-pycache:
	@echo ">> rm __pycache__ trees under $(DST)"
	find $(DST) -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean clean-sc clean-pycache
