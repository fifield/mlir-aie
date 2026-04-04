#
# GEMM-based Conv1x1 for AIE2P
# Uses aie::mmul<8,8,8> with 2x2 register blocking
#

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

targetname = gemm_conv1x1_bf16
mlirFileName = aie

# Design parameters
TILE_M ?= 64
IC ?= 128
OC ?= 64
N_CORES ?= 32
PATCHES_PER_CORE ?= 1

devicename = npu2

VPATH := ${srcdir}

all: build/gemm_conv1x1_bf16.o build/final.xclbin

# Compile kernel with Peano
build/gemm_conv1x1_bf16.o: gemm_conv1x1_bf16.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c ${srcdir}/$< -o ${@F}

# Generate MLIR from IRON program
build/aie.mlir: ${srcdir}/aie2_gemm_conv1x1.py
	mkdir -p ${@D}
	python3 $< $(N_CORES) $(TILE_M) $(IC) $(OC) $(PATCHES_PER_CORE) > $@

# Build xclbin
build/final.xclbin: build/aie.mlir build/gemm_conv1x1_bf16.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

# Pure GEMM (no BN/SiLU) variant
build/aie_nofuse.mlir: ${srcdir}/aie2_gemm_conv1x1.py
	mkdir -p ${@D}
	python3 $< $(N_CORES) $(TILE_M) $(IC) $(OC) $(PATCHES_PER_CORE) --no-fuse > $@

build/final_nofuse.xclbin: build/aie_nofuse.mlir build/gemm_conv1x1_bf16.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts_nofuse.bin $(<:%=../%)

# CPU-only reference test
test:
	python3 ${srcdir}/test_gemm_conv1x1.py --tile-m $(TILE_M) --ic $(IC) --oc $(OC)

test-nofuse:
	python3 ${srcdir}/test_gemm_conv1x1.py --tile-m $(TILE_M) --ic $(IC) --oc $(OC) --no-fuse

# NPU hardware test (fused BN+SiLU)
run: build/final.xclbin
	python3 ${srcdir}/test_gemm_conv1x1.py --tile-m $(TILE_M) --ic $(IC) --oc $(OC) \
		--n-cores $(N_CORES) --patches-per-core $(PATCHES_PER_CORE) \
		-x build/final.xclbin -i build/insts.bin

# NPU hardware test (pure GEMM)
run-nofuse: build/final_nofuse.xclbin
	python3 ${srcdir}/test_gemm_conv1x1.py --tile-m $(TILE_M) --ic $(IC) --oc $(OC) \
		--n-cores $(N_CORES) --patches-per-core $(PATCHES_PER_CORE) --no-fuse \
		-x build/final_nofuse.xclbin -i build/insts_nofuse.bin

# --- Example configurations for MDV6 model layers ---

# RepNCSPELAN internal 1x1: 80x80 spatial, 128->64 channels
# 80*80=6400 pixels. With 32 cores, tile_m=64, patches_per_core=~3
# (6400 / 32 = 200 pixels/core = 3.125 patches -> 3 patches + remainder)
run-repncsp-1x1:
	make clean && make run TILE_M=64 IC=128 OC=64 N_CORES=32 PATCHES_PER_CORE=1

# ELAN internal 1x1: 320x320 spatial, 32->32 channels
run-elan-1x1:
	make clean && make run TILE_M=64 IC=32 OC=32 N_CORES=32 PATCHES_PER_CORE=1

# Deeper 1x1: 20x20 spatial, 256->256 channels
# 400 pixels, 32 cores: only 12 pixels/core. Use fewer cores.
run-deep-1x1:
	make clean && make run TILE_M=64 IC=256 OC=256 N_CORES=8 PATCHES_PER_CORE=1

clean:
	rm -rf build *.elf* *.lst *.bif aie.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin *.log aie_partition.json *.bin BOOT.BIN _x

.PHONY: all test test-nofuse run run-nofuse clean \
        run-repncsp-1x1 run-elan-1x1 run-deep-1x1
