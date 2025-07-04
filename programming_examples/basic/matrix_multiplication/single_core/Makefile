##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=single_core
targetname=single_core
kernels=mm

# M, K, N: final GEMM dimensions
M?=512
K?=512
N?=512

# m, k, n: single AIE core GEMM dimensions
m?=32
k?=32
n?=32

b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}
use_placed?=0
use_iron?=0

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
ifeq (${use_iron}, 1)
$(error Cannot specify both placed design and IRON)
endif
endif

ifeq (${use_iron}, 1)
aie_py_src=${targetname}_iron.py
endif

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n}
ifeq (${b_col_maj}, 1)
	KERNEL_DEFINES+=-DB_COL_MAJ
endif

include ${srcdir}/../makefile-common

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
