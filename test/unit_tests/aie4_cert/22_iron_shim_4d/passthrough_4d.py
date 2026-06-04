#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

# IRON / cert end-to-end coverage for the aiex Python binding
# shim_dma_single_bd_task driving a genuine 4D (D0..D3) shim read.
#
# The existing iron cert tests (08/09/10) only call the helper with contiguous
# sizes=[1,1,1,N], so they never exercise D2/D3 addressing. Here the input is
# read with a single-transfer four-data-dimension access pattern (outermost
# iteration size 1, i.e. no repeat):
#
#   sizes   = [1, 2, 2, 2, 8]    # [iteration, d3, d2, d1, d0]
#   strides = [0, 32, 8, 16, 1]  # d0 contiguous (AIE4 rule); d1..d3 non-collapsible
#
# This is a bijection of [0,64) using all four data dimensions, so the BD lowers
# with d0_size=8, d1/d2/d3_size=2 and exercises the AIE4 shim D3 dimension. The
# passthrough (Shim -> MemTile -> Core copy -> MemTile -> Shim) writes the
# streamed elements back in order, so the output is a known permutation of the
# input, checked element-by-element.
#
# NOTE: this deliberately uses a single transfer (iteration size 1, NO repeat).
# A strided OUTER/iteration dimension is intentionally not used here: through the
# objectfifo helper a self-repeating task does not compose with the fifo's
# object/token model, so shim_dma_single_bd_task rejects a strided outer dim
# (see test/python/shim_dma_task_repeat.py). Strided/advancing iteration is
# covered on the raw dma_configure_task path by 16_shim_dma_4d.

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

N = 64

# [iteration, d3, d2, d1, d0]; iteration size 1 == single transfer (no repeat)
SIZES = [1, 2, 2, 2, 8]
STRIDES = [0, 32, 8, 16, 1]


def passthrough_4d(dev):
    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(N,), np.dtype[np.int32]]

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile = tile(0, 2)

        # Input: Shim -> MemTile -> Core
        inA = object_fifo("inA", ShimTile, MemTile, 2, tile_ty)
        memA = object_fifo("memA", MemTile, ComputeTile, 2, tile_ty)
        object_fifo_link(inA, [memA], [], [])

        # Output: Core -> MemTile -> Shim
        memC = object_fifo("memC", ComputeTile, MemTile, 2, tile_ty)
        outC = object_fifo("outC", MemTile, ShimTile, 2, tile_ty)
        object_fifo_link([memC], outC, [], [])

        @core(ComputeTile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                elem_in = memA.acquire(ObjectFifoPort.Consume, 1)
                for j in range_(N):
                    elem_out[j] = elem_in[j]
                memA.release(ObjectFifoPort.Consume, 1)
                memC.release(ObjectFifoPort.Produce, 1)

        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):
            a_task = shim_dma_single_bd_task(
                inA, A, sizes=SIZES, strides=STRIDES, issue_token=True
            )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, N], issue_token=True
            )
            dma_start_task(a_task, c_task)
            dma_await_task(a_task, c_task)


parser = argparse.ArgumentParser(description="4D single-transfer strided passthrough")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    passthrough_4d(AIEDevice.npu3z)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
