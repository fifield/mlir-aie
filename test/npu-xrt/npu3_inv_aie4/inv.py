#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import memref as memref_d
from aie.ir import F32Type, FloatAttr
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Each output block holds TILE afn reciprocals followed by TILE strict ones.
TILE = 1024
N_TILES = 16


def inv_design(dev):
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(TILE,), np.dtype[np.float32]]
        out_ty = np.ndarray[(2 * TILE,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_fifo = object_fifo("in", shim, core_tile, 2, in_ty)
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(N_TILES):
                    elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(TILE):
                        one = arith_d.constant(f32, 1.0)
                        x = memref_d.load(elem_in, [i])
                        approx = arith_d.divf(
                            one, x, fastmath=arith_d.FastMathFlags.afn
                        )
                        exact = arith_d.divf(one, x)
                        memref_d.store(approx, elem_out, [i])
                        j = arith_d.addi(i, arith_d.constant(i.type, TILE))
                        memref_d.store(exact, elem_out, [j])
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

        total_in = TILE * N_TILES
        total_out = 2 * TILE * N_TILES

        @runtime_sequence(
            np.ndarray[(total_in,), np.dtype[np.float32]],
            np.ndarray[(total_out,), np.dtype[np.float32]],
        )
        def sequence(A, C):
            a_task = shim_dma_single_bd_task(
                in_fifo, A, sizes=[1, 1, 1, total_in], issue_token=True
            )
            c_task = shim_dma_single_bd_task(
                out_fifo, C, sizes=[1, 1, 1, total_out], issue_token=True
            )
            dma_start_task(a_task, c_task)
            dma_await_task(a_task, c_task)


parser = argparse.ArgumentParser(description="AIE4 afn reciprocal accuracy")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    inv_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
