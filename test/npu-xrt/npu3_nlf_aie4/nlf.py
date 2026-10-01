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
from aie.dialects import math as math_d
from aie.dialects import vector as vector_d
from aie.ir import AffineMap, AffineMapAttr, F32Type, VectorType
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

LANES = 32

# Per 32-lane chunk, in f32 words.
# Input:  x | a | b
# Output: rsqrt_afn(x) | a / b with arcp+afn | a / b strict
IN_WORDS = 3 * LANES
OUT_WORDS = 3 * LANES

CHUNKS_PER_TILE = 4
N_TILES = 64


def nlf_design(dev):
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([LANES], f32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))
        afn = arith_d.FastMathFlags.afn
        arcp_afn = arith_d.FastMathFlags.arcp | arith_d.FastMathFlags.afn

        in_ty = np.ndarray[(IN_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_fifo = object_fifo("in", shim, core_tile, 2, in_ty)
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        # The strict division calls compiler-rt's __divsf3, which carries no
        # stack size, and the default 1 KiB is not enough with it.
        @core(core_tile, stack_size=0x1000)
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(N_TILES):
                    elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                    for c in range_(CHUNKS_PER_TILE):
                        pad = arith_d.constant(f32, 0.0)
                        cin = arith_d.muli(c, arith_d.constant(c.type, IN_WORDS))
                        cout = arith_d.muli(c, arith_d.constant(c.type, OUT_WORDS))

                        def in_at(k):
                            idx = arith_d.addi(cin, arith_d.constant(c.type, k * LANES))
                            return vector_d.transfer_read(
                                v32f32, elem_in, [idx], id_map, pad, [True]
                            )

                        def out_at(k, val):
                            idx = arith_d.addi(
                                cout, arith_d.constant(c.type, k * LANES)
                            )
                            vector_d.transfer_write(
                                None, val, elem_out, [idx], id_map, [True]
                            )

                        x, a, b = in_at(0), in_at(1), in_at(2)
                        out_at(0, math_d.rsqrt(x, fastmath=afn))
                        out_at(1, arith_d.divf(a, b, fastmath=arcp_afn))
                        out_at(2, arith_d.divf(a, b))
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

        total_in = IN_WORDS * CHUNKS_PER_TILE * N_TILES
        total_out = OUT_WORDS * CHUNKS_PER_TILE * N_TILES

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


parser = argparse.ArgumentParser(description="AIE4 rsqrt and division")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    nlf_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
