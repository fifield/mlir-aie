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
from aie.ir import AffineMap, AffineMapAttr, BF16Type, F32Type, VectorType
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

LANES = 32
HALF = LANES // 2

# Per 32-lane chunk, in f32 words (bf16 vectors are packed two per word).
# Input:  x (f32) | xb (bf16) | pad
# Output: exp_afn(x) (f32) | exp_afn(xb) (bf16) | pad
# Chunks are padded to 128 bytes: 1024-bit vector accesses must be aligned.
IN_WORDS = 2 * LANES
OUT_WORDS = 2 * LANES

CHUNKS_PER_TILE = 8
N_TILES = 64


def approx_exp_design(dev):
    f32 = F32Type.get()
    bf16 = BF16Type.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([LANES], f32)
        v16f32 = VectorType.get([HALF], f32)
        v32bf16 = VectorType.get([LANES], bf16)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))
        afn = arith_d.FastMathFlags.afn

        in_ty = np.ndarray[(IN_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]

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
                    for c in range_(CHUNKS_PER_TILE):
                        pad = arith_d.constant(f32, 0.0)
                        cin = arith_d.muli(c, arith_d.constant(c.type, IN_WORDS))
                        cout = arith_d.muli(c, arith_d.constant(c.type, OUT_WORDS))

                        def in_at(off, ty):
                            idx = arith_d.addi(cin, arith_d.constant(c.type, off))
                            return vector_d.transfer_read(
                                ty, elem_in, [idx], id_map, pad, [True]
                            )

                        def out_at(off, val):
                            idx = arith_d.addi(cout, arith_d.constant(c.type, off))
                            vector_d.transfer_write(
                                None, val, elem_out, [idx], id_map, [True]
                            )

                        x = in_at(0, v32f32)
                        xb = vector_d.bitcast(v32bf16, in_at(LANES, v16f32))
                        out_at(0, math_d.exp(x, fastmath=afn))
                        e16 = math_d.exp(xb, fastmath=afn)
                        out_at(LANES, vector_d.bitcast(v16f32, e16))
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


parser = argparse.ArgumentParser(description="AIE4 afn exp accuracy")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    approx_exp_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
