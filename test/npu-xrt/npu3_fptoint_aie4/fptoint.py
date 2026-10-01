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
from aie.dialects import vector as vector_d
from aie.ir import AffineMap, AffineMapAttr, F32Type, IntegerType, VectorType
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

LANES = 32

# Per 32-lane chunk, in 32-bit words.
# Input:  x (f32)
# Output: fptosi(x) | fptoui(x) | crRnd, then pad
IN_WORDS = LANES
OUT_WORDS = 3 * LANES

CHUNKS_PER_TILE = 4
N_TILES = 16


def fptoint_design(dev):
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([LANES], f32)
        v32i32 = VectorType.get([LANES], i32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        in_ty = np.ndarray[(IN_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_WORDS * CHUNKS_PER_TILE,), np.dtype[np.int32]]

        # poison.o's main() sets crRnd to nearest even before the core runs;
        # get_rnd reports crRnd as the core sees it.
        get_rnd = external_func(
            "get_rnd", inputs=[], outputs=[np.int32], link_with="poison.o"
        )

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

                        def out_at(k, val):
                            idx = arith_d.addi(
                                cout, arith_d.constant(c.type, k * LANES)
                            )
                            vector_d.transfer_write(
                                None, val, elem_out, [idx], id_map, [True]
                            )

                        x = vector_d.transfer_read(
                            v32f32, elem_in, [cin], id_map, pad, [True]
                        )
                        out_at(0, arith_d.fptosi(v32i32, x))
                        out_at(1, arith_d.fptoui(v32i32, x))
                        idx = arith_d.addi(cout, arith_d.constant(c.type, 2 * LANES))
                        memref_d.store(get_rnd(), elem_out, [idx])
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

        total_in = IN_WORDS * CHUNKS_PER_TILE * N_TILES
        total_out = OUT_WORDS * CHUNKS_PER_TILE * N_TILES

        @runtime_sequence(
            np.ndarray[(total_in,), np.dtype[np.float32]],
            np.ndarray[(total_out,), np.dtype[np.int32]],
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


parser = argparse.ArgumentParser(description="AIE4 fptosi/fptoui")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    fptoint_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
