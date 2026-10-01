#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

# Row softmax using vector/arith/math over 32-lane chunks. Carry the maximum
# and exponential sum across chunks; afn allows approximate exp2 and reciprocal,
# and reassoc allows the sum reduction.

import argparse
import math
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import math as math_d
from aie.dialects import memref as memref_d
from aie.dialects import vector as vector_d
from aie.dialects.scf import yield_
from aie.ir import (
    AffineMap,
    AffineMapAttr,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    IndexType,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

VEC_LANES = 32
# Row lengths in each set; must match test.cpp.
ROW_LENGTHS = [32, 64, 96, 160, 256]
SET_ELEMS = sum(ROW_LENGTHS)
# Per row: max, sum, reciprocal, pad.
STATS = 4
OUT_ELEMS = SET_ELEMS + STATS * len(ROW_LENGTHS)
N_SETS = 64
LOG2E = 1.0 / math.log(2.0)


def softmax_design(dev):
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([VEC_LANES], f32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))
        afn = arith_d.FastMathFlags.afn
        reassoc = arith_d.FastMathFlags.reassoc

        in_ty = np.ndarray[(SET_ELEMS,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_ELEMS,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_fifo = object_fifo("in", shim, core_tile, 2, in_ty)
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        def idx(v):
            return arith_d.constant(IndexType.get(), v)

        def softmax_row(src, dst, base, n, stats_base):
            pad = arith_d.constant(f32, 0.0)

            def read(buf, k):
                i = arith_d.addi(k, idx(base))
                return vector_d.transfer_read(v32f32, buf, [i], id_map, pad, [True])

            def write(val, buf, k):
                i = arith_d.addi(k, idx(base))
                vector_d.transfer_write(None, val, buf, [i], id_map, [True])

            # Row maximum over all chunks.
            neg_inf = arith_d.constant(f32, float("-inf"))
            for k, m, m_out in range_(
                0, n, VEC_LANES, iter_args=[neg_inf], insert_yield=False
            ):
                m_next = vector_d.reduction(
                    f32, vector_d.CombiningKind.MAXIMUMF, read(src, k), acc=m
                )
                yield_([m_next])
            row_max = m_out

            # Exponentials, stored, and their sum over all chunks.
            zero = arith_d.constant(f32, 0.0)
            vmax = vector_d.broadcast(v32f32, row_max)
            vlog2e = arith_d.constant(
                v32f32,
                DenseElementsAttr.get_splat(v32f32, FloatAttr.get(f32, LOG2E)),
            )
            for k, s, s_out in range_(
                0, n, VEC_LANES, iter_args=[zero], insert_yield=False
            ):
                t = arith_d.mulf(arith_d.subf(read(src, k), vmax), vlog2e)
                e = math_d.exp2(t, fastmath=afn)
                write(e, dst, k)
                s_next = vector_d.reduction(
                    f32, vector_d.CombiningKind.ADD, e, acc=s, fastmath=reassoc
                )
                yield_([s_next])
            row_sum = s_out

            # Normalize every exponential by the row's reciprocal sum.
            one = arith_d.constant(f32, 1.0)
            recip = arith_d.divf(one, row_sum, fastmath=afn)
            vrecip = vector_d.broadcast(v32f32, recip)
            for k in range_(0, n, VEC_LANES):
                write(arith_d.mulf(read(dst, k), vrecip), dst, k)

            memref_d.store(row_max, dst, [idx(stats_base)])
            memref_d.store(row_sum, dst, [idx(stats_base + 1)])
            memref_d.store(recip, dst, [idx(stats_base + 2)])

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                for _ in range_(N_SETS):
                    elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                    base = 0
                    for r, n in enumerate(ROW_LENGTHS):
                        softmax_row(elem_in, elem_out, base, n, SET_ELEMS + STATS * r)
                        base += n
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

        total_in = SET_ELEMS * N_SETS
        total_out = OUT_ELEMS * N_SETS

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


parser = argparse.ArgumentParser(description="AIE4 row softmax")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    softmax_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
