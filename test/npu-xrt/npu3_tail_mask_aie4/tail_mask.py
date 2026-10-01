#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

# Masked f32 add and maximum over 1000 elements in 32-lane chunks, using the
# vector.mask form emitted by linalg vectorization. Also copy a full vector
# between unaligned offsets. Guard words detect writes outside the outputs.

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
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
    InsertionPoint,
    IntegerType,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

N = 1000
LANES = 32
FRONT_GUARD = 32
# 32 guard + 1000 data + 55 guard + 1 reduction result, then a 64-word region
# for a full-mask copy at an unaligned offset.
MAX_SLOT = 1087
UNAL_REGION = 1088
UNAL_SRC = 5  # copy a[5..36] ...
UNAL_DST = UNAL_REGION + 3  # ... to out[1091..1122]
OUT_WORDS = UNAL_REGION + 64
SENTINEL_BITS = 0xDEADBEEF  # -6.3e18; no result of the add is near it
N_ITERS = 4


def tail_mask_design(dev):
    f32 = F32Type.get()
    index = IndexType.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([LANES], f32)
        mask_ty = VectorType.get([LANES], IntegerType.get_signless(1))
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        in_ty = np.ndarray[(N,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_WORDS,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_a = object_fifo("inA", shim, core_tile, 2, in_ty)
        in_b = object_fifo("inB", shim, core_tile, 2, in_ty)
        out = object_fifo("out", core_tile, shim, 2, out_ty)

        def cidx(v):
            return arith_d.constant(index, v)

        def masked_read(mask, buf, i, pad):
            op = vector_d.MaskOp([v32f32], mask)
            block = op.regions[0].blocks.append()
            with InsertionPoint(block):
                r = vector_d.transfer_read(v32f32, buf, [i], id_map, pad, [True])
                vector_d.yield_([r])
            return op.results[0]

        def masked_write(mask, val, buf, i):
            op = vector_d.MaskOp([], mask)
            block = op.regions[0].blocks.append()
            with InsertionPoint(block):
                vector_d.transfer_write(None, val, buf, [i], id_map, [True])
                vector_d.yield_([])

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                a = in_a.acquire(ObjectFifoPort.Consume, 1)
                b = in_b.acquire(ObjectFifoPort.Consume, 1)
                o = out.acquire(ObjectFifoPort.Produce, 1)

                # Fill the whole output, guards included, with the sentinel.
                sentinel = arith_d.constant(
                    v32f32,
                    DenseElementsAttr.get(
                        np.full(LANES, SENTINEL_BITS, dtype=np.uint32).view(np.float32),
                        type=v32f32,
                    ),
                )
                for k in range_(0, OUT_WORDS, LANES):
                    vector_d.transfer_write(None, sentinel, o, [k], id_map, [True])

                # The 1000 results live in a subview after the front guard.
                c = memref_d.subview(o, [FRONT_GUARD], [N], [1])

                # Zero padding exposes unmasked reductions on all-negative inputs.
                pad = arith_d.constant(f32, 0.0)
                neg_inf = arith_d.constant(f32, float("-inf"))
                n_total = cidx(N)
                for i, m, m_out in range_(
                    0, N, LANES, iter_args=[neg_inf], insert_yield=False
                ):
                    remaining = arith_d.subi(n_total, i)
                    n = arith_d.minsi(remaining, cidx(LANES))
                    mask = vector_d.create_mask(mask_ty, [n])
                    x = masked_read(mask, a, i, pad)
                    y = masked_read(mask, b, i, pad)
                    masked_write(mask, arith_d.addf(x, y), c, i)
                    red = vector_d.MaskOp([f32], mask)
                    block = red.regions[0].blocks.append()
                    with InsertionPoint(block):
                        r = vector_d.reduction(
                            f32, vector_d.CombiningKind.MAXIMUMF, x, acc=m
                        )
                        vector_d.yield_([r])
                    yield_([red.results[0]])
                memref_d.store(m_out, o, [cidx(MAX_SLOT)])

                # A full mask at unaligned offsets: a whole-vector access would
                # drop the low address bits and move the wrong 32 words.
                full = vector_d.constant_mask(mask_ty, [LANES])
                zeros = arith_d.constant(
                    v32f32, DenseElementsAttr.get_splat(v32f32, FloatAttr.get(f32, 0.0))
                )
                u = vector_d.maskedload(v32f32, a, [cidx(UNAL_SRC)], full, zeros)
                vector_d.maskedstore(o, [cidx(UNAL_DST)], full, u)

                in_a.release(ObjectFifoPort.Consume, 1)
                in_b.release(ObjectFifoPort.Consume, 1)
                out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(N * N_ITERS,), np.dtype[np.float32]],
            np.ndarray[(N * N_ITERS,), np.dtype[np.float32]],
            np.ndarray[(OUT_WORDS * N_ITERS,), np.dtype[np.float32]],
        )
        def sequence(A, B, C):
            ta = shim_dma_single_bd_task(
                in_a, A, sizes=[1, 1, 1, N * N_ITERS], issue_token=True
            )
            tb = shim_dma_single_bd_task(
                in_b, B, sizes=[1, 1, 1, N * N_ITERS], issue_token=True
            )
            tc = shim_dma_single_bd_task(
                out, C, sizes=[1, 1, 1, OUT_WORDS * N_ITERS], issue_token=True
            )
            dma_start_task(ta, tb, tc)
            dma_await_task(ta, tb, tc)


parser = argparse.ArgumentParser(description="AIE4 masked tail, N = 1000")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    tail_mask_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
