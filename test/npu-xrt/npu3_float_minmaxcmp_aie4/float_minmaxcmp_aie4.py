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
from aie.dialects import vector as vector_d
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

# Exercise IEEE source semantics on a 512-bit FP32 vector.
VEC_LANES = 16

# Inputs A/B and 6 f32 outputs per lane: minimumf, maximumf, cmpf(oge) (as
# 1.0f/0.0f), cmpf(olt) (as 1.0f/0.0f), minnumf, maxnumf.
N_OUT_KINDS = 6


def design(dev):
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        v16f32 = VectorType.get([VEC_LANES], f32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        def splat_f32(value):
            return arith_d.constant(
                value=DenseElementsAttr.get_splat(v16f32, FloatAttr.get(f32, value)),
                result=v16f32,
            )

        A_ty = np.ndarray[(VEC_LANES,), np.dtype[np.float32]]
        B_ty = np.ndarray[(VEC_LANES,), np.dtype[np.float32]]
        C_ty = np.ndarray[(VEC_LANES * N_OUT_KINDS,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)

        inA = object_fifo("inA", shim, core_tile, 2, A_ty)
        inB = object_fifo("inB", shim, core_tile, 2, B_ty)
        outC = object_fifo("outC", core_tile, shim, 2, C_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                ea = inA.acquire(ObjectFifoPort.Consume, 1)
                eb = inB.acquire(ObjectFifoPort.Consume, 1)
                ec = outC.acquire(ObjectFifoPort.Produce, 1)

                pad = arith_d.constant(f32, 0.0)
                c0 = arith_d.constant(IndexType.get(), 0)
                va = vector_d.transfer_read(v16f32, ea, [c0], id_map, pad, [True])
                vb = vector_d.transfer_read(v16f32, eb, [c0], id_map, pad, [True])

                # IEEE min/max use integer encodings to preserve NaNs and signed zero.
                r_min = arith_d.minimumf(va, vb)
                r_max = arith_d.maximumf(va, vb)

                # Ordered comparisons also use integer encodings.
                m_oge = arith_d.cmpf(arith_d.CmpFPredicate.OGE, va, vb)
                m_olt = arith_d.cmpf(arith_d.CmpFPredicate.OLT, va, vb)
                one = splat_f32(1.0)
                zero = splat_f32(0.0)
                r_oge_f = arith_d.select(m_oge, one, zero)
                r_olt_f = arith_d.select(m_olt, one, zero)

                # Number semantics select the numeric operand when one is NaN.
                r_minnum = arith_d.minnumf(va, vb)
                r_maxnum = arith_d.maxnumf(va, vb)

                for i, v in enumerate(
                    [r_min, r_max, r_oge_f, r_olt_f, r_minnum, r_maxnum]
                ):
                    off = arith_d.constant(IndexType.get(), i * VEC_LANES)
                    vector_d.transfer_write(None, v, ec, [off], id_map, [True])

                inA.release(ObjectFifoPort.Consume, 1)
                inB.release(ObjectFifoPort.Consume, 1)
                outC.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(A_ty, B_ty, C_ty)
        def sequence(A, B, C):
            a_task = shim_dma_single_bd_task(
                inA, A, sizes=[1, 1, 1, VEC_LANES], issue_token=True
            )
            b_task = shim_dma_single_bd_task(
                inB, B, sizes=[1, 1, 1, VEC_LANES], issue_token=True
            )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, VEC_LANES * N_OUT_KINDS], issue_token=True
            )
            dma_start_task(a_task, b_task, c_task)
            dma_await_task(a_task, b_task, c_task)


parser = argparse.ArgumentParser(
    description="AIE4 fp32 min/max/cmp/minnum/maxnum (slice F4/F5)"
)
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
