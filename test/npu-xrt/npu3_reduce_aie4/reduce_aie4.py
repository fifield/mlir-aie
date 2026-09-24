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
    Attribute,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    Operation,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# One AIE4 vector register's worth of lanes, at 32 bits/lane.
VEC_LANES = 32

# The 12 i32 outputs compare source reductions with explicit native adds;
# entries 6-8 and 11 contain bitcast f32 results.

kIntSigned = [
    -77, 3, 12, -5, 88, -1, 45, -33, 7, 19, -61, 2, 5, -8, 99, -100,
    17, -17, 23, -23, 31, -31, 42, -42, 50, -50, 63, -63, 71, -71, 6, -6,
]
assert len(kIntSigned) == VEC_LANES

kIntSatPos = [10] + [0] * (VEC_LANES - 1)
kIntSatNeg = [-10] + [0] * (VEC_LANES - 1)

# These seeds distinguish wrapping source add from saturating VADDR.
kBigPosSeed = 2147483647 - 5
kBigNegSeed = -2147483648 + 5
kSmallSeed = 1000

# Each sequential f32 addition of 1 to 2^24 rounds back to 2^24.
# Native VADDR.f sums in wider precision and rounds the total to 16777248.
kF32Precision = [16777216.0] + [1.0] * (VEC_LANES - 1)

kF32MinMax = [
    -3.5, 1.25, 12.25, -0.75, 8.5, -11.125, 4.0, -2.25,
    6.75, -9.5, 2.5, -1.75, 10.0, -6.25, 0.125, -0.125,
    3.25, -3.25, 5.5, -5.5, 7.25, -7.25, 9.75, -9.75,
    1.5, -1.5, 2.75, -2.75, 4.5, -4.5, 6.0, -6.0,
]
assert len(kF32MinMax) == VEC_LANES



def reduce_design(dev):
    f32 = F32Type.get()
    i32 = IntegerType.get_signless(32)

    @device(dev)
    def device_body():
        v32i32 = VectorType.get([VEC_LANES], i32)
        v32f32 = VectorType.get([VEC_LANES], f32)

        def const_i32_vec(values):
            return arith_d.constant(
                value=DenseElementsAttr.get(
                    np.array(values, dtype=np.int32), type=v32i32
                ),
                result=v32i32,
            )

        def const_f32_vec(values):
            return arith_d.constant(
                value=DenseElementsAttr.get(
                    np.array(values, dtype=np.float32), type=v32f32
                ),
                result=v32f32,
            )

        def const_i32(value):
            return arith_d.constant(value=IntegerAttr.get(i32, value), result=i32)

        def const_f32(value):
            return arith_d.constant(value=FloatAttr.get(f32, value), result=f32)

        native_add_kind = Attribute.parse("#aievec.aie4_reduce_kind<add>")

        def native_add(result_type, source, seed=None):
            operands = [source] if seed is None else [source, seed]
            return Operation.create(
                "aievec.aie4.reduce",
                results=[result_type],
                operands=operands,
                attributes={"kind": native_add_kind},
            ).results[0]

        shim = tile(0, 0)
        core_tile = tile(0, 2)

        # Pack float bits into i32 outputs to use one output FIFO.
        out_ty = np.ndarray[(12,), np.dtype[np.int32]]
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)

                vIntSigned = const_i32_vec(kIntSigned)
                vIntSatPos = const_i32_vec(kIntSatPos)
                vIntSatNeg = const_i32_vec(kIntSatNeg)
                vF32Precision = const_f32_vec(kF32Precision)
                vF32MinMax = const_f32_vec(kF32MinMax)

                add_kind = vector_d.CombiningKind.ADD
                minsi_kind = vector_d.CombiningKind.MINSI
                maxsi_kind = vector_d.CombiningKind.MAXSI
                minf_kind = vector_d.CombiningKind.MINIMUMF
                maxf_kind = vector_d.CombiningKind.MAXIMUMF

                r_add = vector_d.reduction(i32, add_kind, vIntSigned)
                r_min = vector_d.reduction(i32, minsi_kind, vIntSigned)
                r_max = vector_d.reduction(i32, maxsi_kind, vIntSigned)
                r_addsat_pos = vector_d.reduction(
                    i32, add_kind, vIntSatPos, acc=const_i32(kBigPosSeed)
                )
                r_addsat_neg = vector_d.reduction(
                    i32, add_kind, vIntSatNeg, acc=const_i32(kBigNegSeed)
                )
                r_add_seeded = vector_d.reduction(
                    i32, add_kind, vIntSigned, acc=const_i32(kSmallSeed)
                )
                n_addsat_pos = native_add(
                    i32, vIntSatPos, const_i32(kBigPosSeed)
                )
                n_addsat_neg = native_add(
                    i32, vIntSatNeg, const_i32(kBigNegSeed)
                )

                f_add = vector_d.reduction(f32, add_kind, vF32Precision)
                n_f_add = native_add(f32, vF32Precision)
                f_min = vector_d.reduction(f32, minf_kind, vF32MinMax)
                f_max = vector_d.reduction(f32, maxf_kind, vF32MinMax)

                f_add_bits = arith_d.bitcast(i32, f_add)
                n_f_add_bits = arith_d.bitcast(i32, n_f_add)
                f_min_bits = arith_d.bitcast(i32, f_min)
                f_max_bits = arith_d.bitcast(i32, f_max)

                vOut = vector_d.from_elements(
                    VectorType.get([12], i32),
                    [
                        r_add, r_min, r_max, r_addsat_pos, r_addsat_neg,
                        r_add_seeded, f_add_bits, f_min_bits, f_max_bits,
                        n_addsat_pos, n_addsat_neg, n_f_add_bits,
                    ],
                )

                c0 = arith_d.constant(IndexType.get(), 0)
                vector_d.store(vOut, elem_out, [c0])

                out_fifo.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(out_ty)
        def sequence(out):
            o_task = shim_dma_single_bd_task(
                out_fifo, out, sizes=[1, 1, 1, 12], issue_token=True
            )
            dma_start_task(o_task)
            dma_await_task(o_task)


parser = argparse.ArgumentParser(description="AIE4 vector.reduction (add/min/max)")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    reduce_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
