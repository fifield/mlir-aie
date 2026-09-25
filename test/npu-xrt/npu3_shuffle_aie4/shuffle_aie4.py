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
from aie.dialects import aievec as aievec_d
from aie.ir import (
    Attribute,
    DenseElementsAttr,
    IndexType,
    IntegerType,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Exercise all four shuffle modes directly with distinct constant inputs.
# Capture each square transpose and its round trip; capture both interleave
# halves. test.cpp checks every output lane against an independent model.

VEC_LANES_I32 = 16
VEC_LANES_I8 = 64

# Row-major 4x4 i32 matrix with distinct values.
kT32_4x4 = list(range(1000, 1000 + VEC_LANES_I32))
assert len(kT32_4x4) == VEC_LANES_I32

# Row-major 8x8 i8 matrix; nonzero values make zeroed lanes detectable.
kT8_8x8 = list(range(1, 1 + VEC_LANES_I8))
assert len(kT8_8x8) == VEC_LANES_I8
assert max(kT8_8x8) <= 127

# Distinct inputs for the low/high interleaves.
kZipA = list(range(2000, 2000 + VEC_LANES_I32))
kZipB = list(range(3000, 3000 + VEC_LANES_I32))

# Output layout in i32 words; pack each 64xi8 result into 16xi32.
#   [0:16)   4x4 transpose
#   [16:32)  4x4 round trip
#   [32:48)  8x8 transpose
#   [48:64)  8x8 round trip
#   [64:80)  interleave low
#   [80:96)  interleave high
OUT_LANES = 96


def shuffle_mode(name):
    return Attribute.parse(f"#aievec.aie4_shuffle_mode<{name}>")


def shuffle_design(dev):
    i32 = IntegerType.get_signless(32)
    i8 = IntegerType.get_signless(8)

    @device(dev)
    def device_body():
        v16i32 = VectorType.get([VEC_LANES_I32], i32)
        v64i8 = VectorType.get([VEC_LANES_I8], i8)

        def const_i32_vec(values):
            return arith_d.constant(
                value=DenseElementsAttr.get(
                    np.array(values, dtype=np.int32), type=v16i32
                ),
                result=v16i32,
            )

        def const_i8_vec(values):
            return arith_d.constant(
                value=DenseElementsAttr.get(
                    np.array(values, dtype=np.int8), type=v64i8
                ),
                result=v64i8,
            )

        shim = tile(0, 0)
        core_tile = tile(0, 2)

        out_ty = np.ndarray[(OUT_LANES,), np.dtype[np.int32]]
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)

                v4x4 = const_i32_vec(kT32_4x4)
                v8x8 = const_i8_vec(kT8_8x8)
                vZipA = const_i32_vec(kZipA)
                vZipB = const_i32_vec(kZipB)

                t32_4x4 = shuffle_mode("t32_4x4")
                t8_8x8 = shuffle_mode("t8_8x8")
                t32_2x16_lo = shuffle_mode("t32_2x16_lo")
                t32_2x16_hi = shuffle_mode("t32_2x16_hi")

                r4x4 = aievec_d.AIE4ShuffleOp(
                    results=[v16i32], lhs=v4x4, mode=t32_4x4, rhs=None
                )
                r4x4_rt = aievec_d.AIE4ShuffleOp(
                    results=[v16i32], lhs=r4x4.result, mode=t32_4x4, rhs=None
                )

                r8x8 = aievec_d.AIE4ShuffleOp(
                    results=[v64i8], lhs=v8x8, mode=t8_8x8, rhs=None
                )
                r8x8_rt = aievec_d.AIE4ShuffleOp(
                    results=[v64i8], lhs=r8x8.result, mode=t8_8x8, rhs=None
                )

                r8x8_i32 = vector_d.bitcast(v16i32, r8x8.result)
                r8x8_rt_i32 = vector_d.bitcast(v16i32, r8x8_rt.result)

                rzip_lo = aievec_d.AIE4ShuffleOp(
                    results=[v16i32], lhs=vZipA, mode=t32_2x16_lo, rhs=vZipB
                )
                rzip_hi = aievec_d.AIE4ShuffleOp(
                    results=[v16i32], lhs=vZipA, mode=t32_2x16_hi, rhs=vZipB
                )

                c0 = arith_d.constant(IndexType.get(), 0)
                c16 = arith_d.constant(IndexType.get(), 16)
                c32 = arith_d.constant(IndexType.get(), 32)
                c48 = arith_d.constant(IndexType.get(), 48)
                c64 = arith_d.constant(IndexType.get(), 64)
                c80 = arith_d.constant(IndexType.get(), 80)

                vector_d.store(r4x4.result, elem_out, [c0])
                vector_d.store(r4x4_rt.result, elem_out, [c16])
                vector_d.store(r8x8_i32, elem_out, [c32])
                vector_d.store(r8x8_rt_i32, elem_out, [c48])
                vector_d.store(rzip_lo.result, elem_out, [c64])
                vector_d.store(rzip_hi.result, elem_out, [c80])

                out_fifo.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(out_ty)
        def sequence(out):
            o_task = shim_dma_single_bd_task(
                out_fifo, out, sizes=[1, 1, 1, OUT_LANES], issue_token=True
            )
            dma_start_task(o_task)
            dma_await_task(o_task)


parser = argparse.ArgumentParser(description="AIE4 aievec.aie4.shuffle (vshuffle)")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    shuffle_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
