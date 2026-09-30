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
from aie.ir import (
    AffineMap,
    AffineMapAttr,
    BF16Type,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    Operation,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# One AIE4 vector register of f32 lanes; VEXP2/VTANH process 32 lanes.
VEC_LANES = 32
HALF = VEC_LANES // 2

# Per 32-lane chunk, in f32 words (bf16 vectors are packed two per word).
# Input:  x (f32) | r (f32, raw VEXP2 input) | xb (bf16) | pad
# Output: exp2_afn(x) | tanh_afn(x) | raw_bf20(r) | x + bias | xb + bias |
#         exp2_afn(xb) (bf16) | tanh_afn(xb) (bf16) | raw_bf16(r) (bf16) | pad |
#         raw_bf20(x + ALT_BIAS[0]) | raw_bf20(x + ALT_BIAS[1])
# Chunks are padded to 128 bytes: 1024-bit vector accesses must be aligned.
IN_WORDS = 3 * VEC_LANES
OUT_WORDS = 9 * VEC_LANES

CHUNKS_PER_TILE = 8
N_TILES = 64

# Match the lowering's result-specific biases: half a 2^-11 input step
# for f32, and the vendor aie4_nlf.h constant for bf16.
EXP2_BIAS_F32 = 895.0 + 2.0**-12
EXP2_BIAS_BF16 = 895.001
# Compare the base bias with the f32 lowering's bias; results are diagnostic.
ALT_BIAS = [895.0, 895.0 + 2.0**-12]


def approx_math_design(dev):
    f32 = F32Type.get()
    bf16 = BF16Type.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([VEC_LANES], f32)
        v16f32 = VectorType.get([HALF], f32)
        v32bf16 = VectorType.get([VEC_LANES], bf16)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))
        afn = arith_d.FastMathFlags.afn

        in_ty = np.ndarray[(IN_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]
        out_ty = np.ndarray[(OUT_WORDS * CHUNKS_PER_TILE,), np.dtype[np.float32]]

        shim = tile(0, 0)
        core_tile = tile(0, 2)
        in_fifo = object_fifo("in", shim, core_tile, 2, in_ty)
        out_fifo = object_fifo("out", core_tile, shim, 2, out_ty)

        def raw_exp2(src, result_type):
            return Operation.create(
                "aievec.aie4.biased_exp2", results=[result_type], operands=[src]
            ).results[0]

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
                        r = in_at(VEC_LANES, v32f32)
                        xb = vector_d.bitcast(v32bf16, in_at(2 * VEC_LANES, v16f32))

                        def bias_const(value):
                            return arith_d.constant(
                                v32f32,
                                DenseElementsAttr.get_splat(
                                    v32f32, FloatAttr.get(f32, value)
                                ),
                            )

                        bias_f32 = bias_const(EXP2_BIAS_F32)
                        bias_bf16 = bias_const(EXP2_BIAS_BF16)
                        xb_wide = arith_d.extf(v32f32, xb)

                        out_at(0, math_d.exp2(x, fastmath=afn))
                        out_at(VEC_LANES, math_d.tanh(x, fastmath=afn))
                        out_at(2 * VEC_LANES, raw_exp2(r, v32f32))
                        # Record the biased inputs to check the raw VEXP2 contract.
                        out_at(3 * VEC_LANES, arith_d.addf(x, bias_f32))
                        out_at(4 * VEC_LANES, arith_d.addf(xb_wide, bias_bf16))

                        def packed(v):
                            return vector_d.bitcast(v16f32, v)

                        out_at(5 * VEC_LANES, packed(math_d.exp2(xb, fastmath=afn)))
                        out_at(
                            5 * VEC_LANES + HALF,
                            packed(math_d.tanh(xb, fastmath=afn)),
                        )
                        for k, alt in enumerate(ALT_BIAS):
                            alt_bias = arith_d.constant(
                                v32f32,
                                DenseElementsAttr.get_splat(
                                    v32f32, FloatAttr.get(f32, alt)
                                ),
                            )
                            out_at(
                                (7 + k) * VEC_LANES,
                                raw_exp2(arith_d.addf(x, alt_bias), v32f32),
                            )
                    # Raw bf16 VEXP2 of r runs in its own loop. In the same
                    # block as the f32 VEXP2 of r, Peano's post-RA scheduler
                    # (llvm-aie g663a9e74) moves `x12 = VEXP2_bf16 y6` one
                    # cycle ahead of `y7 = VEXP2_fp32 y6`. x12 is the low half
                    # of y6, so f32 lanes 0-15 then read the bf16 results.
                    for c in range_(CHUNKS_PER_TILE):
                        pad = arith_d.constant(f32, 0.0)
                        cin = arith_d.muli(c, arith_d.constant(c.type, IN_WORDS))
                        cout = arith_d.muli(c, arith_d.constant(c.type, OUT_WORDS))
                        r_idx = arith_d.addi(cin, arith_d.constant(c.type, VEC_LANES))
                        r = vector_d.transfer_read(
                            v32f32, elem_in, [r_idx], id_map, pad, [True]
                        )
                        raw16 = vector_d.bitcast(v16f32, raw_exp2(r, v32bf16))
                        out_idx = arith_d.addi(
                            cout, arith_d.constant(c.type, 5 * VEC_LANES + 2 * HALF)
                        )
                        vector_d.transfer_write(
                            None, raw16, elem_out, [out_idx], id_map, [True]
                        )
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


parser = argparse.ArgumentParser(description="AIE4 afn exp2/tanh accuracy")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    approx_math_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
