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

# One AIE4 vector register's worth of f32 lanes.
VEC_LANES = 32

# Each tile takes 32 vector accumulation steps.
TILE_SIZE = 1024
K_STEPS = TILE_SIZE // VEC_LANES

# Each input tile reduces to a single 32-lane f32 output vector.
OUT_TILE = VEC_LANES

N_CORES = 3
TILES_PER_CORE = 16


def macf32_design(dev):
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([VEC_LANES], f32)
        # transfer_read/write of a contiguous 1-D slice: identity permutation.
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        def splat_f32(value):
            """`arith.constant dense<value> : vector<32xf32>`."""
            return arith_d.constant(
                value=DenseElementsAttr.get_splat(v32f32, FloatAttr.get(f32, value)),
                result=v32f32,
            )

        A_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.float32]]
        B_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.float32]]
        C_ty = np.ndarray[(OUT_TILE,), np.dtype[np.float32]]

        shims = [tile(i, 0) for i in range(N_CORES)]
        for i, shim in enumerate(shims):
            shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")(
                [0, i]
            )
        cores = [tile(i, 2) for i in range(N_CORES)]

        for i in range(N_CORES):
            packetflow(
                i,
                shims[i],
                WireBundle.Control32,
                0,
                [{"dest": shims[0], "port": WireBundle.Controller32, "channel": 0}],
            )

        inA_fifos = [
            object_fifo(f"inA{i}", shims[i], cores[i], 2, A_ty) for i in range(N_CORES)
        ]
        inB_fifos = [
            object_fifo(f"inB{i}", shims[i], cores[i], 2, B_ty) for i in range(N_CORES)
        ]
        outC_fifos = [
            object_fifo(f"outC{i}", cores[i], shims[i], 2, C_ty) for i in range(N_CORES)
        ]

        for i in range(N_CORES):

            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(TILES_PER_CORE):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)

                        # Carry the accumulator across 32 steps; contract on both
                        # operations permits fusion to a native f32 MAC.
                        zero = splat_f32(0.0)
                        for k, acc, acc_out in range_(
                            0, TILE_SIZE, VEC_LANES, iter_args=[zero],
                            insert_yield=False,
                        ):
                            pad = arith_d.constant(f32, 0.0)
                            va = vector_d.transfer_read(
                                v32f32, elem_in_a, [k], id_map, pad, [True]
                            )
                            vb = vector_d.transfer_read(
                                v32f32, elem_in_b, [k], id_map, pad, [True]
                            )
                            prod = arith_d.mulf(
                                va, vb, fastmath=arith_d.FastMathFlags.contract
                            )
                            nxt = arith_d.addf(
                                acc, prod, fastmath=arith_d.FastMathFlags.contract
                            )
                            yield_([nxt])

                        c0 = arith_d.constant(IndexType.get(), 0)
                        vector_d.transfer_write(
                            None, acc_out, elem_out, [c0], id_map, [True]
                        )

                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        in_elems = TILE_SIZE * TILES_PER_CORE * N_CORES
        out_elems = OUT_TILE * TILES_PER_CORE * N_CORES
        in_ty = np.ndarray[(in_elems,), np.dtype[np.float32]]
        out_ty = np.ndarray[(out_elems,), np.dtype[np.float32]]

        @runtime_sequence(in_ty, in_ty, out_ty)
        def sequence(A, B, C):
            a_task, b_task, c_task = [], [], []
            in_per_core = in_elems // N_CORES
            out_per_core = out_elems // N_CORES
            for i in range(N_CORES):
                a_task.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i],
                        A,
                        offset=i * in_per_core,
                        sizes=[1, 1, 1, in_per_core],
                        issue_token=True,
                    )
                )
                b_task.append(
                    shim_dma_single_bd_task(
                        inB_fifos[i],
                        B,
                        offset=i * in_per_core,
                        sizes=[1, 1, 1, in_per_core],
                        issue_token=True,
                    )
                )
                c_task.append(
                    shim_dma_single_bd_task(
                        outC_fifos[i],
                        C,
                        offset=i * out_per_core,
                        sizes=[1, 1, 1, out_per_core],
                        issue_token=True,
                    )
                )
                dma_start_task(a_task[i], b_task[i], c_task[i])

            for i in range(N_CORES):
                dma_await_task(a_task[i], b_task[i], c_task[i])


parser = argparse.ArgumentParser(description="AIE4 f32 MAC (contract-fused)")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    macf32_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
