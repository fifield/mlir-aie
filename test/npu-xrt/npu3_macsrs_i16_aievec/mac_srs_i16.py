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
from aie.dialects import aievec as aievec_d
from aie.dialects import vector as vector_d
from aie.dialects.scf import yield_
from aie.ir import (
    AffineMap,
    AffineMapAttr,
    DenseElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# MAC uses 32 i16 input lanes and 32 i32 accumulator lanes.
VEC_LANES = 32

# i16 elements per input tile. 1024 / 32 = 32 accumulation steps, so the
# accumulator is carried across 32 loop iterations before being narrowed.
TILE_SIZE = 1024
K_STEPS = TILE_SIZE // VEC_LANES

# Each input tile reduces to a single 32-lane i16 output vector.
OUT_TILE = VEC_LANES

# Must match the shift in test.cpp.
SRS_SHIFT = 6

N_CORES = 3
TILES_PER_CORE = 16


def mac_srs_design(dev):
    i16 = IntegerType.get_signless(16)
    i32 = IntegerType.get_signless(32)

    @device(dev)
    def device_body():
        v32i16 = VectorType.get([VEC_LANES], i16)
        v32i32 = VectorType.get([VEC_LANES], i32)
        # transfer_read/write of a contiguous 1-D slice: identity permutation.
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        def splat_i32(value):
            """`arith.constant dense<value> : vector<32xi32>`."""
            return arith_d.constant(
                value=DenseElementsAttr.get_splat(
                    v32i32, IntegerAttr.get(i32, value)
                ),
                result=v32i32,
            )

        A_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int16]]
        B_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int16]]
        C_ty = np.ndarray[(OUT_TILE,), np.dtype[np.int16]]

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

                        # Carry the i32 accumulator through 32 iterations.
                        # Lane j sums A[k*32+j] * B[k*32+j] over k.
                        zero = splat_i32(0)
                        for k, acc, acc_out in range_(
                            0, TILE_SIZE, VEC_LANES, iter_args=[zero],
                            insert_yield=False,
                        ):
                            pad = arith_d.constant(i16, 0)
                            va = vector_d.transfer_read(
                                v32i16, elem_in_a, [k], id_map, pad, [True]
                            )
                            vb = vector_d.transfer_read(
                                v32i16, elem_in_b, [k], id_map, pad, [True]
                            )
                            # VMAC absorbs these sign extensions.
                            ea = arith_d.extsi(v32i32, va)
                            eb = arith_d.extsi(v32i32, vb)
                            prod = arith_d.muli(ea, eb)
                            # Fuse the widened product and add into one MAC.
                            nxt = arith_d.addi(acc, prod)
                            yield_([nxt])

                        # Request the mode-dependent SRS operation explicitly.
                        # This test expects floor rounding without saturation.
                        narrowed = aievec_d.aie4_srs_i32_i16(
                            v32i16, acc_out, SRS_SHIFT
                        )

                        c0 = arith_d.constant(IndexType.get(), 0)
                        vector_d.transfer_write(
                            None, narrowed, elem_out, [c0], id_map, [True]
                        )

                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        in_elems = TILE_SIZE * TILES_PER_CORE * N_CORES
        out_elems = OUT_TILE * TILES_PER_CORE * N_CORES
        in_ty = np.ndarray[(in_elems,), np.dtype[np.int16]]
        out_ty = np.ndarray[(out_elems,), np.dtype[np.int16]]

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


parser = argparse.ArgumentParser(description="AIE4 integer MAC + SRS")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    mac_srs_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
