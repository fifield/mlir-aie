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
from aie.ir import AffineMap, AffineMapAttr, IntegerType, VectorType
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# int8 widening and MACs on one core. Let T = tile_size and j be a chunk's
# starting offset. Pair input chunk `a` at j with `b` at T - vec_lanes - j.
# For each element i, the four output slabs contain:
#   out[0*T + i] = extsi(a)                      unpack signed + UPS
#   out[1*T + i] = extui(a)                      unpack unsigned + UPS
#   out[2*T + i] = extui(a) + extsi(a) * extsi(a) signed MAC, zext accumulator
#   out[3*T + i] = extsi(a) + extui(a) * extsi(b) uint8 x int8 MAC

tile_size = 1024
vec_lanes = 32
n_out = 4


def unpack_i8(dev):
    i8 = IntegerType.get_signless(8)
    i32 = IntegerType.get_signless(32)

    @device(dev)
    def device_body():
        v32i8 = VectorType.get([vec_lanes], i8)
        v32i32 = VectorType.get([vec_lanes], i32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        A_ty = np.ndarray[(tile_size,), np.dtype[np.int8]]
        C_ty = np.ndarray[(n_out * tile_size,), np.dtype[np.int32]]

        shim = tile(0, 0)
        shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, 0])
        core_tile = tile(0, 2)
        packetflow(
            0,
            shim,
            WireBundle.Control32,
            0,
            [{"dest": shim, "port": WireBundle.Controller32, "channel": 0}],
        )

        inA = object_fifo("inA", shim, core_tile, 2, A_ty)
        outC = object_fifo("outC", core_tile, shim, 2, C_ty)

        @core(core_tile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                elem_in = inA.acquire(ObjectFifoPort.Consume, 1)

                def read(idx):
                    pad = arith_d.constant(i8, 0)
                    return vector_d.transfer_read(
                        v32i8, elem_in, [idx], id_map, pad, [True]
                    )

                def write(v, q, j):
                    idx = arith_d.addi(j, arith_d.constant(j.type, q * tile_size))
                    vector_d.transfer_write(None, v, elem_out, [idx], id_map, [True])

                # Standalone widenings.
                for j in range_(0, tile_size, vec_lanes):
                    va = read(j)
                    write(arith_d.extsi(v32i32, va), 0, j)
                    write(arith_d.extui(v32i32, va), 1, j)

                # MAC chains.
                for j in range_(0, tile_size, vec_lanes):
                    jb = arith_d.subi(
                        arith_d.constant(j.type, tile_size - vec_lanes), j
                    )
                    va = read(j)
                    vb = read(jb)
                    sa = arith_d.extsi(v32i32, va)
                    ua = arith_d.extui(v32i32, va)
                    sb = arith_d.extsi(v32i32, vb)
                    write(arith_d.addi(ua, arith_d.muli(sa, sa)), 2, j)
                    ua2 = arith_d.extui(v32i32, va)
                    sa2 = arith_d.extsi(v32i32, va)
                    write(arith_d.addi(sa2, arith_d.muli(ua2, sb)), 3, j)

                inA.release(ObjectFifoPort.Consume, 1)
                outC.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(A_ty, C_ty)
        def sequence(A, C):
            a_task = shim_dma_single_bd_task(
                inA, A, sizes=[1, 1, 1, tile_size], issue_token=True
            )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, n_out * tile_size], issue_token=True
            )
            dma_start_task(a_task, c_task)
            dma_await_task(a_task, c_task)


parser = argparse.ArgumentParser(description="AIE4 int8 unpack and int8 MAC gate")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    unpack_i8(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
