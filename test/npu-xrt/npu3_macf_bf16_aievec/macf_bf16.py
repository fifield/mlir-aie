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
from ml_dtypes import bfloat16

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import vector as vector_d
from aie.dialects.scf import yield_
from aie.ir import (
    AffineMap,
    AffineMapAttr,
    BF16Type,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    IndexType,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# The native bf16 -> f32 form processes 32 lanes.
VEC_LANES = 32

# bf16 elements per input tile.
TILE_SIZE = 1024
K_STEPS = TILE_SIZE // VEC_LANES

# Pack [1024 MUL results, 32 MAC results] into one output FIFO to fit
# NPU3's per-tile DMA channel limit.
OUT_MUL_TILE = TILE_SIZE
OUT_MAC_TILE = VEC_LANES
OUT_TILE = OUT_MUL_TILE + OUT_MAC_TILE

N_CORES = 3
TILES_PER_CORE = 16


def macf_bf16_design(dev):
    bf16 = BF16Type.get()
    f32 = F32Type.get()

    @device(dev)
    def device_body():
        v32bf16 = VectorType.get([VEC_LANES], bf16)
        v32f32 = VectorType.get([VEC_LANES], f32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        def splat_f32(value):
            return arith_d.constant(
                value=DenseElementsAttr.get_splat(v32f32, FloatAttr.get(f32, value)),
                result=v32f32,
            )

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

        # Match the host's two-byte bf16 input layout.
        A_bf16_ty = np.ndarray[(TILE_SIZE,), np.dtype[bfloat16]]
        B_bf16_ty = np.ndarray[(TILE_SIZE,), np.dtype[bfloat16]]

        inA_fifos = [
            object_fifo(f"inA{i}", shims[i], cores[i], 2, A_bf16_ty)
            for i in range(N_CORES)
        ]
        inB_fifos = [
            object_fifo(f"inB{i}", shims[i], cores[i], 2, B_bf16_ty)
            for i in range(N_CORES)
        ]
        outC_fifos = [
            object_fifo(f"outC{i}", cores[i], shims[i], 2, C_ty)
            for i in range(N_CORES)
        ]

        for i in range(N_CORES):

            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(TILES_PER_CORE):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)

                        # Native bf16 -> f32 multiply (conf 107).
                        pad_bf16 = arith_d.constant(bf16, 0.0)
                        for j in range_(0, TILE_SIZE, VEC_LANES):
                            va = vector_d.transfer_read(
                                v32bf16, elem_in_a, [j], id_map, pad_bf16, [True]
                            )
                            vb = vector_d.transfer_read(
                                v32bf16, elem_in_b, [j], id_map, pad_bf16, [True]
                            )
                            ea = arith_d.extf(v32f32, va)
                            eb = arith_d.extf(v32f32, vb)
                            vc = arith_d.mulf(ea, eb)
                            vector_d.transfer_write(
                                None, vc, elem_out, [j], id_map, [True]
                            )

                        # Accumulate with contract-enabled bf16 -> f32 MAC (conf 106).
                        zero = splat_f32(0.0)
                        for k, acc, acc_out in range_(
                            0, TILE_SIZE, VEC_LANES, iter_args=[zero],
                            insert_yield=False,
                        ):
                            va = vector_d.transfer_read(
                                v32bf16, elem_in_a, [k], id_map, pad_bf16, [True]
                            )
                            vb = vector_d.transfer_read(
                                v32bf16, elem_in_b, [k], id_map, pad_bf16, [True]
                            )
                            ea = arith_d.extf(v32f32, va)
                            eb = arith_d.extf(v32f32, vb)
                            prod = arith_d.mulf(
                                ea, eb, fastmath=arith_d.FastMathFlags.contract
                            )
                            nxt = arith_d.addf(
                                acc, prod, fastmath=arith_d.FastMathFlags.contract
                            )
                            yield_([nxt])

                        c_mac_off = arith_d.constant(IndexType.get(), OUT_MUL_TILE)
                        vector_d.transfer_write(
                            None, acc_out, elem_out, [c_mac_off], id_map, [True]
                        )

                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        in_elems = TILE_SIZE * TILES_PER_CORE * N_CORES
        out_elems = OUT_TILE * TILES_PER_CORE * N_CORES
        in_ty = np.ndarray[(in_elems,), np.dtype[bfloat16]]
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


parser = argparse.ArgumentParser(description="AIE4 bf16 -> f32 MUL/MAC")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    macf_bf16_design(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
