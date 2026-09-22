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
from aie.ir import AffineMap, AffineMapAttr, BF16Type, F32Type, VectorType, Operation
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Exercise explicit narrowing followed by exact widening. The fixture expects
# floor rounding; the conversion operation itself does not establish that mode.

tile_size = 1024
vec_lanes = 32
n_cores = 3


def my_bf16_roundtrip(dev):
    @device(dev)
    def device_body():
        v32f32 = VectorType.get([vec_lanes], F32Type.get())
        v32bf16 = VectorType.get([vec_lanes], BF16Type.get())
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        A_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
        C_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]

        shims = [tile(i, 0) for i in range(n_cores)]
        for i, shim in enumerate(shims):
            shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, i])
        cores = [tile(i, 2) for i in range(n_cores)]

        for i in range(n_cores):
            packetflow(i, shims[i], WireBundle.Control32, 0, [{"dest": shims[0], "port": WireBundle.Controller32, "channel": 0}])

        inA_fifos = [object_fifo(f"inA{i}", shims[i], cores[i], 2, A_ty) for i in range(n_cores)]
        outC_fifos = [object_fifo(f"outC{i}", cores[i], shims[i], 2, C_ty) for i in range(n_cores)]

        for i in range(n_cores):

            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    elem_in = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                    for j in range_(0, tile_size, vec_lanes):
                        pad = arith_d.constant(F32Type.get(), 0.0)
                        va = vector_d.transfer_read(
                            v32f32, elem_in, [j], id_map, pad, [True]
                        )
                        vb = Operation.create(
                            "aievec.aie4.conv_f32_bf16", results=[v32bf16], operands=[va]
                        ).results[0]
                        vc = arith_d.extf(v32f32, vb)
                        vector_d.transfer_write(
                            None, vc, elem_out, [j], id_map, [True]
                        )
                    inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                    outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        tensor_ty = np.ndarray[(tile_size * n_cores,), np.dtype[np.float32]]

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):
            a_task, c_task = [], []
            for i in range(n_cores):
                a_task.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i], A, offset=i * tile_size,
                        sizes=[1, 1, 1, tile_size], issue_token=True,
                    )
                )
                c_task.append(
                    shim_dma_single_bd_task(
                        outC_fifos[i], C, offset=i * tile_size,
                        sizes=[1, 1, 1, tile_size], issue_token=True,
                    )
                )
                dma_start_task(a_task[i], c_task[i])
            for i in range(n_cores):
                dma_await_task(a_task[i], c_task[i])


parser = argparse.ArgumentParser(description="AIE4 bf16<->f32 round-trip rounding gate")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    my_bf16_roundtrip(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
