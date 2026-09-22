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
    BoolAttr,
    IntegerAttr,
    IntegerType,
    Operation,
    VectorType,
)
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Exercise explicit UPS at a shift that can overflow i32.
SHIFT = 17


def my_ups_shift(dev):
    i16 = IntegerType.get_signless(16)
    i32 = IntegerType.get_signless(32)

    @device(dev)
    def device_body():
        v32i16 = VectorType.get([32], i16)
        v32i32 = VectorType.get([32], i32)
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        tile_size = 1024
        n_cores = 3

        A_ty = np.ndarray[(tile_size,), np.dtype[np.int16]]
        C_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

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
                    for j in range_(0, tile_size, 32):
                        pad = arith_d.constant(i16, 0)
                        va = vector_d.transfer_read(
                            v32i16, elem_in, [j], id_map, pad, [True]
                        )
                        vc = Operation.create(
                            "aievec.aie4.ups_i16_i32",
                            results=[v32i32],
                            operands=[va],
                            attributes={
                                "shift": IntegerAttr.get(i32, SHIFT),
                                "isSigned": BoolAttr.get(True),
                            },
                        ).results[0]
                        vector_d.transfer_write(
                            None, vc, elem_out, [j], id_map, [True]
                        )
                    inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                    outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        in_ty = np.ndarray[(tile_size * n_cores,), np.dtype[np.int16]]
        out_ty = np.ndarray[(tile_size * n_cores,), np.dtype[np.int32]]

        @runtime_sequence(in_ty, out_ty)
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


parser = argparse.ArgumentParser(description="AIE4 UPS i16->i32 nonzero-shift overflow probe")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    my_ups_shift(AIEDevice.npu3)
    res = ctx.module.operation.verify()
    if res != True:
        print(res)
    text = str(ctx.module)

with open(args.output, "w") as f:
    f.write(text)
