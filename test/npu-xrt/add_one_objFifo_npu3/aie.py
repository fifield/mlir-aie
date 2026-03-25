# aie_npu3.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

N = 64
line_size = 8
add_val = 41
dev = AIEDevice.npu3
col = 0


def add_one_objFifo():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile = tile(col, 2)

            controller_id_attr = AttrBuilder.get("PacketInfoAttr")([0, 1])
            ShimTile.attributes["controller_id"] = controller_id_attr

            # AIE-array data movement with object fifos
            of_in0 = object_fifo("in0", ShimTile, MemTile, 2, line_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile, 2, line_ty)
            object_fifo_link(of_in0, of_in1)

            of_out1 = object_fifo("out1", ComputeTile, MemTile, 2, line_ty)
            of_out0 = object_fifo("out0", MemTile, ShimTile, 2, line_ty)
            object_fifo_link(of_out1, of_out0)

            packetflow(
                1,
                ShimTile,
                WireBundle.Control32,
                0,
                [
                    {
                        "dest": ShimTile,
                        "port": WireBundle.Controller32,
                        "channel": 0,
                    }
                ],
            )

            # Core logic: add 41 to each element
            @core(ComputeTile)
            def core_body():
                for _ in range_(N // line_size):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for j in range_(line_size):
                        elem_out[j] = elem_in[j] + add_val
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                in_task = shim_dma_single_bd_task(
                    of_in0, A, sizes=[1, 1, 1, N], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    of_out0, C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output", default=None)
    args = parser.parse_args()
    if args.output:
        with open(args.output, "w") as f:
            f.write(str(ctx.module))
    else:
        print(ctx.module)


add_one_objFifo()
