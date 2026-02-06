# passthrough_dmas/passthrough_dmas_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

N = 4096
dev = AIEDevice.npu3
line_size = 1024
col = 0

def my_passthrough():
    with mlir_mod_ctx() as ctx:

        
        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile1 = tile(col, 1)

            controller_id_attr = AttrBuilder.get("PacketInfoAttr")([0, 1])
            ShimTile.attributes["controller_id"] = controller_id_attr

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile1, 2, line_ty)
            of_out = object_fifo("out", MemTile1, ShimTile, 2, line_ty)
            object_fifo_link(of_in, of_out)

            packetflow(1, ShimTile, WireBundle.Control32, 0, [{"dest":ShimTile, "port": WireBundle.Controller32, "channel": 0}])

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                in_task = shim_dma_single_bd_task(
                    of_in, A, sizes=[1, 1, 1, N], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    print(ctx.module)


my_passthrough()
