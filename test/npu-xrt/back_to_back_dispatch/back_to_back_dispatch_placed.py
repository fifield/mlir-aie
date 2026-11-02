# back_to_back_dispatch/back_to_back_dispatch_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

# out = in + K over 1024 x i32, streamed
#   shim -> memtile -> core -> memtile -> shim
# through depth-2 objectFifos. The memtile objectFifo BD chains loop forever,
# which is what makes back-to-back dispatch of this design interesting
# (issue #423); the dataflow itself is the placed vector_scalar_add.
#
#   python back_to_back_dispatch_placed.py [--name main] [--k 1] [--output-first]
#
# --name names the device (the kernel is "<name>:sequence"), --k sets K, and
# --output-first makes the core acquire its output buffer before its input.
import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64  # memtile slot, in elements
AIE_TILE_WIDTH = 32  # core buffer, in elements
DEPTH = 2


def back_to_back_dispatch(name, k, output_first):
    @device(AIEDevice.npu3, sym_name=name)
    def device_body():
        mem_tile_ty = np.ndarray[(MEM_TILE_WIDTH,), np.dtype[np.int32]]
        aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
        all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile = tile(0, 2)

        # AIE4 needs a controller_id on the shim tile and Control32 ->
        # Controller32 packet flows so dma_await_task has a return path.
        ShimTile.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, 0])
        for ctrl_ch in (0, 1):
            packetflow(
                0,
                ShimTile,
                WireBundle.Control32,
                ctrl_ch,
                [{"dest": ShimTile, "port": WireBundle.Controller32, "channel": 0}],
            )

        # Input: shim -> memtile (64-element slots) -> core (32-element buffers)
        of_in0 = object_fifo("in0", ShimTile, MemTile, DEPTH, mem_tile_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile, DEPTH, aie_tile_ty)
        object_fifo_link(of_in0, of_in1)

        # Output: core -> memtile -> shim
        of_out0 = object_fifo("out0", MemTile, ShimTile, DEPTH, mem_tile_ty)
        of_out1 = object_fifo("out1", ComputeTile, MemTile, DEPTH, aie_tile_ty)
        object_fifo_link(of_out1, of_out0)

        @core(ComputeTile)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                if output_first:
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                else:
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(AIE_TILE_WIDTH):
                    elem_out[i] = elem_in[i] + k
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(all_data_ty, all_data_ty)
        def sequence(inTensor, outTensor):
            in_task = shim_dma_single_bd_task(
                of_in0, inTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out0, outTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="main")
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--output-first", action="store_true")
args = parser.parse_args()

with mlir_mod_ctx() as ctx:
    back_to_back_dispatch(args.name, args.k, args.output_first)
    assert ctx.module.operation.verify(), "module verification failed"
    print(ctx.module)
