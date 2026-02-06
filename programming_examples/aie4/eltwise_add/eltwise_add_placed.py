#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
# from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape


import aie.utils.trace as trace_utils



def my_eltwise_add(dev, in1_size, in2_size, out_size, dtype=np.int32):
    in1_dtype = dtype
    in2_dtype = dtype
    out_dtype = dtype

    tensor_size = in1_size

    # Tile sizes
    tile_size = 128
    tensor_div_tile = tensor_size // tile_size

    target_model = get_target_model(dev)
    n_cores = 1 #target_model.columns()

    tiles = tensor_div_tile // n_cores
    buffer_depth = 2

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the tile memory
        A_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        B_ty = np.ndarray[(tile_size,), np.dtype[in2_dtype]]
        C_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the memory tile which aggregates across the 2 cores
        A_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in1_dtype]]
        B_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in2_dtype]]
        C_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[out_dtype]]

        # AIE Core Function declarations

        # eltwise_add_bf16_scalar = external_func(
        #     "eltwise_add_bf16_scalar", inputs=[tile_ty, tile_ty, tile_ty]
        # )
        # eltwise_add_bf16_vector = external_func(
        #     "eltwise_add_bf16_vector", inputs=[tile_ty, tile_ty, tile_ty]
        # )

        # Tile declarations
        ShimTiles = [tile(i, 0) for i in range(n_cores)]
        # MemTiles = [tile(i, 1) for i in range(n_cores)]
        cores = [tile(i, 2) for i in range(n_cores)]

        controller_id_attr = AttrBuilder.get("PacketInfoAttr")([0, 3])
        ShimTiles[0].attributes["controller_id"] = controller_id_attr

        # inB_fifos = []
        # outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        inA = [
            object_fifo(
                f"inA{i}", ShimTiles[i], cores[i], buffer_depth, A_ty
            )
            for i in range(n_cores)
        ]
        # inA_fifos = [
        #     object_fifo(f"memA{i}", MemTiles[i], cores[i], buffer_depth, A_ty)
        #     for i in range(n_cores)
        # ]
        # [object_fifo_link(inA[i], inA_fifos[i], [], []) for i in range(n_cores)]

        # Input B
        inB = [
            object_fifo(
                f"inB{i}", ShimTiles[i], cores[i], buffer_depth, B_ty
            )
            for i in range(n_cores)
        ]
        # inB_fifos = [
        #     object_fifo(f"memB{i}", MemTiles[i], cores[i], buffer_depth, B_ty)
        #     for i in range(n_cores)
        # ]
        # [object_fifo_link(inB[i], inB_fifos[i], [], []) for i in range(n_cores)]

        # Output C
        outC = [object_fifo(f"outC{i}", cores[i], ShimTiles[i], buffer_depth, C_ty)
                 for i in range(n_cores)]
        # outC_fifos = [
        #     object_fifo(f"memC{i}", cores[i], MemTiles[i], buffer_depth, C_ty)
        #     for i in range(n_cores)
        # ]
        # [object_fifo_link(outC_fifos[i], outC[i], [], []) for i in range(n_cores)]

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(tiles):
                        elem_out = outC[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB[i].acquire(ObjectFifoPort.Consume, 1)
                        for j in range_(tile_size):
                            elem_out[j] = elem_in_a[j] + elem_in_b[j] + 42
                        inA[i].release(ObjectFifoPort.Consume, 1)
                        inB[i].release(ObjectFifoPort.Consume, 1)
                        outC[i].release(ObjectFifoPort.Produce, 1)

        packetflow(3, ShimTiles[0], WireBundle.Control32, 0, [{"dest":ShimTiles[0], "port": WireBundle.Controller32, "channel": 0}])
        # packet_flow(ShimTiles[1], WireBundle.Control32, 0, ShimTiles[0], WireBundle.Controller32, 0)
        # packet_flow(ShimTiles[2], WireBundle.Control32, 0, ShimTiles[0], WireBundle.Controller32, 0)

        # To/from AIE-array data movement

        tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            a_tasks = [
                shim_dma_single_bd_task(
                    inA[i],
                    A,
                    sizes=[1, 1, 1, tensor_size // n_cores],
                    offset=i * (tensor_size // n_cores),
                    issue_token=True,
                )
                for i in range(n_cores)
            ]
            b_tasks = [
                shim_dma_single_bd_task(
                    inB[i],
                    B,
                    sizes=[1, 1, 1, tensor_size // n_cores],
                    offset=i * (tensor_size // n_cores),
                    issue_token=True,
                )
                for i in range(n_cores)
            ]
            c_tasks = [
                shim_dma_single_bd_task(
                    outC[i],
                    C,
                    sizes=[1, 1, 1, tensor_size // n_cores],
                    offset=i * (tensor_size // n_cores),
                    issue_token=True,
                )
                for i in range(n_cores)
            ]

            for i in range(n_cores):
                dma_start_task(a_tasks[i], b_tasks[i], c_tasks[i])
            for i in range(n_cores):
                dma_await_task(a_tasks[i], b_tasks[i], c_tasks[i])


if len(sys.argv) < 5:
    raise ValueError(
        "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
    )


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")

opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
elif opts.device == "npu3":
    dev = AIEDevice.npu3
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(opts.in1_size)
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)

with mlir_mod_ctx() as ctx:
    my_eltwise_add(dev, in1_size, in2_size, out_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
