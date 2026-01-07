#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_eltwise_mul(dev, in1_size, in2_size, out_size, dtype=np.int32):
    in1_dtype = dtype
    in2_dtype = dtype
    out_dtype = dtype

    tensor_size = in1_size

    # Tile sizes
    tile_size = 1024
    tensor_div_tile = tensor_size // tile_size

    n_cores = 4
    tiles = tensor_div_tile // n_cores
    buffer_depth = 2

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    @device(dev)
    def device_body():

        # Type used in the tile memory
        A_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        B_ty = np.ndarray[(tile_size,), np.dtype[in2_dtype]]
        C_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the memory tile which aggregates across the n_cores
        A_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in1_dtype]]
        B_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in2_dtype]]
        C_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[out_dtype]]

        # Tile declarations
        ShimTile = tile(0, 0)

        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        inB_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, A_memTile_ty)
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, A_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inA, inA_fifos, [], of_offsets)

        # Input B
        inB = object_fifo("inB", ShimTile, MemTile, buffer_depth, B_memTile_ty)
        for i in range(n_cores):
            inB_fifos.append(
                object_fifo(f"memB{i}", MemTile, cores[i], buffer_depth, B_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(B_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inB, inB_fifos, [], of_offsets)

        # Output C
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, C_ty)
            )
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, C_memTile_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifos, outC, of_offsets, [])

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(tiles):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        for j in range_(tile_size):
                            elem_out[j] = elem_in_a[j] * elem_in_b[j]
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):

            a_task = shim_dma_single_bd_task(
                inA,
                A,
                sizes=[1, 1, 1, tensor_size],
                issue_token=True,
            )
            b_task = shim_dma_single_bd_task(
                inB,
                B,
                sizes=[1, 1, 1, tensor_size],
                issue_token=True,
            )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(a_task, b_task, c_task)
            dma_await_task(a_task, b_task, c_task)


parser = argparse.ArgumentParser(description="Vector-Vector Multiplication Program")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

dev = AIEDevice.npu3z

in1_size = 8192
in2_size = 8192
out_size = 8192

with mlir_mod_ctx() as ctx:
    my_eltwise_mul(dev, in1_size, in2_size, out_size)
    res = ctx.module.operation.verify()
    if res == True:
        pass
    else:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
