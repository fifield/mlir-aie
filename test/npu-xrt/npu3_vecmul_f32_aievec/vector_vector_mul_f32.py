#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices Inc.

# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith as arith_d
from aie.dialects import vector as vector_d
from aie.ir import VectorType, F32Type, AffineMap, AffineMapAttr
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape

tile_size = 1024

def my_eltwise_mul(dev, in1_size, in2_size, out_size, dtype=np.float32):
    in1_dtype = dtype
    in2_dtype = dtype
    out_dtype = dtype

    tensor_size = in1_size

    # Tile sizes
    tensor_div_tile = tensor_size // tile_size

    n_cores = 3
    tiles = tensor_div_tile // n_cores
    buffer_depth = 2

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    # 1024 bits / 32 bits per f32 = 32 lanes: exactly one AIE4 vector register,
    # which is the only shape llvm.aie4.*.fp.mul.ewise.conf accepts.
    vec_lanes = 32

    @device(dev)
    def device_body():
        v32f32 = VectorType.get([vec_lanes], F32Type.get())
        # transfer_read/write of a contiguous 1-D slice: identity permutation.
        id_map = AffineMapAttr.get(AffineMap.get_identity(1))

        # Type used in the tile memory
        A_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        B_ty = np.ndarray[(tile_size,), np.dtype[in2_dtype]]
        C_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the memory tile which aggregates across the n_cores
        A_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in1_dtype]]
        B_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in2_dtype]]
        C_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[out_dtype]]

        # Tile declarations
        shims = [tile(i, 0) for i in range(n_cores)]
        for i, shim in enumerate(shims):
            shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, i])
        # mems = [tile(i, 1) for i in range(n_cores)]
        cores = [tile(i, 2) for i in range(n_cores)]

        for i in range(n_cores):
            packetflow(i, shims[i], WireBundle.Control32, 0, [{"dest":shims[0], "port": WireBundle.Controller32, "channel": 0}])

        inA_fifos = []
        inB_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"inA{i}", shims[i], cores[i], buffer_depth, A_ty)
            )

        # Input B
        for i in range(n_cores):
            inB_fifos.append(
                object_fifo(f"inB{i}", shims[i], cores[i], buffer_depth, B_ty)
            )

        # Output C
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"outC{i}", cores[i], shims[i], buffer_depth, C_ty)
            )

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
                        # Explicit 32-lane f32 vector multiply. This is the
                        # whole point of the test: `arith.mulf` on
                        # vector<32xf32> is the op that convert-vector-to-aievec
                        # turns into aievec.aie4.mul_ewise_f32, which becomes a
                        # single `vmul.f` instruction. No external kernel object
                        # supplies this multiply.
                        for j in range_(0, tile_size, vec_lanes):
                            pad = arith_d.constant(F32Type.get(), 0.0)
                            va = vector_d.transfer_read(
                                v32f32, elem_in_a, [j], id_map, pad, [True]
                            )
                            vb = vector_d.transfer_read(
                                v32f32, elem_in_b, [j], id_map, pad, [True]
                            )
                            vc = arith_d.mulf(va, vb)
                            vector_d.transfer_write(
                                None, vc, elem_out, [j], id_map, [True]
                            )
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):

            a_task = []
            b_task = []
            c_task = []
            for i in range(n_cores):
                a_task.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i],
                        A,
                        offset=i * (tensor_size // n_cores),
                        sizes=[1, 1, 1, tensor_size // n_cores],
                        issue_token=True,
                    )
                )
                b_task.append(
                    shim_dma_single_bd_task(
                        inB_fifos[i],
                        B,
                        offset=i * (tensor_size // n_cores),
                        sizes=[1, 1, 1, tensor_size // n_cores],
                        issue_token=True,
                    )
                )
                c_task.append(
                    shim_dma_single_bd_task(
                        outC_fifos[i],
                        C,
                        offset=i * (tensor_size // n_cores),
                        sizes=[1, 1, 1, tensor_size // n_cores],
                        issue_token=True,
                    )
                )
                dma_start_task(a_task[i], b_task[i], c_task[i])

            for i in range(n_cores):
                dma_await_task(a_task[i], b_task[i], c_task[i])


parser = argparse.ArgumentParser(description="Vector-Vector Multiplication Program")
parser.add_argument("-o", "--output", type=str, required=True, help="Output filename")
args = parser.parse_args()

dev = AIEDevice.npu3

in1_size = tile_size * 3 * 16
in2_size = tile_size * 3 * 16
out_size = tile_size * 3 * 16

with mlir_mod_ctx() as ctx:
    my_eltwise_mul(dev, in1_size, in2_size, out_size)
    res = ctx.module.operation.verify()
    if res == True:
        pass
    else:
        print(res)

with open(args.output, "w") as f:
    f.write(str(ctx.module))
