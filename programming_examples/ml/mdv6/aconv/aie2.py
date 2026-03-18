#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import numpy as np
import sys

from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def aconv_layer_bf16(
    dev,
    input_height: int,
    input_width: int,
    input_channels: int,
    output_channels: int,
):
    """
    AConv layer for MDV6 on AIE2P.
    
    AConv = AvgPool2d(2×2, stride=1) + Conv3x3(stride=2)
    This is a downsampling layer used in the MDV6 backbone.
    """

    # Calculate intermediate and output dimensions
    # After AvgPool: (H-1, W-1, C_in)
    pooled_height = input_height - 1
    pooled_width = input_width - 1
    
    # After Conv (stride=2, padding=1): ((H-1+2-3)/2+1, (W-1+2-3)/2+1, C_out)
    output_height = (pooled_height + 2 - 3) // 2 + 1
    output_width = (pooled_width + 2 - 3) // 2 + 1

    # Calculate tensor sizes
    input_size = input_height * input_width * input_channels
    temp_size = pooled_height * pooled_width * input_channels
    weight_size = output_channels * input_channels * 3 * 3
    bn_param_size = 4 * output_channels  # weight, bias, mean, var
    total_weight_size = weight_size + bn_param_size
    output_size = output_height * output_width * output_channels

    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    temp_ty = np.ndarray[(temp_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    # AIE Core Function declaration
    aconv_kernel = Kernel(
        "aconv_bf16",
        "aconv_bf16.o",
        [
            input_ty,   # input
            weight_ty,  # weights
            output_ty,  # output
            temp_ty,    # temp buffer
            np.int32,   # input_height
            np.int32,   # input_width
            np.int32,   # input_channels
            np.int32,   # output_channels
        ],
    )

    # AIE-array data movement with object fifos
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Local buffer for intermediate pooled output
    temp_buffer = Buffer(temp_ty, name="temp_pooled")

    # Task for the core to perform
    def core_fn(of_in, of_wts, of_out, temp_buf, kernel):
        # Acquire input and output buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the fused AConv kernel
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            temp_buf,
            input_height,
            input_width,
            input_channels,
            output_channels,
        )

        # Release buffers
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(
        core_fn,
        [
            of_input_L3L2.cons(),
            of_weights_L3L2.cons(),
            of_output_L2L3.prod(),
            temp_buffer,
            aconv_kernel,
        ],
    )

    # Runtime operations
    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_input_L3L2.prod(), I)
        rt.fill(of_weights_L3L2.prod(), W)
        rt.drain(of_output_L2L3.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse command line arguments
try:
    device_name = str(sys.argv[1])
    if device_name != "npu2":
        raise ValueError(f"[ERROR] Device name {device_name} not supported")
    
    dev = NPU2Col1()
    
    input_height = int(sys.argv[2])
    input_width = int(sys.argv[3])
    input_channels = int(sys.argv[4])
    output_channels = int(sys.argv[5])
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out>")
    print("Example: python aie2.py npu2 8 8 8 8")
    sys.exit(1)

module = aconv_layer_bf16(dev, input_height, input_width, input_channels, output_channels)
print(module)
