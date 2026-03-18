#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import numpy as np
import sys

from aie.iron import (
    Kernel,
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def bottleneck_layer_bf16(
    dev,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 1,
    residual: int = 1,
):
    """
    Bottleneck layer for MDV6 on AIE2P.
    
    Bottleneck = RepConv → Conv+BN+SiLU → optional residual add
    
    With expand=1.0, neck_channels = out_channels
    Residual only active when in_channels == out_channels
    """

    # Calculate output dimensions
    output_height = (height + 2 * padding - 3) // stride + 1
    output_width = (width + 2 * padding - 3) // stride + 1

    # Calculate tensor sizes
    input_size = height * width * in_channels
    output_size = output_height * output_width * out_channels
    temp_size = output_size  # All temp buffers same size
    
    # Weight sizes (with expand=1.0, neck_channels = out_channels)
    # RepConv weights
    conv3x3_weight_size = out_channels * in_channels * 3 * 3
    conv1x1_weight_size = out_channels * in_channels * 1 * 1
    bn_param_size = 4 * out_channels
    repconv_weight_size = conv3x3_weight_size + bn_param_size + conv1x1_weight_size + bn_param_size
    
    # Conv2 weights (out_channels → out_channels)
    conv2_weight_size = out_channels * out_channels * 3 * 3
    conv2_bn_size = bn_param_size
    
    total_weight_size = repconv_weight_size + conv2_weight_size + conv2_bn_size

    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    temp_ty = np.ndarray[(temp_size,), np.dtype[np.uint16]]
    input_copy_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]

    # AIE Core Function declaration
    bottleneck_kernel = Kernel(
        "bottleneck_bf16",
        "bottleneck_bf16.o",
        [
            input_ty,       # input
            weight_ty,      # weights_and_bn
            output_ty,      # output
            input_copy_ty,  # input_copy (for residual)
            temp_ty,        # temp1
            temp_ty,        # temp2
            temp_ty,        # temp3
            temp_ty,        # temp4
            np.int32,       # height
            np.int32,       # width
            np.int32,       # in_channels
            np.int32,       # out_channels
            np.int32,       # stride
            np.int32,       # padding
            np.int32,       # residual
        ],
    )

    # AIE-array data movement with object fifos
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Local buffers for intermediate results (created at top level, placed by Worker)
    input_copy_buffer = Buffer(input_copy_ty, name="input_copy")
    temp1_buffer = Buffer(temp_ty, name="temp1_conv3x3")
    temp2_buffer = Buffer(temp_ty, name="temp2_conv1x1")
    temp3_buffer = Buffer(temp_ty, name="temp3_repconv")
    temp4_buffer = Buffer(temp_ty, name="temp4_conv2")

    # Task for the core to perform
    def core_fn(of_in, of_wts, of_out, kernel, input_copy_buffer, temp1_buffer, temp2_buffer, temp3_buffer, temp4_buffer):
        # Acquire input and output buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the Bottleneck kernel
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            input_copy_buffer,
            temp1_buffer,
            temp2_buffer,
            temp3_buffer,
            temp4_buffer,
            height,
            width,
            in_channels,
            out_channels,
            stride,
            padding,
            residual,
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
            bottleneck_kernel,
            input_copy_buffer,
            temp1_buffer,
            temp2_buffer,
            temp3_buffer,
            temp4_buffer,
        ],
        stack_size=4096,
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
    
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    in_channels = int(sys.argv[4])
    out_channels = int(sys.argv[5])
    stride = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    padding = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    residual = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out> [stride] [padding] [residual]")
    print("Example: python aie2.py npu2 8 8 8 8 1 1 1")
    sys.exit(1)

module = bottleneck_layer_bf16(dev, height, width, in_channels, out_channels, stride, padding, residual)
print(module)
