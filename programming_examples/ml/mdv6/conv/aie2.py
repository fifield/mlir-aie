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
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def conv_layer_bf16(
    dev,
    input_height: int,
    input_width: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    """
    Simple Conv layer implementation for MDV6 on AIE2P.
    
    This is a basic single-tile implementation to start with.
    We'll optimize and parallelize later.
    """

    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Calculate tensor sizes
    input_size = input_height * input_width * input_channels
    weight_size = output_channels * input_channels * kernel_size * kernel_size
    output_size = output_height * output_width * output_channels

    # Type definitions (bfloat16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]  # bf16 as uint16
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    # AIE Core Function declarations
    if kernel_size == 3:
        conv_kernel = Kernel(
            "conv3x3_bf16",
            "rep_elan_bf16.o",
            [
                input_ty,   # input pointer (bf16)
                weight_ty,  # weight pointer (bf16)
                output_ty,  # output pointer (bf16)
                np.int32,   # input_height
                np.int32,   # input_width
                np.int32,   # input_channels
                np.int32,   # output_channels
                np.int32,   # stride
                np.int32,   # padding
            ],
        )
    elif kernel_size == 1:
        conv_kernel = Kernel(
            "conv1x1_bf16",
            "rep_elan_bf16.o",
            [
                input_ty,   # input pointer (bf16)
                weight_ty,  # weight pointer (bf16)
                output_ty,  # output pointer (bf16)
                np.int32,   # input_height
                np.int32,   # input_width
                np.int32,   # input_channels
                np.int32,   # output_channels
            ],
        )
    else:
        raise ValueError(f"Kernel size {kernel_size} not supported yet")

    # AIE-array data movement with object fifos
    # Input activations from L3 to L2 to core
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    
    # Weights from L3 to L2 to core
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    
    # Output from core to L2 to L3
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Task for the core to perform
    def core_fn(of_in, of_wts, of_out, kernel):
        # Acquire input and weight buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the convolution kernel
        if kernel_size == 3:
            kernel(
                elem_in,
                elem_wts,
                elem_out,
                input_height,
                input_width,
                input_channels,
                output_channels,
                stride,
                padding,
            )
        elif kernel_size == 1:
            kernel(
                elem_in,
                elem_wts,
                elem_out,
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
            conv_kernel,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (I, W, O):
        # Start worker
        rt.start(worker)

        # Fill input and weight ObjectFifos, drain output
        rt.fill(of_input_L3L2.prod(), I)
        rt.fill(of_weights_L3L2.prod(), W)
        rt.drain(of_output_L2L3.cons(), O, wait=True)

    # Place components and generate MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse command line arguments
try:
    device_name = str(sys.argv[1])
    if device_name != "npu2":
        raise ValueError(f"[ERROR] Device name {device_name} not supported. Use 'npu2'")
    
    dev = NPU2Col1()
    
    input_height = int(sys.argv[2])
    input_width = int(sys.argv[3])
    input_channels = int(sys.argv[4])
    output_channels = int(sys.argv[5])
    kernel_size = int(sys.argv[6])
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    padding = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    
    # Validate parameters
    if kernel_size not in [1, 3]:
        raise ValueError(f"Kernel size must be 1 or 3, got {kernel_size}")
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out> <K> [stride] [padding]")
    print("Example: python aie2.py npu2 32 32 32 32 3 1 1")
    sys.exit(1)

module = conv_layer_bf16(
    dev,
    input_height,
    input_width,
    input_channels,
    output_channels,
    kernel_size,
    stride,
    padding,
)
print(module)
