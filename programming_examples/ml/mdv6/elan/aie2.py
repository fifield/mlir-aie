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
    LocalBuffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def elan_layer_bf16(
    dev,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    part_channels: int,
    process_channels: int = None,
):
    """
    ELAN layer for MDV6 on AIE2P.
    
    ELAN = Efficient Layer Aggregation Network
    Architecture: Conv1 → split → Conv2 → Conv3 → 4-way concat → Conv4
    
    Default: process_channels = part_channels // 2
    """

    # Calculate channel dimensions
    if process_channels is None:
        process_channels = part_channels // 2
    
    half_part = part_channels // 2
    concat_channels = part_channels + 2 * process_channels

    # Calculate tensor sizes
    input_size = height * width * in_channels
    output_size = height * width * out_channels
    conv1_size = height * width * part_channels
    process_size = height * width * process_channels
    concat_size = height * width * concat_channels
    
    # Weight sizes
    # Conv1: in_channels → part_channels (1×1)
    conv1_weight_size = part_channels * in_channels
    conv1_bn_size = 4 * part_channels
    
    # Conv2: half_part → process_channels (3×3)
    conv2_weight_size = process_channels * half_part * 3 * 3
    conv2_bn_size = 4 * process_channels
    
    # Conv3: process_channels → process_channels (3×3)
    conv3_weight_size = process_channels * process_channels * 3 * 3
    conv3_bn_size = 4 * process_channels
    
    # Conv4: concat_channels → out_channels (1×1)
    conv4_weight_size = out_channels * concat_channels
    conv4_bn_size = 4 * out_channels
    
    total_weight_size = (conv1_weight_size + conv1_bn_size +
                         conv2_weight_size + conv2_bn_size +
                         conv3_weight_size + conv3_bn_size +
                         conv4_weight_size + conv4_bn_size)

    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    process_ty = np.ndarray[(process_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]

    # AIE Core Function declaration
    elan_kernel = Kernel(
        "elan_bf16",
        "elan_bf16.o",
        [
            input_ty,    # input
            weight_ty,   # weights_and_bn
            output_ty,   # output
            conv1_ty,    # conv1_output (contains x1 and x2)
            process_ty,  # x3
            process_ty,  # x4
            concat_ty,   # concat_buffer
            np.int32,    # height
            np.int32,    # width
            np.int32,    # in_channels
            np.int32,    # out_channels
            np.int32,    # part_channels
            np.int32,    # process_channels
        ],
    )

    # AIE-array data movement with object fifos
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Task for the core to perform
    def core_fn(of_in, of_wts, of_out, kernel):
        # Allocate local buffers for intermediate results
        conv1_output_buffer = LocalBuffer(conv1_ty, name="conv1_output")
        x3_buffer = LocalBuffer(process_ty, name="x3")
        x4_buffer = LocalBuffer(process_ty, name="x4")
        concat_buffer = LocalBuffer(concat_ty, name="concat_buffer")
        
        # Acquire input and output buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the ELAN kernel
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            conv1_output_buffer,
            x3_buffer,
            x4_buffer,
            concat_buffer,
            height,
            width,
            in_channels,
            out_channels,
            part_channels,
            process_channels,
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
            elan_kernel,
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
    
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    in_channels = int(sys.argv[4])
    out_channels = int(sys.argv[5])
    part_channels = int(sys.argv[6])
    process_channels = int(sys.argv[7]) if len(sys.argv) > 7 else part_channels // 2
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out> <part_ch> [process_ch]")
    print("Example: python aie2.py npu2 8 8 32 32 32 16")
    sys.exit(1)

module = elan_layer_bf16(dev, height, width, in_channels, out_channels, 
                         part_channels, process_channels)
print(module)
