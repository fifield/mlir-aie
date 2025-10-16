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


def repncsp_layer_bf16(
    dev,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    csp_expand: float = 0.5,
):
    """
    RepNCSP layer for MDV6 on AIE2P.
    
    RepNCSP = CSP block with RepConv bottlenecks
    Architecture: Conv1 → Bottleneck → x1 + Conv2 → x2 → Concat → Conv3
    
    Default: csp_expand=0.5, repeat_num=1, kernel_size=1
    """

    # Calculate channel dimensions
    neck_channels = int(out_channels * csp_expand)
    concat_channels = 2 * neck_channels

    # Calculate tensor sizes
    input_size = height * width * in_channels
    output_size = height * width * out_channels
    neck_size = height * width * neck_channels
    concat_size = height * width * concat_channels
    
    # Weight sizes
    # Conv1: in_channels → neck_channels (1×1)
    conv1_weight_size = neck_channels * in_channels
    conv1_bn_size = 4 * neck_channels
    
    # Bottleneck: neck_channels → neck_channels
    bn_conv3x3_size = neck_channels * neck_channels * 3 * 3
    bn_conv1x1_size = neck_channels * neck_channels * 1 * 1
    bn_bn_params = 4 * neck_channels
    bn_conv2_size = neck_channels * neck_channels * 3 * 3
    bottleneck_weight_size = (bn_conv3x3_size + bn_bn_params + 
                              bn_conv1x1_size + bn_bn_params +
                              bn_conv2_size + bn_bn_params)
    
    # Conv2: in_channels → neck_channels (1×1)
    conv2_weight_size = neck_channels * in_channels
    conv2_bn_size = 4 * neck_channels
    
    # Conv3: concat_channels → out_channels (1×1)
    conv3_weight_size = out_channels * concat_channels
    conv3_bn_size = 4 * out_channels
    
    total_weight_size = (conv1_weight_size + conv1_bn_size +
                         bottleneck_weight_size +
                         conv2_weight_size + conv2_bn_size +
                         conv3_weight_size + conv3_bn_size)

    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    neck_ty = np.ndarray[(neck_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]

    # AIE Core Function declaration
    repncsp_kernel = Kernel(
        "repncsp_bf16",
        "repncsp_bf16.o",
        [
            input_ty,    # input
            weight_ty,   # weights_and_bn
            output_ty,   # output
            neck_ty,     # x1_conv1
            neck_ty,     # x1_bottleneck
            neck_ty,     # x2_conv2
            concat_ty,   # concat_buffer
            neck_ty,     # bn_input_copy
            neck_ty,     # bn_temp1
            neck_ty,     # bn_temp2
            neck_ty,     # bn_temp3
            neck_ty,     # bn_temp4
            np.int32,    # height
            np.int32,    # width
            np.int32,    # in_channels
            np.int32,    # out_channels
            np.float32,  # csp_expand
        ],
    )

    # AIE-array data movement with object fifos
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Task for the core to perform
    def core_fn(of_in, of_wts, of_out, kernel):
        # Allocate local buffers for intermediate results
        x1_conv1_buffer = LocalBuffer(neck_ty, name="x1_conv1")
        x1_bottleneck_buffer = LocalBuffer(neck_ty, name="x1_bottleneck")
        x2_conv2_buffer = LocalBuffer(neck_ty, name="x2_conv2")
        concat_buffer = LocalBuffer(concat_ty, name="concat_buffer")
        
        # Bottleneck internal buffers
        bn_input_copy_buffer = LocalBuffer(neck_ty, name="bn_input_copy")
        bn_temp1_buffer = LocalBuffer(neck_ty, name="bn_temp1")
        bn_temp2_buffer = LocalBuffer(neck_ty, name="bn_temp2")
        bn_temp3_buffer = LocalBuffer(neck_ty, name="bn_temp3")
        bn_temp4_buffer = LocalBuffer(neck_ty, name="bn_temp4")
        
        # Acquire input and output buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the RepNCSP kernel
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            x1_conv1_buffer,
            x1_bottleneck_buffer,
            x2_conv2_buffer,
            concat_buffer,
            bn_input_copy_buffer,
            bn_temp1_buffer,
            bn_temp2_buffer,
            bn_temp3_buffer,
            bn_temp4_buffer,
            height,
            width,
            in_channels,
            out_channels,
            csp_expand,
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
            repncsp_kernel,
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
    csp_expand = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out> [csp_expand]")
    print("Example: python aie2.py npu2 8 8 16 16 0.5")
    sys.exit(1)

module = repncsp_layer_bf16(dev, height, width, in_channels, out_channels, csp_expand)
print(module)
