# repncsp_elan/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

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


def repncsp_elan_bf16(
    dev,
    height=8,
    width=8,
    in_channels=32,
    out_channels=32,
    part_channels=32,
    process_channels=None,
):
    """
    RepNCSPELAN layer for MDV6 on AIE2P.
    
    RepNCSPELAN = RepNCSP + ELAN structure
    Most complex layer in MDV6 with nested RepNCSP blocks
    
    Architecture:
        Input → Conv1 → split → [x1, x2]
                                   ↓
                               x2 → RepNCSP → Conv3x3 → x3
                                                   ↓
                                               x3 → RepNCSP → Conv3x3 → x4
                                   ↓              ↓              ↓
                        Concat [x1, x2, x3, x4]
                                   ↓
                               Conv4 → Output
    
    Default: process_channels = part_channels // 2
    """

    # Calculate channel dimensions
    if process_channels is None:
        process_channels = part_channels // 2
    
    half_part = part_channels // 2
    concat_channels = part_channels + 2 * process_channels
    
    # RepNCSP neck channels (csp_expand = 0.5)
    rn1_neck = process_channels // 2
    rn2_neck = process_channels // 2

    # Calculate tensor sizes
    input_size = height * width * in_channels
    output_size = height * width * out_channels
    conv1_size = height * width * part_channels
    repncsp_size = height * width * process_channels
    concat_size = height * width * concat_channels
    
    # RepNCSP internal sizes
    rn1_neck_size = height * width * rn1_neck
    rn1_concat_size = height * width * 2 * rn1_neck
    rn2_neck_size = height * width * rn2_neck
    rn2_concat_size = height * width * 2 * rn2_neck
    
    # Weight sizes calculation
    # Conv1: in_channels → part_channels (1×1)
    conv1_weight_size = part_channels * in_channels
    conv1_bn_size = 4 * part_channels
    
    # RepNCSP #1: half_part → process_channels
    rn1_conv1_wsize = rn1_neck * half_part
    rn1_conv1_bn_size = 4 * rn1_neck
    rn1_bn_conv3x3_wsize = rn1_neck * rn1_neck * 9
    rn1_bn_bn3x3_size = 4 * rn1_neck
    rn1_bn_conv1x1_wsize = rn1_neck * rn1_neck
    rn1_bn_bn1x1_size = 4 * rn1_neck
    rn1_bn_conv2_wsize = rn1_neck * rn1_neck * 9
    rn1_bn_bn2_size = 4 * rn1_neck
    rn1_conv2_wsize = rn1_neck * half_part
    rn1_conv2_bn_size = 4 * rn1_neck
    rn1_conv3_wsize = process_channels * 2 * rn1_neck
    rn1_conv3_bn_size = 4 * process_channels
    
    rn1_total = (rn1_conv1_wsize + rn1_conv1_bn_size +
                 rn1_bn_conv3x3_wsize + rn1_bn_bn3x3_size +
                 rn1_bn_conv1x1_wsize + rn1_bn_bn1x1_size +
                 rn1_bn_conv2_wsize + rn1_bn_bn2_size +
                 rn1_conv2_wsize + rn1_conv2_bn_size +
                 rn1_conv3_wsize + rn1_conv3_bn_size)
    
    # Conv3x3 #1: process_channels → process_channels (3×3)
    conv3x3_1_wsize = process_channels * process_channels * 9
    conv3x3_1_bn_size = 4 * process_channels
    
    # RepNCSP #2: process_channels → process_channels
    rn2_conv1_wsize = rn2_neck * process_channels
    rn2_conv1_bn_size = 4 * rn2_neck
    rn2_bn_conv3x3_wsize = rn2_neck * rn2_neck * 9
    rn2_bn_bn3x3_size = 4 * rn2_neck
    rn2_bn_conv1x1_wsize = rn2_neck * rn2_neck
    rn2_bn_bn1x1_size = 4 * rn2_neck
    rn2_bn_conv2_wsize = rn2_neck * rn2_neck * 9
    rn2_bn_bn2_size = 4 * rn2_neck
    rn2_conv2_wsize = rn2_neck * process_channels
    rn2_conv2_bn_size = 4 * rn2_neck
    rn2_conv3_wsize = process_channels * 2 * rn2_neck
    rn2_conv3_bn_size = 4 * process_channels
    
    rn2_total = (rn2_conv1_wsize + rn2_conv1_bn_size +
                 rn2_bn_conv3x3_wsize + rn2_bn_bn3x3_size +
                 rn2_bn_conv1x1_wsize + rn2_bn_bn1x1_size +
                 rn2_bn_conv2_wsize + rn2_bn_bn2_size +
                 rn2_conv2_wsize + rn2_conv2_bn_size +
                 rn2_conv3_wsize + rn2_conv3_bn_size)
    
    # Conv3x3 #2: process_channels → process_channels (3×3)
    conv3x3_2_wsize = process_channels * process_channels * 9
    conv3x3_2_bn_size = 4 * process_channels
    
    # Conv4: concat_channels → out_channels (1×1)
    conv4_wsize = out_channels * concat_channels
    conv4_bn_size = 4 * out_channels
    
    total_weight_size = (conv1_weight_size + conv1_bn_size +
                         rn1_total +
                         conv3x3_1_wsize + conv3x3_1_bn_size +
                         rn2_total +
                         conv3x3_2_wsize + conv3x3_2_bn_size +
                         conv4_wsize + conv4_bn_size)

    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    repncsp_ty = np.ndarray[(repncsp_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]
    rn1_neck_ty = np.ndarray[(rn1_neck_size,), np.dtype[np.uint16]]
    rn1_concat_ty = np.ndarray[(rn1_concat_size,), np.dtype[np.uint16]]
    rn2_neck_ty = np.ndarray[(rn2_neck_size,), np.dtype[np.uint16]]
    rn2_concat_ty = np.ndarray[(rn2_concat_size,), np.dtype[np.uint16]]

    # AIE Core Function declaration
    repncsp_elan_kernel = Kernel(
        "repncsp_elan_bf16",
        "repncsp_elan_bf16.o",
        [
            input_ty,       # input
            weight_ty,      # weights_and_bn
            output_ty,      # output
            conv1_ty,       # conv1_output
            repncsp_ty,     # x3_repncsp_out
            repncsp_ty,     # x3_conv_out
            repncsp_ty,     # x4_repncsp_out
            repncsp_ty,     # x4_conv_out
            concat_ty,      # concat_buffer
            rn1_neck_ty,    # rn1_conv1_out
            rn1_neck_ty,    # rn1_bottleneck_out
            rn1_neck_ty,    # rn1_conv2_out
            rn1_concat_ty,  # rn1_concat
            rn1_neck_ty,    # rn1_bn_input_copy
            rn1_neck_ty,    # rn1_bn_temp1
            rn1_neck_ty,    # rn1_bn_temp2
            rn1_neck_ty,    # rn1_bn_temp3
            rn1_neck_ty,    # rn1_bn_temp4
            rn2_neck_ty,    # rn2_conv1_out
            rn2_neck_ty,    # rn2_bottleneck_out
            rn2_neck_ty,    # rn2_conv2_out
            rn2_concat_ty,  # rn2_concat
            rn2_neck_ty,    # rn2_bn_input_copy
            rn2_neck_ty,    # rn2_bn_temp1
            rn2_neck_ty,    # rn2_bn_temp2
            rn2_neck_ty,    # rn2_bn_temp3
            rn2_neck_ty,    # rn2_bn_temp4
            np.int32,       # height
            np.int32,       # width
            np.int32,       # in_channels
            np.int32,       # out_channels
            np.int32,       # part_channels
            np.int32,       # process_channels
        ],
    )

    # AIE-array data movement with object fifos
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights_L3L2 = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Local buffers for intermediate results (created at top level, placed by Worker)
    conv1_output_buffer = Buffer(conv1_ty, name="conv1_output")
    # x3 and x4 share buffers (x3 repncsp output is consumed by x3 conv before x4 starts)
    x3_repncsp_buffer = Buffer(repncsp_ty, name="x3x4_repncsp_out")
    x3_conv_buffer = Buffer(repncsp_ty, name="x3x4_conv_out")
    x4_repncsp_buffer = x3_repncsp_buffer  # shared
    x4_conv_buffer = x3_conv_buffer        # shared
    concat_buffer = Buffer(concat_ty, name="concat_buffer")

    # RepNCSP #1 buffers
    rn1_conv1_buffer = Buffer(rn1_neck_ty, name="rn1_conv1_out")
    rn1_bottleneck_buffer = Buffer(rn1_neck_ty, name="rn1_bottleneck_out")
    rn1_conv2_buffer = Buffer(rn1_neck_ty, name="rn1_conv2_out")
    rn1_concat_buffer = Buffer(rn1_concat_ty, name="rn1_concat")
    rn1_bn_input_copy_buffer = Buffer(rn1_neck_ty, name="rn1_bn_input_copy")
    rn1_bn_temp1_buffer = Buffer(rn1_neck_ty, name="rn1_bn_temp1")
    rn1_bn_temp2_buffer = Buffer(rn1_neck_ty, name="rn1_bn_temp2")
    rn1_bn_temp3_buffer = Buffer(rn1_neck_ty, name="rn1_bn_temp3")
    rn1_bn_temp4_buffer = Buffer(rn1_neck_ty, name="rn1_bn_temp4")

    # RepNCSP #2 shares buffers with #1 (sequential execution, same sizes)
    rn2_conv1_buffer = rn1_conv1_buffer
    rn2_bottleneck_buffer = rn1_bottleneck_buffer
    rn2_conv2_buffer = rn1_conv2_buffer
    rn2_concat_buffer = rn1_concat_buffer
    rn2_bn_input_copy_buffer = rn1_bn_input_copy_buffer
    rn2_bn_temp1_buffer = rn1_bn_temp1_buffer
    rn2_bn_temp2_buffer = rn1_bn_temp2_buffer
    rn2_bn_temp3_buffer = rn1_bn_temp3_buffer
    rn2_bn_temp4_buffer = rn1_bn_temp4_buffer

    # Task for the core to perform
    # rn1_* and rn2_* share the same buffers (sequential execution)
    def core_fn(of_in, of_wts, of_out, kernel, conv1_out, x3x4_repncsp, x3x4_conv, concat_buf, rn_conv1, rn_bottleneck, rn_conv2, rn_concat, rn_bn_input_copy, rn_bn_temp1, rn_bn_temp2, rn_bn_temp3, rn_bn_temp4):
        # Acquire input and output buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the RepNCSPELAN kernel
        # Pass shared rn_* buffers for both RepNCSP #1 and #2
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            conv1_out,
            x3x4_repncsp,
            x3x4_conv,
            x3x4_repncsp,  # x4 reuses x3 buffers
            x3x4_conv,
            concat_buf,
            rn_conv1,        # rn1 buffers
            rn_bottleneck,
            rn_conv2,
            rn_concat,
            rn_bn_input_copy,
            rn_bn_temp1,
            rn_bn_temp2,
            rn_bn_temp3,
            rn_bn_temp4,
            rn_conv1,        # rn2 reuses same buffers
            rn_bottleneck,
            rn_conv2,
            rn_concat,
            rn_bn_input_copy,
            rn_bn_temp1,
            rn_bn_temp2,
            rn_bn_temp3,
            rn_bn_temp4,
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
    # Only pass unique buffers (rn1 and rn2 share)
    worker = Worker(
        core_fn,
        [
            of_input_L3L2.cons(),
            of_weights_L3L2.cons(),
            of_output_L2L3.prod(),
            repncsp_elan_kernel,
            conv1_output_buffer,
            x3_repncsp_buffer,
            x3_conv_buffer,
            concat_buffer,
            rn1_conv1_buffer,
            rn1_bottleneck_buffer,
            rn1_conv2_buffer,
            rn1_concat_buffer,
            rn1_bn_input_copy_buffer,
            rn1_bn_temp1_buffer,
            rn1_bn_temp2_buffer,
            rn1_bn_temp3_buffer,
            rn1_bn_temp4_buffer,
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
    part_channels = int(sys.argv[6])
    process_channels = int(sys.argv[7]) if len(sys.argv) > 7 else part_channels // 2
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out> <part_ch> [process_ch]")
    print("Example: python aie2.py npu2 8 8 32 32 32 16")
    sys.exit(1)

module = repncsp_elan_bf16(dev, height, width, in_channels, out_channels, 
                           part_channels, process_channels)
print(module)
