# sppelan/aie2.py -*- Python -*-
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
    LocalBuffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1


def sppelan_bf16(
    dev,
    height=8,
    width=8,
    in_channels=16,
    out_channels=16,
    neck_channels=8,
    kernel_size=5,
    stride=1,
    padding=2,
):
    """
    SPPELAN (Spatial Pyramid Pooling ELAN) layer for MDV6
    
    Architecture:
        Input → Conv1 (1×1) → f0
                    ↓
                MaxPool → f1
                    ↓
                MaxPool → f2
                    ↓
                MaxPool → f3
                    ↓
            Concat [f0,f1,f2,f3]
                    ↓
                Conv5 (1×1) → Output
    
    Args:
        height: Input height
        width: Input width
        in_channels: Input channels
        out_channels: Output channels
        neck_channels: Intermediate channels (typically in_channels // 2)
        kernel_size: MaxPool kernel size (default 5)
        stride: MaxPool stride (default 1)
        padding: MaxPool padding (default 2)
    """
    
    # Calculate sizes
    input_size = height * width * in_channels
    conv1_size = height * width * neck_channels
    pool_size = height * width * neck_channels  # Same size (stride=1, padding=2)
    concat_size = height * width * 4 * neck_channels
    output_size = height * width * out_channels
    
    # Calculate weight sizes
    # Conv1: in_channels → neck_channels (1×1)
    conv1_weight_size = neck_channels * in_channels
    conv1_bn_size = 4 * neck_channels  # weight, bias, mean, var
    
    # Conv5: 4*neck_channels → out_channels (1×1)
    conv5_weight_size = out_channels * 4 * neck_channels
    conv5_bn_size = 4 * out_channels
    
    total_weight_size = conv1_weight_size + conv1_bn_size + conv5_weight_size + conv5_bn_size
    
    # Type definitions (bfloat16 as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    pool_ty = np.ndarray[(pool_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]
    
    # Declare kernel
    sppelan_kernel = Kernel(
        "sppelan_bf16",
        "sppelan_bf16.o",
        [
            input_ty,    # input
            weight_ty,   # weights_and_bn
            output_ty,   # output
            conv1_ty,    # conv1_output
            pool_ty,     # pool1_output
            pool_ty,     # pool2_output
            pool_ty,     # pool3_output
            concat_ty,   # concat_buffer
            np.int32,    # height
            np.int32,    # width
            np.int32,    # in_channels
            np.int32,    # out_channels
            np.int32,    # neck_channels
            np.int32,    # kernel_size
            np.int32,    # stride
            np.int32,    # padding
        ],
    )
    
    # Create ObjectFIFOs for data movement (depth=1, single buffering)
    of_in = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_wts = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_out = ObjectFifo(output_ty, depth=1, name="output_L2L3")
    
    def core_fn(of_in, of_wts, of_out, kernel):
        # Allocate LocalBuffers for intermediate results
        conv1_buf = LocalBuffer(conv1_ty, name="conv1_output")
        pool1_buf = LocalBuffer(pool_ty, name="pool1_output")
        pool2_buf = LocalBuffer(pool_ty, name="pool2_output")
        pool3_buf = LocalBuffer(pool_ty, name="pool3_output")
        concat_buf = LocalBuffer(concat_ty, name="concat_buffer")
        
        # Acquire input and weight buffers
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        
        # Call kernel
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            conv1_buf,
            pool1_buf,
            pool2_buf,
            pool3_buf,
            concat_buf,
            height,
            width,
            in_channels,
            out_channels,
            neck_channels,
            kernel_size,
            stride,
            padding,
        )
        
        # Release buffers
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)
    
    # Create worker
    worker = Worker(
        core_fn,
        [of_in.cons(), of_wts.cons(), of_out.prod(), sppelan_kernel],
    )
    
    # Create runtime sequence
    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_in.prod(), I)
        rt.fill(of_wts.prod(), W)
        rt.drain(of_out.cons(), O, wait=True)
    
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
    neck_channels = in_channels // 2  # Default
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C_in> <C_out>")
    print("Example: python aie2.py npu2 8 8 16 16")
    sys.exit(1)

module = sppelan_bf16(dev, height, width, in_channels, out_channels, neck_channels)
print(module)
