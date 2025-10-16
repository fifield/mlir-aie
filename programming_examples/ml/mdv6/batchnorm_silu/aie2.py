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


def batchnorm_silu_layer_bf16(
    dev,
    height: int,
    width: int,
    channels: int,
    use_silu: bool = True,
):
    """
    BatchNorm + SiLU layer implementation for MDV6 on AIE2P.
    
    This implements the fused BatchNorm + SiLU activation used after
    every Conv layer in MDV6.
    """

    # Calculate tensor sizes
    input_size = height * width * channels
    bn_param_size = channels  # weight and bias are per-channel
    output_size = height * width * channels

    # Type definitions (bfloat16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]  # bf16 as uint16
    # Combine BN weight and bias into single buffer (workaround for 2-input limit)
    bn_params_ty = np.ndarray[(2 * bn_param_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    # AIE Core Function declarations
    if use_silu:
        kernel = Kernel(
            "batchnorm_silu_bf16",
            "batchnorm_silu_bf16.o",
            [
                input_ty,      # input pointer (bf16)
                bn_params_ty,  # bn params pointer (weight + bias concatenated)
                output_ty,     # output pointer (bf16)
                np.int32,      # height
                np.int32,      # width
                np.int32,      # channels
            ],
        )
    else:
        kernel = Kernel(
            "batchnorm_bf16",
            "batchnorm_silu_bf16.o",
            [
                input_ty,      # input pointer (bf16)
                bn_params_ty,  # bn params pointer (weight + bias concatenated)
                output_ty,     # output pointer (bf16)
                np.int32,      # height
                np.int32,      # width
                np.int32,      # channels
            ],
        )

    # AIE-array data movement with object fifos
    # Input activations from L3 to L2 to core
    of_input_L3L2 = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    
    # BN params (weight + bias) from L3 to L2 to core
    of_bn_params_L3L2 = ObjectFifo(bn_params_ty, depth=1, name="bn_params_L3L2")
    
    # Output from core to L2 to L3
    of_output_L2L3 = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Task for the core to perform
    def core_fn(of_in, of_bn_params, of_out, kernel_fn):
        # Acquire buffers
        elem_in = of_in.acquire(1)
        elem_bn_params = of_bn_params.acquire(1)
        elem_out = of_out.acquire(1)

        # Call the kernel (it will split params into weight and bias internally)
        kernel_fn(
            elem_in,
            elem_bn_params,
            elem_out,
            height,
            width,
            channels,
        )

        # Release buffers
        of_in.release(1)
        of_bn_params.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(
        core_fn,
        [
            of_input_L3L2.cons(),
            of_bn_params_L3L2.cons(),
            of_output_L2L3.prod(),
            kernel,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(input_ty, bn_params_ty, output_ty) as (I, P, O):
        # Start worker
        rt.start(worker)

        # Fill input and parameter ObjectFifos, drain output
        rt.fill(of_input_L3L2.prod(), I)
        rt.fill(of_bn_params_L3L2.prod(), P)
        rt.drain(of_output_L2L3.cons(), O, wait=True)

    # Place components and generate MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse command line arguments
try:
    device_name = str(sys.argv[1])
    if device_name != "npu2":
        raise ValueError(f"[ERROR] Device name {device_name} not supported. Use 'npu2'")
    
    dev = NPU2Col1()
    
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    channels = int(sys.argv[4])
    use_silu = True if len(sys.argv) <= 5 else (sys.argv[5].lower() == "true")
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <H> <W> <C> [use_silu]")
    print("Example: python aie2.py npu2 8 8 8 true")
    sys.exit(1)

module = batchnorm_silu_layer_bf16(dev, height, width, channels, use_silu)
print(module)
