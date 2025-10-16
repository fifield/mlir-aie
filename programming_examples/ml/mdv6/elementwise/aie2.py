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


def elementwise_layer_bf16(
    dev,
    size: int,
    operation: str = "add",
):
    """
    Element-wise operations for MDV6 on AIE2P.
    
    Supports: add, max, mul
    Used for residual connections and RepConv.
    """

    # Type definitions (bfloat16)
    tensor_ty = np.ndarray[(size,), np.dtype[np.uint16]]  # bf16 as uint16

    # AIE Core Function declarations
    if operation == "add":
        kernel = Kernel(
            "add_bf16",
            "elementwise_bf16.o",
            [tensor_ty, tensor_ty, tensor_ty, np.int32],
        )
    elif operation == "max":
        kernel = Kernel(
            "max_bf16",
            "elementwise_bf16.o",
            [tensor_ty, tensor_ty, tensor_ty, np.int32],
        )
    elif operation == "mul":
        kernel = Kernel(
            "mul_bf16",
            "elementwise_bf16.o",
            [tensor_ty, tensor_ty, tensor_ty, np.int32],
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # AIE-array data movement with object fifos
    of_input_a_L3L2 = ObjectFifo(tensor_ty, depth=1, name="input_a_L3L2")
    of_input_b_L3L2 = ObjectFifo(tensor_ty, depth=1, name="input_b_L3L2")
    of_output_L2L3 = ObjectFifo(tensor_ty, depth=1, name="output_L2L3")

    # Task for the core to perform
    def core_fn(of_a, of_b, of_out, kernel_fn):
        elem_a = of_a.acquire(1)
        elem_b = of_b.acquire(1)
        elem_out = of_out.acquire(1)

        kernel_fn(elem_a, elem_b, elem_out, size)

        of_a.release(1)
        of_b.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(
        core_fn,
        [
            of_input_a_L3L2.cons(),
            of_input_b_L3L2.cons(),
            of_output_L2L3.prod(),
            kernel,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, O):
        rt.start(worker)
        rt.fill(of_input_a_L3L2.prod(), A)
        rt.fill(of_input_b_L3L2.prod(), B)
        rt.drain(of_output_L2L3.cons(), O, wait=True)

    # Place components and generate MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


# Parse command line arguments
try:
    device_name = str(sys.argv[1])
    if device_name != "npu2":
        raise ValueError(f"[ERROR] Device name {device_name} not supported. Use 'npu2'")
    
    dev = NPU2Col1()
    
    size = int(sys.argv[2])
    operation = str(sys.argv[3]) if len(sys.argv) > 3 else "add"
    
    if operation not in ["add", "max", "mul"]:
        raise ValueError(f"Unknown operation: {operation}. Use add, max, or mul")
    
except (IndexError, ValueError) as e:
    print(f"Error: {e}")
    print("Usage: python aie2.py npu2 <size> [operation]")
    print("Example: python aie2.py npu2 512 add")
    sys.exit(1)

module = elementwise_layer_bf16(dev, size, operation)
print(module)
