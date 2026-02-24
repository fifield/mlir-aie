#!/usr/bin/env python3
# vector_add_inline.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --tensor-size 4096 --work-dir ./vector_add_inline_build | FileCheck %s
# CHECK: PASS!

"""Single-file end-to-end example: PythoC inline kernel with IRON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.utils.compile import compile_mlir_module
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.placers import SequentialPlacer
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

# Import PythoC types and operations for inline kernel definition
from pythoc import ptr, i32
from pythoc.aie.operations import load_v, store_v, vector_add
from pythoc.aie.vector import aie_vector

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "vector_add_inline_build"


# Define PythoC kernel inline using @aie_kernel decorator
@aie_kernel
def add_kernel(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True], N: i32):
    """Vectorized element-wise addition kernel.

    Adds two int32 arrays element-wise using AIE2 vector operations.
    Processes 16 elements per iteration using 512-bit vectors.

    Args:
        a: Input array A (restrict pointer)
        b: Input array B (restrict pointer)
        c: Output array C (restrict pointer)
        N: Number of elements to process
    """
    event0()

    vec_size: i32 = 16  # AIE2 processes 16 x i32 per vector
    iterations: i32 = N // vec_size

    # Use pointer arithmetic (PythoC pattern)
    pA: ptr[i32] = a
    pB: ptr[i32] = b
    pC: ptr[i32] = c

    i: i32 = 0
    while i < iterations:
        # Load 16 elements from each input
        va: aie_vector[i32, 16] = load_v(pA, 16)
        vb: aie_vector[i32, 16] = load_v(pB, 16)

        # Perform vectorized addition
        vc: aie_vector[i32, 16] = vector_add(va, vb)

        # Store result
        store_v(pC, vc)

        # Advance pointers
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1

    event1()


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end PythoC inline kernel example with IRON",
    )
    parser.add_argument(
        "--device",
        choices=("npu", "npu1", "npu2"),
        default="npu2",
        help="Target device: npu/npu1 (AIE-ML) or npu2 (Strix)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Directory for generated MLIR, objects, and xclbin artifacts",
    )
    parser.add_argument(
        "--tensor-size",
        type=int,
        default=4096,
        help="Total vector length processed (must be divisible by 4 and 16)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit verbose compiler logging",
    )
    return parser.parse_args()


def pick_device(name: str):
    normalized = name.lower()
    if normalized == "npu2":
        return NPU2Col1(), "aie2p"
    return NPU1Col1(), "aie2"


def build_mlir_module(device, tensor_size: int):
    """Build IRON program with inline PythoC kernel."""
    tiles_per_factor = 4
    if tensor_size % tiles_per_factor:
        raise ValueError("tensor_size must be divisible by 4 for this example")
    if tensor_size % 16:
        raise ValueError("tensor_size must be divisible by 16 for vectorization")

    tile_size = tensor_size // tiles_per_factor

    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

    # Create PythocKernel from decorated function
    # The kernel is compiled automatically during initialization
    kernel = PythocKernel(
        add_kernel,  # Pass decorated function directly
        [tile_ty, tile_ty, tile_ty, np.int32],  # Type signature
    )

    # Define ObjectFifos for data movement
    of_a = ObjectFifo(tile_ty, name="in_a")
    of_b = ObjectFifo(tile_ty, name="in_b")
    of_c = ObjectFifo(tile_ty, name="out")

    # Define core function that uses the kernel
    def core_fn(of_a, of_b, of_c, kernel):
        """AIE core function: acquire buffers, call kernel, release."""
        for _ in range_(tiles_per_factor):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)

            # Call PythoC kernel
            kernel(elem_a, elem_b, elem_c, tile_size)

            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    # Create worker with kernel
    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), kernel])

    # Build IRON program with runtime sequence
    runtime = Runtime()
    with runtime.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_in, c_out):
        runtime.start(worker)
        runtime.fill(of_a.prod(), a_in)
        runtime.fill(of_b.prod(), b_in)
        runtime.drain(of_c.cons(), c_out, wait=True)

    program = Program(device, runtime)
    module = program.resolve_program(SequentialPlacer())
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


def save_module(module, mlir_path: Path):
    """Save MLIR module to file."""
    with open(mlir_path, "w", encoding="utf-8") as handle:
        print(module, file=handle)


def compile_design(
    module, insts_path: Path, xclbin_path: Path, work_dir: Path, verbose: bool
):
    """Compile MLIR module to xclbin using aiecc."""
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=verbose,
    )


def run_with_xrt(xclbin_path: Path, insts_path: Path, tensor_size: int, verbose: bool):
    """Execute design on NPU using XRT and verify results."""
    # Create NPU kernel handle
    npu_kernel = NPUKernel(
        str(xclbin_path),
        str(insts_path),
        kernel_name="MLIR_AIE",
    )
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # Prepare test data using iron.tensor
    a_data = np.arange(1, tensor_size + 1, dtype=np.int32)
    b_data = np.arange(1, tensor_size + 1, dtype=np.int32) * 2
    in_a = iron.tensor(a_data, dtype=np.int32)
    in_b = iron.tensor(b_data, dtype=np.int32)
    out_c = iron.zeros(tensor_size, dtype=np.int32)

    # Run on NPU
    DefaultNPURuntime.run(kernel_handle, [in_a, in_b, out_c])

    # Verify results
    output_vec = out_c.numpy()
    expected = a_data + b_data
    np.testing.assert_array_equal(output_vec, expected)

    # Return a COPY of the data, since output_vec is a view into XRT buffer
    # that will be freed when out_c goes out of scope
    return np.array(output_vec)


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device, target_arch = pick_device(args.device)

    try:
        print(f"[1/3] Building IRON program with inline PythoC kernel ({target_arch})")
        module = build_mlir_module(device, args.tensor_size)
        mlir_path = work_dir / "kernel.mlir"
        save_module(module, mlir_path)
        print(f"      -> {mlir_path}")
        
        print("[2/3] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_design(module, insts_path, xclbin_path, work_dir, args.verbose)
        print(f"      -> {xclbin_path}\n      -> {insts_path}")

        print("[3/3] Running with pyxrt and validating results")
        output_vec = run_with_xrt(
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            tensor_size=args.tensor_size,
            verbose=args.verbose,
        )
        preview = np.asarray(output_vec[:8])
        print(f"      First elements: {preview}")
        print("PASS!")
        return 0
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
