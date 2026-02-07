#!/usr/bin/env python3
# vector_mul_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""End-to-end example: External PythoC kernel with IRON.

This example demonstrates how to:
1. Compile an external PythoC kernel to LLVM IR
2. Create an IRON Kernel from the compiled IR
3. Build and compile a complete MLIR design
4. Run on NPU hardware using pyxrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.utils.compile import compile_mlir_module
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.placers import SequentialPlacer
from aie.iron.pythoc import compile_pythoc_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "vector_mul_pythoc_build"


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end external PythoC kernel example with IRON",
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
        default=1024,
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


def build_mlir_module(device, target_arch: str, tensor_size: int, verbose: bool):
    """Build IRON program with external PythoC kernel."""
    tiles_per_factor = 4
    if tensor_size % tiles_per_factor:
        raise ValueError("tensor_size must be divisible by 4 for this example")
    if tensor_size % 16:
        raise ValueError("tensor_size must be divisible by 16 for vectorization")

    tile_size = tensor_size // tiles_per_factor

    # Type definitions (bf16 = bfloat16, represented as uint16 in NumPy)
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.uint16]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.uint16]]
    
    # Step 1: Compile external PythoC kernel to LLVM IR
    if verbose:
        print(f"  Compiling PythoC kernel (eltwise_mul_vectorized, {target_arch})...")
    pythoc_kernel_path = Path("/work/npu-dev/PythoC/pythoc_kernels/mul.py")
    
    ll_file = compile_pythoc_kernel(
        str(pythoc_kernel_path),
        function_name="eltwise_mul_vectorized",
        target_arch=target_arch,
        verbose=verbose
    )
    
    if verbose:
        print(f"  LLVM IR: {ll_file}")
    
    # Step 2: Create IRON Kernel from compiled IR
    mul_kernel = PythocKernel(
        "eltwise_mul_vectorized",
        str(ll_file),
        [tile_ty, tile_ty, tile_ty, np.int32]  # Types: input_a, input_b, output, size
    )

    # Define ObjectFifos for data movement
    of_a = ObjectFifo(tile_ty, name="in_a")
    of_b = ObjectFifo(tile_ty, name="in_b")
    of_c = ObjectFifo(tile_ty, name="out")

    # Define core function that uses the kernel
    def core_fn(of_a, of_b, of_c, mul_kernel):
        """AIE core function: acquire buffers, call kernel, release."""
        for _ in range_(tiles_per_factor):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)

            # Call PythoC kernel
            mul_kernel(elem_a, elem_b, elem_c, tile_size)

            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    # Create worker with kernel
    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), mul_kernel])

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

    # Prepare test data using bf16 (represented as uint16 in XRT buffers).
    # Convert float32 → bfloat16 → view as uint16 for a 1:1 element mapping.
    a_data = np.arange(1, tensor_size + 1, dtype=np.float32)
    b_data = np.full(tensor_size, 2.0, dtype=np.float32)

    a_bf16 = a_data.astype(bfloat16)
    b_bf16 = b_data.astype(bfloat16)

    in_a = iron.tensor(a_bf16.view(np.uint16), dtype=np.uint16)
    in_b = iron.tensor(b_bf16.view(np.uint16), dtype=np.uint16)
    out_c = iron.zeros(tensor_size, dtype=np.uint16)

    # Run on NPU
    DefaultNPURuntime.run(kernel_handle, [in_a, in_b, out_c])

    # Convert results back from bf16 for verification.
    # output as uint16 → view as bfloat16 → cast to float32 for comparison
    output_u16 = out_c.numpy()
    output_bf16 = np.array(output_u16, dtype=np.uint16).view(bfloat16)
    output_f32 = output_bf16.astype(np.float32)

    # Compute expected result in float32, truncate to bf16, back to float32
    expected_bf16 = (a_data * b_data).astype(bfloat16)
    expected_f32 = expected_bf16.astype(np.float32)

    # Return float32 arrays for readable comparison
    return output_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device, target_arch = pick_device(args.device)

    try:
        print(f"[1/3] Building IRON program with external PythoC kernel ({target_arch})")
        module = build_mlir_module(device, target_arch, args.tensor_size, args.verbose)
        mlir_path = work_dir / "kernel.mlir"
        save_module(module, mlir_path)
        print(f"      -> {mlir_path}")
        
        print("[2/3] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_design(module, insts_path, xclbin_path, work_dir, args.verbose)
        print(f"      -> {xclbin_path}\n      -> {insts_path}")

        print("[3/3] Running with pyxrt and validating results")
        output_vec, expected_vec = run_with_xrt(
            xclbin_path=xclbin_path,
            insts_path=insts_path,
            tensor_size=args.tensor_size,
            verbose=args.verbose,
        )
        
        # Show first few elements (float32 values)
        preview_out = output_vec[:8]
        preview_exp = expected_vec[:8]
        print(f"      Output (f32):   {preview_out}")
        print(f"      Expected (f32): {preview_exp}")
        
        # Use approximate comparison (bf16 has limited precision)
        if np.allclose(output_vec, expected_vec, rtol=1e-2, atol=1e-2):
            print("PASS!")
            return 0
        else:
            mismatches = ~np.isclose(output_vec, expected_vec, rtol=1e-2, atol=1e-2)
            n_mismatch = np.sum(mismatches)
            print(f"FAILED: {n_mismatch}/{len(output_vec)} mismatches")
            # Show first few mismatches
            idxs = np.where(mismatches)[0][:5]
            for i in idxs:
                print(f"        [{i}] got {output_vec[i]}, expected {expected_vec[i]}")
            return 1
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
