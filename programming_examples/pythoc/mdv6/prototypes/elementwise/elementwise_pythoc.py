#!/usr/bin/env python3
# elementwise_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --size 512 --operation add --work-dir ./elementwise_pythoc_add_build | FileCheck %s
# RUN: %python %s --device npu2 --size 512 --operation mul --work-dir ./elementwise_pythoc_mul_build | FileCheck %s
# RUN: %python %s --device npu2 --size 512 --operation max --work-dir ./elementwise_pythoc_max_build | FileCheck %s
# CHECK: PASS!

"""MDV6 element-wise (add/max/mul) layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/elementwise/{aie2.py,elementwise_bf16.cc}
that replaces the external C++ kernel with inline PythoC kernels.

Operations (bfloat16):
    add: out[i] = a[i] + b[i]      -- vector_add
    mul: out[i] = a[i] * b[i]      -- vector_mul
    max: out[i] = max(a[i], b[i])  -- vmax_ltbf16 (bf16 doesn't work with vector_max)
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
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import ptr, i32, bf16
from pythoc.aie import (
    aie_vector,
    load_v,
    store_v,
    vector_add,
    vector_mul,
    vmax_ltbf16,
)
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "elementwise_pythoc_build"


# ── PythoC kernels (bf16, 32-wide vectors) ─────────────────────────────


@aie_kernel
def add_bf16_kernel(
    a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[bf16, True], n: i32
):
    """C[i] = A[i] + B[i] for bfloat16, processed 32 elements per iteration."""
    event0()
    vec_size: i32 = 32
    iters: i32 = n // vec_size

    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pC: ptr[bf16] = c

    i: i32 = 0
    while i < iters:
        va: aie_vector[bf16, 32] = load_v(pA, 32)
        vb: aie_vector[bf16, 32] = load_v(pB, 32)
        vc: aie_vector[bf16, 32] = vector_add(va, vb)
        store_v(pC, vc)
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1
    event1()


@aie_kernel
def mul_bf16_kernel(
    a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[bf16, True], n: i32
):
    """C[i] = A[i] * B[i] for bfloat16."""
    event0()
    vec_size: i32 = 32
    iters: i32 = n // vec_size

    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pC: ptr[bf16] = c

    i: i32 = 0
    while i < iters:
        va: aie_vector[bf16, 32] = load_v(pA, 32)
        vb: aie_vector[bf16, 32] = load_v(pB, 32)
        vc: aie_vector[bf16, 32] = vector_mul(va, vb)
        store_v(pC, vc)
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1
    event1()


@aie_kernel
def max_bf16_kernel(
    a: ptr[bf16, True], b: ptr[bf16, True], c: ptr[bf16, True], n: i32
):
    """C[i] = max(A[i], B[i]) for bfloat16, via vmax_ltbf16 intrinsic.

    vector_max uses icmp and only supports signed integers, so bf16 max must
    go through the AIE vmax_ltbf16 intrinsic (which returns (max, lt_mask)).
    """
    event0()
    vec_size: i32 = 32
    iters: i32 = n // vec_size

    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pC: ptr[bf16] = c

    i: i32 = 0
    while i < iters:
        va: aie_vector[bf16, 32] = load_v(pA, 32)
        vb: aie_vector[bf16, 32] = load_v(pB, 32)
        vc, _mask = vmax_ltbf16(va, vb)
        store_v(pC, vc)
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1
    event1()


KERNELS = {
    "add": add_bf16_kernel,
    "mul": mul_bf16_kernel,
    "max": max_bf16_kernel,
}


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 element-wise layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument(
        "--size", type=int, default=512,
        help="Number of bf16 elements (must be divisible by 32)",
    )
    parser.add_argument(
        "--operation", "-op", choices=("add", "mul", "max"), default="add",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── MLIR / IRON construction ──────────────────────────────────────────


def build_mlir_module(device, size: int, operation: str):
    if size % 32:
        raise ValueError("size must be divisible by 32 for bf16 vectorization")

    # bf16 carried as uint16 in IRON/numpy
    tensor_ty = np.ndarray[(size,), np.dtype[np.uint16]]

    # vmax_ltbf16 isn't in @aie_kernel's default global set; inject it for max.
    extra = {"vmax_ltbf16": vmax_ltbf16} if operation == "max" else None
    kernel = PythocKernel(
        KERNELS[operation],
        [tensor_ty, tensor_ty, tensor_ty, np.int32],
        extra_globals=extra,
    )

    of_a = ObjectFifo(tensor_ty, depth=1, name="input_a")
    of_b = ObjectFifo(tensor_ty, depth=1, name="input_b")
    of_c = ObjectFifo(tensor_ty, depth=1, name="output")

    def core_fn(of_a, of_b, of_c, kernel):
        elem_a = of_a.acquire(1)
        elem_b = of_b.acquire(1)
        elem_c = of_c.acquire(1)
        kernel(elem_a, elem_b, elem_c, size)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), kernel])

    def sequence(A, B, C, of_a_prod, of_b_prod, of_c_cons):
        of_a_prod.fill(A)
        of_b_prod.fill(B)
        of_c_cons.drain(C, wait=True)

    runtime = Runtime(
        sequence,
        [tensor_ty, tensor_ty, tensor_ty, of_a.prod(), of_b.prod(), of_c.cons()],
    )

    program = Program(device, runtime, workers=[worker])
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── Compile & run ─────────────────────────────────────────────────────


def run_with_xrt(xclbin_path: Path, insts_path: Path, size: int, operation: str):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    rng = np.random.default_rng(42)
    a_f32 = rng.standard_normal(size).astype(np.float32)
    b_f32 = rng.standard_normal(size).astype(np.float32)
    a_bf16 = a_f32.astype(bfloat16)
    b_bf16 = b_f32.astype(bfloat16)

    in_a = iron.tensor(a_bf16.view(np.uint16), dtype=np.uint16)
    in_b = iron.tensor(b_bf16.view(np.uint16), dtype=np.uint16)
    out_c = iron.zeros(size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_a, in_b, out_c])

    out_u16 = out_c.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    a_ref = a_bf16.astype(np.float32)
    b_ref = b_bf16.astype(np.float32)
    if operation == "add":
        expected_f32 = (a_ref + b_ref).astype(bfloat16).astype(np.float32)
    elif operation == "mul":
        expected_f32 = (a_ref * b_ref).astype(bfloat16).astype(np.float32)
    elif operation == "max":
        expected_f32 = np.maximum(a_ref, b_ref).astype(bfloat16).astype(np.float32)
    else:
        raise ValueError(operation)

    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()

    try:
        print(f"[1/3] Building IRON program (operation={args.operation}, size={args.size})")
        module = build_mlir_module(device, args.size, args.operation)
        mlir_path = work_dir / "kernel.mlir"
        with open(mlir_path, "w", encoding="utf-8") as fh:
            print(module, file=fh)
        print(f"      -> {mlir_path}")

        print("[2/3] Compiling design with aiecc")
        insts_path = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_mlir_module(
            mlir_module=module,
            insts_path=str(insts_path),
            xclbin_path=str(xclbin_path),
            work_dir=str(work_dir),
            verbose=args.verbose,
        )
        print(f"      -> {xclbin_path}\n      -> {insts_path}")

        print("[3/3] Running with pyxrt and validating results")
        actual, expected = run_with_xrt(xclbin_path, insts_path, args.size, args.operation)
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        if np.allclose(actual, expected, rtol=1e-2, atol=1e-2):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=1e-2, atol=1e-2)
        print(f"FAILED: {int(mism.sum())}/{len(actual)} mismatches")
        for i in np.where(mism)[0][:5]:
            print(f"        [{i}] got {actual[i]}, expected {expected[i]}")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
