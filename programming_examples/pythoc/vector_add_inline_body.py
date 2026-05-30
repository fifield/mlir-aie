#!/usr/bin/env python3
# vector_add_inline_body.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --tensor-size 4096 --work-dir ./vector_add_inline_body_build | FileCheck %s
# CHECK: PASS!

"""PythoC kernel body written *literally inside* the aie.core via @pythoc_inline.

Unlike vector_add_inline.py (which defines a separate @aie_kernel and calls it),
here the compute is written as a nested, type-annotated PythoC function right in
the traced core body. @pythoc_inline lifts it into a synthetic kernel, compiles
it through the inline path (alwaysinline .ll), and aiecc llvm-links + inlines it
into the core -- so there is no func.call boundary and no separate object file.
The tile size is baked in as a compile-time constant.
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
from aie.iron.pythoc import pythoc_inline
from aie.utils import DefaultNPURuntime, NPUKernel

# PythoC types/ops referenced by the inline kernel body. They are module globals,
# so @pythoc_inline makes them available to the lifted kernel automatically.
from pythoc import ptr, i32
from pythoc.aie.operations import load_v, store_v, vector_add
from pythoc.aie.vector import aie_vector
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "vector_add_inline_body_build"


def parse_args():
    parser = argparse.ArgumentParser(
        description="PythoC inline-body kernel example with IRON",
    )
    parser.add_argument("--device", choices=("npu", "npu1", "npu2"), default="npu2")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--tensor-size", type=int, default=4096)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def pick_device(name: str):
    return (NPU2Col1(), "aie2p") if name.lower() == "npu2" else (NPU1Col1(), "aie2")


def build_mlir_module(device, tensor_size: int):
    if tensor_size % 4 or tensor_size % 16:
        raise ValueError("tensor_size must be divisible by 4 and 16")
    tile_size = tensor_size // 4

    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

    of_a = ObjectFifo(tile_ty, name="in_a")
    of_b = ObjectFifo(tile_ty, name="in_b")
    of_c = ObjectFifo(tile_ty, name="out")

    def core_fn(of_a, of_b, of_c):
        for _ in range_(4):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)

            # The kernel body lives right here in the core. `tile_size` is baked
            # in as a compile-time constant; the buffers are passed positionally.
            @pythoc_inline(elem_a, elem_b, elem_c, N=tile_size)
            def _add(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True]):
                event0()
                i: i32 = 0
                while i < N:
                    va: aie_vector[i32, 16] = load_v(a + i, 16)
                    vb: aie_vector[i32, 16] = load_v(b + i, 16)
                    vc: aie_vector[i32, 16] = vector_add(va, vb)
                    store_v(c + i, vc)
                    i = i + 16
                event1()

            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod()])

    runtime = Runtime()
    with runtime.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_in, c_out):
        runtime.start(worker)
        runtime.fill(of_a.prod(), a_in)
        runtime.fill(of_b.prod(), b_in)
        runtime.drain(of_c.cons(), c_out, wait=True)

    program = Program(device, runtime)
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


def run_with_xrt(xclbin_path: Path, insts_path: Path, tensor_size: int):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    a_data = np.arange(1, tensor_size + 1, dtype=np.int32)
    b_data = np.arange(1, tensor_size + 1, dtype=np.int32) * 2
    in_a = iron.tensor(a_data, dtype=np.int32)
    in_b = iron.tensor(b_data, dtype=np.int32)
    out_c = iron.zeros(tensor_size, dtype=np.int32)

    DefaultNPURuntime.run(handle, [in_a, in_b, out_c])

    output_vec = out_c.numpy()
    np.testing.assert_array_equal(output_vec, a_data + b_data)
    return np.array(output_vec)


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    device, target_arch = pick_device(args.device)

    try:
        print(f"[1/3] Building IRON program with inline PythoC body ({target_arch})")
        module = build_mlir_module(device, args.tensor_size)
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
        output_vec = run_with_xrt(xclbin_path, insts_path, args.tensor_size)
        print(f"      First elements: {np.asarray(output_vec[:8])}")
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
