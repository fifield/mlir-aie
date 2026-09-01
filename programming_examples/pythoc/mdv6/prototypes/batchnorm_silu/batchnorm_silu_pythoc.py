#!/usr/bin/env python3
# batchnorm_silu_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --channels 16 --work-dir ./batchnorm_silu_pythoc_build | FileCheck %s
# RUN: %python %s --device npu2 --height 8 --width 8 --channels 16 --no-silu --work-dir ./batchnorm_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 BatchNorm + SiLU layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/batchnorm_silu/{aie2.py,batchnorm_silu_bf16.cc}
that replaces the external C++ kernel with an inline PythoC kernel.

Math (bfloat16):
    BN:   y = w * x + b           (per-channel affine)
    SiLU: y = y * sigmoid(y)
          sigmoid(z) ~= 0.5 * (1 + tanh(0.5 * z))   (LUT-based tanh on AIE)

Input layout is HWC so channels are the innermost (and contiguous) dimension,
allowing a 16-wide bf16 vectorisation along C.
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
    broadcast,
    getTanhBf16,
    load_v,
    store_v,
    vector_add,
    vector_mul,
)
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "batchnorm_silu_pythoc_build"

VEC = 16  # bf16 vector width used by the inner C-axis loop


# ── PythoC kernels (bf16, 16-wide vectors along C) ─────────────────────


@aie_kernel
def batchnorm_silu_bf16_kernel(
    input: ptr[bf16, True],
    bn_params: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    channels: i32,
):
    """Fused BatchNorm + SiLU over an HWC tensor.

    bn_params is laid out as [weight (C), bias (C)] in bfloat16.
    Inner loop processes 16 channels at a time, so `channels` must be a
    multiple of 16.
    """
    event0()

    vec: i32 = 16
    spatial: i32 = height * width

    # SiLU constants broadcast once at the top.
    half: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(0.5))
    one: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.0))

    # bn_weight = bn_params,  bn_bias = bn_params + C
    bn_weight: ptr[bf16] = bn_params
    bn_bias: ptr[bf16] = bn_params + channels

    p_in: ptr[bf16] = input
    p_out: ptr[bf16] = output

    hw: i32 = 0
    while hw < spatial:
        p_w: ptr[bf16] = bn_weight
        p_b: ptr[bf16] = bn_bias

        c: i32 = 0
        while c < channels:
            x: aie_vector[bf16, 16] = load_v(p_in, 16)
            w: aie_vector[bf16, 16] = load_v(p_w, 16)
            b: aie_vector[bf16, 16] = load_v(p_b, 16)

            # BatchNorm: y = w * x + b
            wx: aie_vector[bf16, 16] = vector_mul(w, x)
            y: aie_vector[bf16, 16] = vector_add(wx, b)

            # SiLU: y * 0.5 * (1 + tanh(0.5 * y))
            y_half: aie_vector[bf16, 16] = vector_mul(y, half)
            t: aie_vector[bf16, 16] = getTanhBf16(y_half)
            one_plus: aie_vector[bf16, 16] = vector_add(one, t)
            sigmoid: aie_vector[bf16, 16] = vector_mul(half, one_plus)
            silu: aie_vector[bf16, 16] = vector_mul(y, sigmoid)

            store_v(p_out, silu)

            p_in = p_in + vec
            p_out = p_out + vec
            p_w = p_w + vec
            p_b = p_b + vec
            c = c + vec

        hw = hw + 1
    event1()


@aie_kernel
def batchnorm_bf16_kernel(
    input: ptr[bf16, True],
    bn_params: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    channels: i32,
):
    """BatchNorm only (no SiLU). Same layout as batchnorm_silu_bf16_kernel."""
    event0()

    vec: i32 = 16
    spatial: i32 = height * width

    bn_weight: ptr[bf16] = bn_params
    bn_bias: ptr[bf16] = bn_params + channels

    p_in: ptr[bf16] = input
    p_out: ptr[bf16] = output

    hw: i32 = 0
    while hw < spatial:
        p_w: ptr[bf16] = bn_weight
        p_b: ptr[bf16] = bn_bias

        c: i32 = 0
        while c < channels:
            x: aie_vector[bf16, 16] = load_v(p_in, 16)
            w: aie_vector[bf16, 16] = load_v(p_w, 16)
            b: aie_vector[bf16, 16] = load_v(p_b, 16)

            wx: aie_vector[bf16, 16] = vector_mul(w, x)
            y: aie_vector[bf16, 16] = vector_add(wx, b)
            store_v(p_out, y)

            p_in = p_in + vec
            p_out = p_out + vec
            p_w = p_w + vec
            p_b = p_b + vec
            c = c + vec

        hw = hw + 1
    event1()


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 BatchNorm + SiLU layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument(
        "--channels", "-c", type=int, default=16,
        help="Number of channels (must be a multiple of 16)",
    )
    parser.add_argument(
        "--no-silu", action="store_true",
        help="BatchNorm only (skip SiLU activation)",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── MLIR / IRON construction ──────────────────────────────────────────


def build_mlir_module(device, height: int, width: int, channels: int, use_silu: bool):
    if channels % VEC:
        raise ValueError(f"channels must be a multiple of {VEC} for bf16 vectorisation")

    input_size = height * width * channels
    bn_params_size = 2 * channels  # [weight (C), bias (C)]
    output_size = input_size

    # bf16 carried as uint16 in IRON / numpy
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    bn_params_ty = np.ndarray[(bn_params_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    kernel_fn = batchnorm_silu_bf16_kernel if use_silu else batchnorm_bf16_kernel
    # getTanhBf16 is not in @aie_kernel's default global set; inject for SiLU.
    extra = {"getTanhBf16": getTanhBf16} if use_silu else None
    kernel = PythocKernel(
        kernel_fn,
        [input_ty, bn_params_ty, output_ty, np.int32, np.int32, np.int32],
        extra_globals=extra,
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_params = ObjectFifo(bn_params_ty, depth=1, name="bn_params_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    def core_fn(of_in, of_params, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_params = of_params.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_params, elem_out, height, width, channels)
        of_in.release(1)
        of_params.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_input.cons(), of_params.cons(), of_output.prod(), kernel],
    )

    def sequence(I, P, O, of_input_prod, of_params_prod, of_output_cons):
        of_input_prod.fill(I)
        of_params_prod.fill(P)
        of_output_cons.drain(O, wait=True)

    runtime = Runtime(
        sequence,
        [input_ty, bn_params_ty, output_ty, of_input.prod(), of_params.prod(), of_output.cons()],
    )

    program = Program(device, runtime, workers=[worker])
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── Compile & run ─────────────────────────────────────────────────────


def numpy_reference(input_bf16, bn_w_bf16, bn_b_bf16, height, width, channels, use_silu):
    """bf16 reference matching the on-device PythoC kernel.

    Computes everything in fp32 then rounds the result through bf16 once at the
    end to model the kernel's per-vector bf16 rounding. tanh is the math tanh
    (the device uses a LUT approximation — that's the dominant source of error
    we tolerate with rtol/atol = 1e-2 / 5e-2).
    """
    x = input_bf16.reshape(height, width, channels).astype(np.float32)
    w = bn_w_bf16.astype(np.float32)
    b = bn_b_bf16.astype(np.float32)

    y = x * w + b  # broadcast over (H, W)
    y = y.astype(bfloat16).astype(np.float32)  # round BN output to bf16

    if use_silu:
        sigmoid = 0.5 * (1.0 + np.tanh(0.5 * y))
        y = y * sigmoid
        y = y.astype(bfloat16).astype(np.float32)

    return y.reshape(-1)


def run_with_xrt(xclbin_path: Path, insts_path: Path,
                 height: int, width: int, channels: int, use_silu: bool):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    input_size = height * width * channels

    rng = np.random.default_rng(42)
    input_f32 = rng.standard_normal(input_size).astype(np.float32)
    weight_f32 = rng.standard_normal(channels).astype(np.float32)
    bias_f32 = rng.standard_normal(channels).astype(np.float32)

    input_bf16 = input_f32.astype(bfloat16)
    weight_bf16 = weight_f32.astype(bfloat16)
    bias_bf16 = bias_f32.astype(bfloat16)

    bn_params_bf16 = np.concatenate([weight_bf16, bias_bf16])

    in_input = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    in_params = iron.tensor(bn_params_bf16.view(np.uint16), dtype=np.uint16)
    out_tensor = iron.zeros(input_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_input, in_params, out_tensor])

    out_u16 = out_tensor.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = numpy_reference(
        input_bf16, weight_bf16, bias_bf16, height, width, channels, use_silu
    )
    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()
    use_silu = not args.no_silu

    try:
        print(
            f"[1/3] Building IRON program "
            f"(H={args.height}, W={args.width}, C={args.channels}, silu={use_silu})"
        )
        module = build_mlir_module(
            device, args.height, args.width, args.channels, use_silu
        )
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
        actual, expected = run_with_xrt(
            xclbin_path, insts_path,
            args.height, args.width, args.channels, use_silu,
        )
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        # SiLU uses a LUT tanh approximation; allow a moderate tolerance.
        rtol = 5e-2 if use_silu else 1e-2
        atol = 5e-2 if use_silu else 1e-2
        if np.allclose(actual, expected, rtol=rtol, atol=atol):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=rtol, atol=atol)
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
