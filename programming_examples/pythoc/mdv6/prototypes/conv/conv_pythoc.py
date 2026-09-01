#!/usr/bin/env python3
# conv_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 4 --out-channels 4 --kernel-size 3 --work-dir ./conv_pythoc_k3_build | FileCheck %s
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 4 --out-channels 4 --kernel-size 1 --work-dir ./conv_pythoc_k1_build | FileCheck %s
# CHECK: PASS!

"""MDV6 Conv2D (3x3 / 1x1, bf16) as a PythoC + IRON example.

Port of programming_examples/pythoc/mdv6/conv/{aie2.py,conv_bf16.cc} that
replaces the external C++ kernel with an inline PythoC kernel implementing
the canonical scalar conv2d (zero-padded, stride 1).

Data layout (matches conv_bf16.cc):
    input   : (H, W, C_in)              row-major bf16
    weights : (C_out, C_in, K, K)       row-major bf16
    output  : (H_out, W_out, C_out)     row-major bf16

Two variants are implemented:
    --kernel-size 3 : conv3x3 (uses stride, padding)
    --kernel-size 1 : conv1x1 (pointwise; stride/padding ignored)

Both use f32 accumulation with bf16 multiplicands, matching the C++ kernel.

Variants NOT ported (left as future work):
    - conv_bf16_vec.cc      (vectorised scalar baseline)
    - conv3x3_vec_packed.cc (mmul<4,8,8> packed-weight conv+BN+SiLU)
    - aie2_tiled*.py, aie2_multicore*.py, aie2_fused.py (multi-tile / fused
      pipelines built on top of the same kernel)
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

from pythoc import ptr, i32, bf16, f32
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "conv_pythoc_build"


# ── PythoC kernels (scalar bf16, f32 accumulation) ──────────────────────


@aie_kernel
def conv3x3_bf16_kernel(
    input: ptr[bf16, True],
    weights: ptr[bf16, True],
    output: ptr[bf16, True],
    input_height: i32,
    input_width: i32,
    input_channels: i32,
    output_channels: i32,
    stride: i32,
    padding: i32,
):
    """Scalar conv2d 3x3 (bf16 in/out, f32 accumulation).

    Mirrors conv3x3_bf16_scalar in conv_bf16.cc:
        output[oh, ow, oc] = sum over (ic, kh, kw) of
                                input[oh*s+kh-p, ow*s+kw-p, ic]
                              * weights[oc, ic, kh, kw]
    Zero padding is implemented with an in-bounds check.
    """
    event0()

    k: i32 = 3
    output_height: i32 = (input_height + 2 * padding - k) // stride + 1
    output_width: i32 = (input_width + 2 * padding - k) // stride + 1

    oc: i32 = 0
    while oc < output_channels:
        oh: i32 = 0
        while oh < output_height:
            ow: i32 = 0
            while ow < output_width:
                acc: f32 = 0.0

                ic: i32 = 0
                while ic < input_channels:
                    kh: i32 = 0
                    while kh < k:
                        kw: i32 = 0
                        while kw < k:
                            ih: i32 = oh * stride + kh - padding
                            iw: i32 = ow * stride + kw - padding

                            if ih >= 0:
                                if ih < input_height:
                                    if iw >= 0:
                                        if iw < input_width:
                                            input_idx: i32 = (ih * input_width + iw) * input_channels + ic
                                            weight_idx: i32 = ((oc * input_channels + ic) * k + kh) * k + kw
                                            in_val: bf16 = input[input_idx]
                                            w_val: bf16 = weights[weight_idx]
                                            acc = acc + (f32(in_val) * f32(w_val))
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1

                output_idx: i32 = (oh * output_width + ow) * output_channels + oc
                output[output_idx] = bf16(acc)

                ow = ow + 1
            oh = oh + 1
        oc = oc + 1

    event1()


@aie_kernel
def conv1x1_bf16_kernel(
    input: ptr[bf16, True],
    weights: ptr[bf16, True],
    output: ptr[bf16, True],
    input_height: i32,
    input_width: i32,
    input_channels: i32,
    output_channels: i32,
):
    """Scalar conv2d 1x1 (bf16 in/out, f32 accumulation).

    Mirrors conv1x1_bf16_scalar in conv_bf16.cc:
        output[h, w, oc] = sum over ic of input[h, w, ic] * weights[oc, ic]
    """
    event0()

    h: i32 = 0
    while h < input_height:
        w: i32 = 0
        while w < input_width:
            oc: i32 = 0
            while oc < output_channels:
                acc: f32 = 0.0

                ic: i32 = 0
                while ic < input_channels:
                    input_idx: i32 = (h * input_width + w) * input_channels + ic
                    weight_idx: i32 = oc * input_channels + ic
                    in_val: bf16 = input[input_idx]
                    w_val: bf16 = weights[weight_idx]
                    acc = acc + (f32(in_val) * f32(w_val))
                    ic = ic + 1

                output_idx: i32 = (h * input_width + w) * output_channels + oc
                output[output_idx] = bf16(acc)

                oc = oc + 1
            w = w + 1
        h = h + 1

    event1()


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 Conv2D layer (PythoC + IRON, bf16, scalar)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", type=int, default=8, help="Input height H")
    parser.add_argument("--width", type=int, default=8, help="Input width W")
    parser.add_argument("--in-channels", type=int, default=4, help="C_in")
    parser.add_argument("--out-channels", type=int, default=4, help="C_out")
    parser.add_argument(
        "--kernel-size", type=int, choices=(1, 3), default=3,
        help="Spatial kernel size (1 or 3)",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--padding", type=int, default=1,
        help="Zero-padding (ignored for kernel-size=1)",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── MLIR / IRON construction ──────────────────────────────────────────


def build_mlir_module(
    device,
    input_height: int,
    input_width: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    input_size = input_height * input_width * input_channels
    weight_size = output_channels * input_channels * kernel_size * kernel_size
    output_size = output_height * output_width * output_channels

    # bf16 carried as uint16 in IRON/numpy
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    if kernel_size == 3:
        kernel = PythocKernel(
            conv3x3_bf16_kernel,
            [
                input_ty,
                weight_ty,
                output_ty,
                np.int32,  # input_height
                np.int32,  # input_width
                np.int32,  # input_channels
                np.int32,  # output_channels
                np.int32,  # stride
                np.int32,  # padding
            ],
        )
    else:  # kernel_size == 1
        kernel = PythocKernel(
            conv1x1_bf16_kernel,
            [
                input_ty,
                weight_ty,
                output_ty,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    def core_fn(of_in, of_wts, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        if kernel_size == 3:
            kernel(
                elem_in,
                elem_wts,
                elem_out,
                input_height,
                input_width,
                input_channels,
                output_channels,
                stride,
                padding,
            )
        else:
            kernel(
                elem_in,
                elem_wts,
                elem_out,
                input_height,
                input_width,
                input_channels,
                output_channels,
            )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_input.cons(), of_weights.cons(), of_output.prod(), kernel],
    )

    def sequence(I, W, O, of_input_prod, of_weights_prod, of_output_cons):
        of_input_prod.fill(I)
        of_weights_prod.fill(W)
        of_output_cons.drain(O, wait=True)

    runtime = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_input.prod(), of_weights.prod(), of_output.cons()],
    )

    program = Program(device, runtime, workers=[worker])
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── Compile & run ─────────────────────────────────────────────────────


def numpy_conv_reference(
    input_hwc_bf16: np.ndarray,
    weights_oikk_bf16: np.ndarray,
    output_height: int,
    output_width: int,
    output_channels: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    """Reference conv2d on the CPU. Uses f32 arithmetic with bf16 round at end."""
    in_f = input_hwc_bf16.astype(np.float32)
    w_f = weights_oikk_bf16.astype(np.float32)

    out = np.zeros((output_height, output_width, output_channels), dtype=np.float32)
    for oc in range(output_channels):
        for oh in range(output_height):
            for ow in range(output_width):
                s = 0.0
                for ic in range(input_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = oh * stride + kh - padding
                            iw = ow * stride + kw - padding
                            if 0 <= ih < input_height and 0 <= iw < input_width:
                                s += in_f[ih, iw, ic] * w_f[oc, ic, kh, kw]
                out[oh, ow, oc] = s

    return out.astype(bfloat16).astype(np.float32)


def run_with_xrt(
    xclbin_path: Path,
    insts_path: Path,
    input_height: int,
    input_width: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    input_size = input_height * input_width * input_channels
    weight_size = output_channels * input_channels * kernel_size * kernel_size
    output_size = output_height * output_width * output_channels

    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    rng = np.random.default_rng(42)
    # Smaller-magnitude random values keep the running f32 sum within bf16
    # round-to-nearest tolerance for small channel counts.
    input_f32 = (rng.standard_normal(input_size) * 0.5).astype(np.float32)
    weights_f32 = (rng.standard_normal(weight_size) * 0.5).astype(np.float32)
    input_bf16 = input_f32.astype(bfloat16)
    weights_bf16 = weights_f32.astype(bfloat16)

    in_buf = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    wts_buf = iron.tensor(weights_bf16.view(np.uint16), dtype=np.uint16)
    out_buf = iron.zeros(output_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_buf, wts_buf, out_buf])

    out_u16 = out_buf.numpy()[:output_size]
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)
    actual = actual_f32.reshape(output_height, output_width, output_channels)

    input_hwc = input_bf16.reshape(input_height, input_width, input_channels)
    weights_oikk = weights_bf16.reshape(
        output_channels, input_channels, kernel_size, kernel_size
    )
    expected = numpy_conv_reference(
        input_hwc,
        weights_oikk,
        output_height,
        output_width,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
    )

    return actual, expected


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()

    # Effective stride/padding (kernel-size 1 always uses stride=1, pad=0)
    if args.kernel_size == 1:
        stride, padding = 1, 0
    else:
        stride, padding = args.stride, args.padding

    try:
        print(
            f"[1/3] Building IRON program (k={args.kernel_size}, "
            f"H={args.height}, W={args.width}, Cin={args.in_channels}, "
            f"Cout={args.out_channels}, s={stride}, p={padding})"
        )
        module = build_mlir_module(
            device,
            args.height,
            args.width,
            args.in_channels,
            args.out_channels,
            args.kernel_size,
            stride,
            padding,
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
            xclbin_path,
            insts_path,
            args.height,
            args.width,
            args.in_channels,
            args.out_channels,
            args.kernel_size,
            stride,
            padding,
        )
        print(f"      actual[0,0,:]   = {actual[0, 0, :]}")
        print(f"      expected[0,0,:] = {expected[0, 0, :]}")

        # bf16 conv has accumulation rounding; allow ~5% relative tolerance.
        if np.allclose(actual, expected, rtol=5e-2, atol=5e-2):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=5e-2, atol=5e-2)
        n_mism = int(mism.sum())
        print(f"FAILED: {n_mism}/{actual.size} mismatches")
        if n_mism:
            for idx in list(zip(*np.where(mism)))[:5]:
                print(f"        {idx}: got {actual[idx]}, expected {expected[idx]}")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
