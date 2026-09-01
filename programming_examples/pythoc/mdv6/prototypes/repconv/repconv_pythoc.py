#!/usr/bin/env python3
# repconv_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 8 --out-channels 8 --work-dir ./repconv_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 RepConv layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/repconv/{aie2.py,repconv_bf16.cc} that
replaces the external C++ kernel with an inline PythoC kernel.

RepConv architecture (bfloat16):
    Input -> Conv3x3 + BN (no activation) -> temp1
    Input -> Conv1x1 + BN (no activation) -> temp2
    Output = SiLU(temp1 + temp2)

Implementation:
- Conv3x3+BN and Conv1x1+BN are scalar loops (matching the C++ reference)
  using the AIE2P `invsqrt` intrinsic for BN's 1/sqrt(var+eps).
- The final Add+SiLU is vectorized over 16-element bf16 vectors with
  `getTanhBf16` (sigmoid(x) = 0.5 * (1 + tanh(0.5*x))) — matching the
  pattern from PythoC/pythoc_kernels/silu.py. This avoids the LLVM
  auto-vectorizer producing an <N x bf16> fptrunc the AIE2P backend
  can't legalize.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.utils.compile import compile_mlir_module
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import bf16, f32, i32, ptr
from pythoc.aie import (
    aie_vector,
    broadcast,
    getTanhBf16,
    invsqrt,
    load_v,
    store_v,
    vector_add,
    vector_mul,
)
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "repconv_pythoc_build"

BN_EPS = 1e-3


# ── PythoC kernel ──────────────────────────────────────────────────────


@aie_kernel
def repconv_bf16_kernel(
    input: ptr[bf16, True],
    weights_and_bn: ptr[bf16, True],
    output: ptr[bf16, True],
    temp1: ptr[bf16, True],
    temp2: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
    stride: i32,
    padding: i32,
):
    """RepConv = Conv3x3+BN + Conv1x1+BN -> Add -> SiLU.

    Scalar implementation that matches repconv_bf16.cc bit-for-bit (modulo
    HW approximation of invsqrt).
    """
    event0()

    bn_eps: f32 = 0.001
    half: f32 = 0.5
    one: f32 = 1.0
    two: f32 = 2.0
    zero: f32 = 0.0

    output_height: i32 = (height + 2 * padding - 3) // stride + 1
    output_width: i32 = (width + 2 * padding - 3) // stride + 1

    conv3x3_weight_size: i32 = out_channels * in_channels * 3 * 3
    conv1x1_weight_size: i32 = out_channels * in_channels

    # Weight section offsets (within weights_and_bn):
    #   [conv3x3_w | bn3x3_gamma | bn3x3_beta | bn3x3_mean | bn3x3_var |
    #    conv1x1_w | bn1x1_gamma | bn1x1_beta | bn1x1_mean | bn1x1_var]
    off_bn3x3_gamma: i32 = conv3x3_weight_size
    off_bn3x3_beta: i32 = off_bn3x3_gamma + out_channels
    off_bn3x3_mean: i32 = off_bn3x3_beta + out_channels
    off_bn3x3_var: i32 = off_bn3x3_mean + out_channels
    off_conv1x1: i32 = off_bn3x3_var + out_channels
    off_bn1x1_gamma: i32 = off_conv1x1 + conv1x1_weight_size
    off_bn1x1_beta: i32 = off_bn1x1_gamma + out_channels
    off_bn1x1_mean: i32 = off_bn1x1_gamma + 2 * out_channels
    off_bn1x1_var: i32 = off_bn1x1_gamma + 3 * out_channels

    # ── Stage 1: Conv3x3 + BN -> temp1 ───────────────────────────────
    oc: i32 = 0
    while oc < out_channels:
        gamma: f32 = f32(weights_and_bn[off_bn3x3_gamma + oc])
        beta: f32 = f32(weights_and_bn[off_bn3x3_beta + oc])
        mean: f32 = f32(weights_and_bn[off_bn3x3_mean + oc])
        var: f32 = f32(weights_and_bn[off_bn3x3_var + oc])
        inv_std: f32 = invsqrt(var + bn_eps)

        oh: i32 = 0
        while oh < output_height:
            ow: i32 = 0
            while ow < output_width:
                sum_val: f32 = zero

                ic: i32 = 0
                while ic < in_channels:
                    kh: i32 = 0
                    while kh < 3:
                        kw: i32 = 0
                        while kw < 3:
                            ih: i32 = oh * stride + kh - padding
                            iw: i32 = ow * stride + kw - padding
                            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                                input_idx: i32 = (ih * width + iw) * in_channels + ic
                                weight_idx: i32 = ((oc * in_channels + ic) * 3 + kh) * 3 + kw
                                a: f32 = f32(input[input_idx])
                                w: f32 = f32(weights_and_bn[weight_idx])
                                sum_val = sum_val + a * w
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1

                bn_out: f32 = gamma * (sum_val - mean) * inv_std + beta
                temp_idx: i32 = (oh * output_width + ow) * out_channels + oc
                temp1[temp_idx] = bf16(bn_out)
                ow = ow + 1
            oh = oh + 1
        oc = oc + 1

    # ── Stage 2: Conv1x1 + BN -> temp2 ───────────────────────────────
    oc = 0
    while oc < out_channels:
        gamma2: f32 = f32(weights_and_bn[off_bn1x1_gamma + oc])
        beta2: f32 = f32(weights_and_bn[off_bn1x1_beta + oc])
        mean2: f32 = f32(weights_and_bn[off_bn1x1_mean + oc])
        var2: f32 = f32(weights_and_bn[off_bn1x1_var + oc])
        inv_std2: f32 = invsqrt(var2 + bn_eps)

        oh2: i32 = 0
        while oh2 < output_height:
            ow2: i32 = 0
            while ow2 < output_width:
                sum_val2: f32 = zero

                ic2: i32 = 0
                while ic2 < in_channels:
                    ih2: i32 = oh2 * stride
                    iw2: i32 = ow2 * stride
                    if ih2 >= 0 and ih2 < height and iw2 >= 0 and iw2 < width:
                        input_idx2: i32 = (ih2 * width + iw2) * in_channels + ic2
                        weight_idx2: i32 = off_conv1x1 + oc * in_channels + ic2
                        a2: f32 = f32(input[input_idx2])
                        w2: f32 = f32(weights_and_bn[weight_idx2])
                        sum_val2 = sum_val2 + a2 * w2
                    ic2 = ic2 + 1

                bn_out2: f32 = gamma2 * (sum_val2 - mean2) * inv_std2 + beta2
                temp_idx2: i32 = (oh2 * output_width + ow2) * out_channels + oc
                temp2[temp_idx2] = bf16(bn_out2)
                ow2 = ow2 + 1
            oh2 = oh2 + 1
        oc = oc + 1

    # ── Stage 3: Add + SiLU -> output ────────────────────────────────
    # Vectorized: 16-element bf16 vectors. SiLU(x) = x * sigmoid(x),
    # sigmoid(x) ~ 0.5 * (1 + tanh(0.5 * x)). Matches silu_vectorized in
    # PythoC/pythoc_kernels/silu.py.
    output_size: i32 = output_height * output_width * out_channels
    half_bf: bf16 = 0.5
    one_bf: bf16 = 1.0
    v0_5: aie_vector[bf16, 16] = broadcast(bf16, 16, half_bf)
    v1: aie_vector[bf16, 16] = broadcast(bf16, 16, one_bf)

    i: i32 = 0
    while i < output_size:
        vt1: aie_vector[bf16, 16] = load_v(temp1 + i, 16)
        vt2: aie_vector[bf16, 16] = load_v(temp2 + i, 16)
        vsum: aie_vector[bf16, 16] = vector_add(vt1, vt2)
        half_x: aie_vector[bf16, 16] = vector_mul(vsum, v0_5)
        tanh_half_x: aie_vector[bf16, 16] = getTanhBf16(half_x)
        tanh_plus_1: aie_vector[bf16, 16] = vector_add(tanh_half_x, v1)
        sigmoid: aie_vector[bf16, 16] = vector_mul(tanh_plus_1, v0_5)
        result: aie_vector[bf16, 16] = vector_mul(vsum, sigmoid)
        store_v(output + i, result)
        i = i + 16

    event1()


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 RepConv layer (PythoC + IRON, bf16)"
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", "-ic", type=int, default=8)
    parser.add_argument("--out-channels", "-oc", type=int, default=8)
    parser.add_argument("--stride", "-s", type=int, default=1)
    parser.add_argument("--padding", "-p", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── MLIR / IRON construction ──────────────────────────────────────────


def _sizes(args):
    output_height = (args.height + 2 * args.padding - 3) // args.stride + 1
    output_width = (args.width + 2 * args.padding - 3) // args.stride + 1
    input_size = args.height * args.width * args.in_channels
    output_size = output_height * output_width * args.out_channels
    temp_size = output_size
    conv3x3_w = args.out_channels * args.in_channels * 9
    conv1x1_w = args.out_channels * args.in_channels
    bn_w = 4 * args.out_channels
    total_w = conv3x3_w + bn_w + conv1x1_w + bn_w
    return output_height, output_width, input_size, output_size, temp_size, total_w


def build_mlir_module(device, args):
    oh, ow, input_size, output_size, temp_size, total_w = _sizes(args)

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_w,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    temp_ty = np.ndarray[(temp_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        repconv_bf16_kernel,
        [
            input_ty,
            weight_ty,
            output_ty,
            temp_ty,
            temp_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        extra_globals={
            "invsqrt": invsqrt,
            "getTanhBf16": getTanhBf16,
        },
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    temp1_buffer = Buffer(temp_ty, name="temp1_conv3x3")
    temp2_buffer = Buffer(temp_ty, name="temp2_conv1x1")

    def core_fn(of_in, of_wts, of_out, kernel, temp1, temp2):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            temp1,
            temp2,
            args.height,
            args.width,
            args.in_channels,
            args.out_channels,
            args.stride,
            args.padding,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_input.cons(),
            of_weights.cons(),
            of_output.prod(),
            kernel,
            temp1_buffer,
            temp2_buffer,
        ],
        stack_size=4096,
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


# ── Numpy reference (matches the kernel byte-for-byte semantics) ──────


def _tanh_sigmoid_np(x):
    """Sigmoid via tanh: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def numpy_reference(input_bf16, weights_bf16, args):
    """Mirror the kernel's scalar implementation exactly in fp32+bf16."""
    H, W, IC, OC = args.height, args.width, args.in_channels, args.out_channels
    S, P = args.stride, args.padding
    oh = (H + 2 * P - 3) // S + 1
    ow = (W + 2 * P - 3) // S + 1

    conv3x3_size = OC * IC * 9
    conv1x1_size = OC * IC

    # Slice weights
    w3 = weights_bf16[0:conv3x3_size].reshape(OC, IC, 3, 3).astype(np.float32)
    base = conv3x3_size
    gamma3 = weights_bf16[base : base + OC].astype(np.float32); base += OC
    beta3  = weights_bf16[base : base + OC].astype(np.float32); base += OC
    mean3  = weights_bf16[base : base + OC].astype(np.float32); base += OC
    var3   = weights_bf16[base : base + OC].astype(np.float32); base += OC
    w1 = weights_bf16[base : base + conv1x1_size].reshape(OC, IC).astype(np.float32); base += conv1x1_size
    gamma1 = weights_bf16[base : base + OC].astype(np.float32); base += OC
    beta1  = weights_bf16[base : base + OC].astype(np.float32); base += OC
    mean1  = weights_bf16[base : base + OC].astype(np.float32); base += OC
    var1   = weights_bf16[base : base + OC].astype(np.float32); base += OC

    inp = input_bf16.reshape(H, W, IC).astype(np.float32)

    temp1 = np.zeros((oh, ow, OC), dtype=np.float32)
    inv_std3 = 1.0 / np.sqrt(var3 + BN_EPS)
    for o in range(OC):
        for y in range(oh):
            for x in range(ow):
                s = 0.0
                for ic in range(IC):
                    for kh in range(3):
                        for kw in range(3):
                            ih = y * S + kh - P
                            iw = x * S + kw - P
                            if 0 <= ih < H and 0 <= iw < W:
                                s += inp[ih, iw, ic] * w3[o, ic, kh, kw]
                bn_out = gamma3[o] * (s - mean3[o]) * inv_std3[o] + beta3[o]
                # Match kernel storing as bf16
                temp1[y, x, o] = float(np.float32(bn_out).astype(bfloat16))

    temp2 = np.zeros((oh, ow, OC), dtype=np.float32)
    inv_std1 = 1.0 / np.sqrt(var1 + BN_EPS)
    for o in range(OC):
        for y in range(oh):
            for x in range(ow):
                s = 0.0
                ih = y * S
                iw = x * S
                if 0 <= ih < H and 0 <= iw < W:
                    for ic in range(IC):
                        s += inp[ih, iw, ic] * w1[o, ic]
                bn_out = gamma1[o] * (s - mean1[o]) * inv_std1[o] + beta1[o]
                temp2[y, x, o] = float(np.float32(bn_out).astype(bfloat16))

    # Stage 3 runs on bf16 vectors: cast inputs to bf16 first to match HW.
    summed_bf16 = (temp1 + temp2).astype(bfloat16)
    summed_f32 = summed_bf16.astype(np.float32)
    half_x = (summed_bf16 * np.float32(0.5)).astype(bfloat16).astype(np.float32)
    tanh_half = np.tanh(half_x).astype(bfloat16).astype(np.float32)
    tanh_plus_1 = (tanh_half + 1.0).astype(bfloat16).astype(np.float32)
    sigmoid = (tanh_plus_1 * np.float32(0.5)).astype(bfloat16).astype(np.float32)
    out_f32 = (summed_f32 * sigmoid).astype(bfloat16).astype(np.float32)
    return out_f32.flatten()


# ── Compile & run ─────────────────────────────────────────────────────


def run_with_xrt(xclbin_path: Path, insts_path: Path, args):
    oh, ow, input_size, output_size, temp_size, total_w = _sizes(args)

    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    rng = np.random.default_rng(42)
    inp_f32 = rng.standard_normal(input_size).astype(np.float32) * 0.5
    inp_bf16 = inp_f32.astype(bfloat16)

    # Weights: small random conv filters; reasonable BN params (gamma~1, beta~0,
    # mean~0, var ~ 0.5 .. 1.5).
    conv3x3_size = args.out_channels * args.in_channels * 9
    conv1x1_size = args.out_channels * args.in_channels
    OC = args.out_channels

    w3 = (rng.standard_normal(conv3x3_size).astype(np.float32) * 0.1)
    w1 = (rng.standard_normal(conv1x1_size).astype(np.float32) * 0.1)
    gamma3 = (rng.uniform(0.8, 1.2, size=OC)).astype(np.float32)
    beta3 = (rng.uniform(-0.1, 0.1, size=OC)).astype(np.float32)
    mean3 = (rng.uniform(-0.2, 0.2, size=OC)).astype(np.float32)
    var3 = (rng.uniform(0.5, 1.5, size=OC)).astype(np.float32)
    gamma1 = (rng.uniform(0.8, 1.2, size=OC)).astype(np.float32)
    beta1 = (rng.uniform(-0.1, 0.1, size=OC)).astype(np.float32)
    mean1 = (rng.uniform(-0.2, 0.2, size=OC)).astype(np.float32)
    var1 = (rng.uniform(0.5, 1.5, size=OC)).astype(np.float32)

    weights_f32 = np.concatenate([
        w3, gamma3, beta3, mean3, var3,
        w1, gamma1, beta1, mean1, var1,
    ])
    weights_bf16 = weights_f32.astype(bfloat16)

    in_input = iron.tensor(inp_bf16.view(np.uint16), dtype=np.uint16)
    in_weights = iron.tensor(weights_bf16.view(np.uint16), dtype=np.uint16)
    out_buf = iron.zeros(output_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_input, in_weights, out_buf])

    out_u16 = out_buf.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = numpy_reference(inp_bf16, weights_bf16, args)
    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()

    try:
        oh, ow, input_size, output_size, temp_size, total_w = _sizes(args)
        print(
            f"[1/3] Building IRON program ({args.height}x{args.width}x{args.in_channels} "
            f"-> {oh}x{ow}x{args.out_channels}, stride={args.stride}, padding={args.padding})"
        )
        module = build_mlir_module(device, args)
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
        actual, expected = run_with_xrt(xclbin_path, insts_path, args)
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        # invsqrt is a HW approximation (few-ULP error). Loose tolerance since
        # error compounds through 2 BN stages + sigmoid approximation.
        if np.allclose(actual, expected, rtol=5e-2, atol=5e-2):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=5e-2, atol=5e-2)
        print(f"FAILED: {int(mism.sum())}/{len(actual)} mismatches")
        for i in np.where(mism)[0][:10]:
            print(f"        [{i}] got {actual[i]}, expected {expected[i]}")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
