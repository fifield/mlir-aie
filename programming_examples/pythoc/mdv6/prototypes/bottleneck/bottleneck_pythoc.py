#!/usr/bin/env python3
# bottleneck_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./bottleneck_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 Bottleneck block as a PythoC + IRON single-file example.

Port of programming_examples/ml/mdv6/bottleneck/{aie2.py, bottleneck_bf16.cc}
that replaces the external C++ kernel with an inline PythoC kernel.

Bottleneck = RepConv -> Conv+BN+SiLU -> optional residual add
where RepConv = (Conv3x3+BN) + (Conv1x1+BN) -> Add -> SiLU.

The kernel mirrors the C++ scalar reference closely:
  - per-channel BN normalization uses the AIE2P `invsqrt` scalar intrinsic
  - SiLU uses `getTanhBf16` (sigmoid(x) ~= 0.5 * (1 + tanh(0.5 * x)))
  - all reductions accumulate in f32, weights/inputs/outputs are bf16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import ptr, i32, bf16, f32
from pythoc.aie import (
    aie_vector,
    broadcast,
    load_v,
    store_v,
    vector_add,
    vector_mul,
    invsqrt,
    getTanhBf16,
)
from pythoc.aie.profiling import event0, event1


DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "bottleneck_pythoc_build"


# ── Extra globals for @aie_kernel ──────────────────────────────────────────
# The PythocKernel inline compiler's default user_globals does not contain
# `invsqrt`, `getTanhBf16`, `broadcast` is already there but the scalar
# intrinsics need to be injected manually.
KERNEL_EXTRA_GLOBALS = {
    "invsqrt": invsqrt,
    "getTanhBf16": getTanhBf16,
}


# ── PythoC kernel ──────────────────────────────────────────────────────────
#
# Scalar reference implementation matching bottleneck_bf16.cc::bottleneck_bf16_scalar.
# All convolution accumulations are in f32 to preserve precision; SiLU uses the
# tanh-based sigmoid approximation (instead of the simple x/(2*(1+|x|)) used
# in the C++ reference) so we can leverage the hardware getTanhBf16 LUT.
#
# Note: this is a scalar (non-vectorized) kernel: simple, correct, fits in one
# function. Vectorization is left as future work — the C++ reference is also
# scalar.


@aie_kernel
def bottleneck_bf16_pythoc(
    input: ptr[bf16, True],            # HxWxIC input feature map
    weights_and_bn: ptr[bf16, True],   # packed weights+BN: see layout below
    output: ptr[bf16, True],           # OHxOWxOC output feature map
    input_copy: ptr[bf16, True],       # IRON scratch buffer (residual)
    temp1: ptr[bf16, True],            # Conv3x3+BN  (RepConv branch 1)
    temp2: ptr[bf16, True],            # Conv1x1+BN  (RepConv branch 2)
    temp3: ptr[bf16, True],            # RepConv output (Add + SiLU)
    temp4: ptr[bf16, True],            # Final conv+BN+SiLU output
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
    stride: i32,
    padding: i32,
    residual: i32,
):
    """Bottleneck (RepConv -> Conv+BN+SiLU -> optional residual)."""
    event0()

    bn_eps: f32 = 1.0e-3
    half: bf16 = bf16(0.5)
    one_bf16: bf16 = bf16(1.0)

    output_height: i32 = (height + 2 * padding - 3) // stride + 1
    output_width: i32 = (width + 2 * padding - 3) // stride + 1
    output_size: i32 = output_height * output_width * out_channels

    # ── Copy input → input_copy (only used if residual) ────────────────
    # Copy unconditionally — simpler and the buffer is small. Use explicit
    # 16-wide bf16 loads/stores to keep llc's auto-vectorizer from emitting
    # a 32-wide bf16 G_FADD (which the AIE2P legalizer can't handle).
    input_size: i32 = height * width * in_channels
    ci: i32 = 0
    while ci < input_size:
        vc: aie_vector[bf16, 16] = load_v(input + ci, 16)
        store_v(input_copy + ci, vc)
        ci = ci + 16

    # ── Weight pointer arithmetic (matches C++ layout) ────────────────
    conv3x3_weight_size: i32 = out_channels * in_channels * 3 * 3
    conv1x1_weight_size: i32 = out_channels * in_channels * 1 * 1
    conv2_weight_size: i32 = out_channels * out_channels * 3 * 3

    # RepConv 3x3 weights + BN params (gamma, beta, mean, var, each [OC])
    conv3x3_weights: ptr[bf16] = weights_and_bn
    bn3x3_weight: ptr[bf16] = conv3x3_weights + conv3x3_weight_size
    bn3x3_bias: ptr[bf16] = bn3x3_weight + out_channels
    bn3x3_mean: ptr[bf16] = bn3x3_bias + out_channels
    bn3x3_var: ptr[bf16] = bn3x3_mean + out_channels

    # RepConv 1x1 weights + BN params
    conv1x1_weights: ptr[bf16] = bn3x3_var + out_channels
    bn1x1_weight: ptr[bf16] = conv1x1_weights + conv1x1_weight_size
    bn1x1_bias: ptr[bf16] = bn1x1_weight + out_channels
    bn1x1_mean: ptr[bf16] = bn1x1_bias + out_channels
    bn1x1_var: ptr[bf16] = bn1x1_mean + out_channels

    # Conv2 (3x3 on RepConv output) + BN params
    conv2_weights: ptr[bf16] = bn1x1_var + out_channels
    bn2_weight: ptr[bf16] = conv2_weights + conv2_weight_size
    bn2_bias: ptr[bf16] = bn2_weight + out_channels
    bn2_mean: ptr[bf16] = bn2_bias + out_channels
    bn2_var: ptr[bf16] = bn2_mean + out_channels

    # =====================================================================
    # Stage 1a: Conv3x3 + BN  →  temp1
    # =====================================================================
    oc: i32 = 0
    while oc < out_channels:
        gamma: f32 = f32(bn3x3_weight[oc])
        beta: f32 = f32(bn3x3_bias[oc])
        mean: f32 = f32(bn3x3_mean[oc])
        var: f32 = f32(bn3x3_var[oc])
        inv_std: f32 = invsqrt(var + bn_eps)

        oh: i32 = 0
        while oh < output_height:
            ow: i32 = 0
            while ow < output_width:
                acc: f32 = 0.0
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
                                acc = acc + f32(input[input_idx]) * f32(conv3x3_weights[weight_idx])
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1

                bn_out: f32 = gamma * (acc - mean) * inv_std + beta
                temp_idx: i32 = (oh * output_width + ow) * out_channels + oc
                temp1[temp_idx] = bf16(bn_out)
                ow = ow + 1
            oh = oh + 1
        oc = oc + 1

    # =====================================================================
    # Stage 1b: Conv1x1 + BN  →  temp2
    # =====================================================================
    oc2: i32 = 0
    while oc2 < out_channels:
        gamma2: f32 = f32(bn1x1_weight[oc2])
        beta2: f32 = f32(bn1x1_bias[oc2])
        mean2: f32 = f32(bn1x1_mean[oc2])
        var2: f32 = f32(bn1x1_var[oc2])
        inv_std2: f32 = invsqrt(var2 + bn_eps)

        oh2: i32 = 0
        while oh2 < output_height:
            ow2: i32 = 0
            while ow2 < output_width:
                acc2: f32 = 0.0
                ic2: i32 = 0
                while ic2 < in_channels:
                    ih2: i32 = oh2 * stride
                    iw2: i32 = ow2 * stride
                    if ih2 >= 0 and ih2 < height and iw2 >= 0 and iw2 < width:
                        input_idx2: i32 = (ih2 * width + iw2) * in_channels + ic2
                        weight_idx2: i32 = oc2 * in_channels + ic2
                        acc2 = acc2 + f32(input[input_idx2]) * f32(conv1x1_weights[weight_idx2])
                    ic2 = ic2 + 1

                bn_out2: f32 = gamma2 * (acc2 - mean2) * inv_std2 + beta2
                temp_idx2: i32 = (oh2 * output_width + ow2) * out_channels + oc2
                temp2[temp_idx2] = bf16(bn_out2)
                ow2 = ow2 + 1
            oh2 = oh2 + 1
        oc2 = oc2 + 1

    # =====================================================================
    # Stage 1c: Add + SiLU  →  temp3   (RepConv output)
    #
    # silu(x) = x * sigmoid(x)
    # sigmoid(x) ~= 0.5 * (1 + tanh(0.5 * x))   via getTanhBf16 LUT
    #
    # Vectorized: 16 bf16 elements per iteration. output_size is small for
    # the default H=W=8, IC=OC=8 case (= 512), divisible by 16.
    # =====================================================================
    v_half: aie_vector[bf16, 16] = broadcast(bf16, 16, half)
    v_one: aie_vector[bf16, 16] = broadcast(bf16, 16, one_bf16)

    i_silu: i32 = 0
    while i_silu < output_size:
        va: aie_vector[bf16, 16] = load_v(temp1 + i_silu, 16)
        vb: aie_vector[bf16, 16] = load_v(temp2 + i_silu, 16)
        vsum: aie_vector[bf16, 16] = vector_add(va, vb)
        half_x: aie_vector[bf16, 16] = vector_mul(vsum, v_half)
        tanh_half: aie_vector[bf16, 16] = getTanhBf16(half_x)
        tanh_plus_1: aie_vector[bf16, 16] = vector_add(tanh_half, v_one)
        sig: aie_vector[bf16, 16] = vector_mul(tanh_plus_1, v_half)
        out_silu: aie_vector[bf16, 16] = vector_mul(vsum, sig)
        store_v(temp3 + i_silu, out_silu)
        i_silu = i_silu + 16

    # =====================================================================
    # Stage 2: Conv3x3 + BN + SiLU on temp3  →  temp4
    # =====================================================================
    oc3: i32 = 0
    while oc3 < out_channels:
        gamma3: f32 = f32(bn2_weight[oc3])
        beta3: f32 = f32(bn2_bias[oc3])
        mean3: f32 = f32(bn2_mean[oc3])
        var3: f32 = f32(bn2_var[oc3])
        inv_std3: f32 = invsqrt(var3 + bn_eps)

        oh3: i32 = 0
        while oh3 < output_height:
            ow3: i32 = 0
            while ow3 < output_width:
                acc3: f32 = 0.0

                # 3x3 conv on temp3 (output_height x output_width x out_channels)
                ic3: i32 = 0
                while ic3 < out_channels:
                    kh3: i32 = 0
                    while kh3 < 3:
                        kw3: i32 = 0
                        while kw3 < 3:
                            ih3: i32 = oh3 * stride + kh3 - padding
                            iw3: i32 = ow3 * stride + kw3 - padding
                            if ih3 >= 0 and ih3 < output_height and iw3 >= 0 and iw3 < output_width:
                                src_idx: i32 = (ih3 * output_width + iw3) * out_channels + ic3
                                w_idx: i32 = ((oc3 * out_channels + ic3) * 3 + kh3) * 3 + kw3
                                acc3 = acc3 + f32(temp3[src_idx]) * f32(conv2_weights[w_idx])
                            kw3 = kw3 + 1
                        kh3 = kh3 + 1
                    ic3 = ic3 + 1

                bn_out3: f32 = gamma3 * (acc3 - mean3) * inv_std3 + beta3
                # SiLU on a scalar via the C++-reference simple sigmoid form:
                #   sigmoid(x) ~= 0.5 + x / (2 * (1 + |x|))
                # silu(x) = x * sigmoid(x)
                abs_x: f32 = bn_out3
                if abs_x < 0.0:
                    abs_x = -abs_x
                sig3: f32 = 0.5 + bn_out3 / (2.0 * (1.0 + abs_x))
                silu_out3: f32 = bn_out3 * sig3
                out_idx3: i32 = (oh3 * output_width + ow3) * out_channels + oc3
                temp4[out_idx3] = bf16(silu_out3)
                ow3 = ow3 + 1
            oh3 = oh3 + 1
        oc3 = oc3 + 1

    # =====================================================================
    # Stage 3: Residual + writeback to output
    #   residual condition: residual!=0 AND in_channels==out_channels AND
    #                       height==output_height AND width==output_width
    # =====================================================================
    if residual != 0 and in_channels == out_channels and height == output_height and width == output_width:
        # Vectorized add (16-wide) — keeps llc legalizer happy
        i_out: i32 = 0
        while i_out < output_size:
            vic: aie_vector[bf16, 16] = load_v(input_copy + i_out, 16)
            vt4: aie_vector[bf16, 16] = load_v(temp4 + i_out, 16)
            vsum_out: aie_vector[bf16, 16] = vector_add(vic, vt4)
            store_v(output + i_out, vsum_out)
            i_out = i_out + 16
    else:
        # Vectorized copy
        j_out: i32 = 0
        while j_out < output_size:
            vj: aie_vector[bf16, 16] = load_v(temp4 + j_out, 16)
            store_v(output + j_out, vj)
            j_out = j_out + 16

    event1()


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="MDV6 Bottleneck (PythoC + IRON, bf16)"
    )
    p.add_argument("--device", choices=("npu2",), default="npu2")
    p.add_argument("--height", "-ht", type=int, default=8)
    p.add_argument("--width", "-wd", type=int, default=8)
    p.add_argument("--in-channels", "-ic", type=int, default=8)
    p.add_argument("--out-channels", "-oc", type=int, default=8)
    p.add_argument("--stride", "-s", type=int, default=1)
    p.add_argument("--padding", "-pd", type=int, default=1)
    p.add_argument("--residual", "-r", type=int, default=1, help="0 or 1")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.4,
                   help="max abs diff vs PyTorch reference (default 0.4)")
    return p.parse_args()


# ── MLIR / IRON construction ───────────────────────────────────────────


def build_mlir_module(
    device, height, width, in_channels, out_channels,
    stride, padding, residual,
):
    output_height = (height + 2 * padding - 3) // stride + 1
    output_width = (width + 2 * padding - 3) // stride + 1
    input_size = height * width * in_channels
    output_size = output_height * output_width * out_channels
    temp_size = output_size

    # weight layout: [conv3x3 | bn3x3(g,b,m,v) | conv1x1 | bn1x1(g,b,m,v) |
    #                 conv2   | bn2  (g,b,m,v)]
    conv3x3_w_sz = out_channels * in_channels * 3 * 3
    conv1x1_w_sz = out_channels * in_channels * 1 * 1
    bn_param_sz = 4 * out_channels
    conv2_w_sz = out_channels * out_channels * 3 * 3
    total_weight_size = (
        conv3x3_w_sz + bn_param_sz + conv1x1_w_sz + bn_param_sz
        + conv2_w_sz + bn_param_sz
    )

    if output_size % 16 != 0:
        raise ValueError(
            f"output_size {output_size} must be divisible by 16 "
            "(vectorized SiLU uses 16-wide bf16 vectors)"
        )
    if input_size % 16 != 0:
        raise ValueError(
            f"input_size {input_size} must be divisible by 16 "
            "(vectorized residual input-copy uses 16-wide bf16 vectors)"
        )

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    temp_ty = np.ndarray[(temp_size,), np.dtype[np.uint16]]
    input_copy_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        bottleneck_bf16_pythoc,
        [
            input_ty,
            weight_ty,
            output_ty,
            input_copy_ty,
            temp_ty, temp_ty, temp_ty, temp_ty,
            np.int32, np.int32, np.int32, np.int32,
            np.int32, np.int32, np.int32,
        ],
        extra_globals=KERNEL_EXTRA_GLOBALS,
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    input_copy_buf = Buffer(input_copy_ty, name="input_copy")
    temp1_buf = Buffer(temp_ty, name="temp1_conv3x3")
    temp2_buf = Buffer(temp_ty, name="temp2_conv1x1")
    temp3_buf = Buffer(temp_ty, name="temp3_repconv")
    temp4_buf = Buffer(temp_ty, name="temp4_conv2")

    def core_fn(of_in, of_wts, of_out, kernel,
                input_copy, t1, t2, t3, t4):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(
            elem_in, elem_wts, elem_out, input_copy,
            t1, t2, t3, t4,
            height, width, in_channels, out_channels,
            stride, padding, residual,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_input.cons(), of_weights.cons(), of_output.prod(), kernel,
            input_copy_buf, temp1_buf, temp2_buf, temp3_buf, temp4_buf,
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

    module = Program(device, runtime, workers=[worker]).resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── Reference + run on hardware ───────────────────────────────────────


def _silu_tanh(x: np.ndarray) -> np.ndarray:
    """SiLU via tanh-based sigmoid (matches stage-1c kernel path)."""
    return x * 0.5 * (1.0 + np.tanh(0.5 * x.astype(np.float32)))


def _silu_simple(x: np.ndarray) -> np.ndarray:
    """SiLU via simple sigmoid 0.5 + x/(2*(1+|x|))  (matches stage-2 kernel path)."""
    x = x.astype(np.float32)
    return x * (0.5 + x / (2.0 * (1.0 + np.abs(x))))


def numpy_reference(
    input_hwc_bf16: np.ndarray,
    weights_bf16: np.ndarray,
    height, width, in_channels, out_channels,
    stride, padding, residual,
):
    """Reference implementation matching the bottleneck PythoC kernel.

    Uses the same tanh-based SiLU approximation; computes in fp32. Done on
    bf16-rounded values to allow tight comparison with hardware output.
    """
    output_height = (height + 2 * padding - 3) // stride + 1
    output_width = (width + 2 * padding - 3) // stride + 1

    x = input_hwc_bf16.astype(np.float32).reshape(height, width, in_channels)
    w = weights_bf16.astype(np.float32)

    conv3x3_w_sz = out_channels * in_channels * 9
    conv1x1_w_sz = out_channels * in_channels
    bn_sz = 4 * out_channels
    conv2_w_sz = out_channels * out_channels * 9

    off = 0
    conv3x3 = w[off:off + conv3x3_w_sz].reshape(out_channels, in_channels, 3, 3); off += conv3x3_w_sz
    bn3 = w[off:off + bn_sz].reshape(4, out_channels); off += bn_sz
    conv1x1 = w[off:off + conv1x1_w_sz].reshape(out_channels, in_channels); off += conv1x1_w_sz
    bn1 = w[off:off + bn_sz].reshape(4, out_channels); off += bn_sz
    conv2 = w[off:off + conv2_w_sz].reshape(out_channels, out_channels, 3, 3); off += conv2_w_sz
    bn2 = w[off:off + bn_sz].reshape(4, out_channels); off += bn_sz

    eps = 1e-3

    def conv3x3_bn(src, conv, bn):
        # src: (H, W, IC); conv: (OC, IC, 3, 3); bn: (4, OC) = (g,b,m,v)
        H, W, IC = src.shape
        OC = conv.shape[0]
        out = np.zeros((output_height, output_width, OC), dtype=np.float32)
        gamma, beta, mean, var = bn[0], bn[1], bn[2], bn[3]
        inv_std = 1.0 / np.sqrt(var + eps)
        for oc in range(OC):
            for oh in range(output_height):
                for ow in range(output_width):
                    acc = 0.0
                    for ic in range(IC):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                if 0 <= ih < H and 0 <= iw < W:
                                    acc += src[ih, iw, ic] * conv[oc, ic, kh, kw]
                    out[oh, ow, oc] = gamma[oc] * (acc - mean[oc]) * inv_std[oc] + beta[oc]
        return out

    def conv1x1_bn(src, conv, bn):
        H, W, IC = src.shape
        OC = conv.shape[0]
        out = np.zeros((output_height, output_width, OC), dtype=np.float32)
        gamma, beta, mean, var = bn[0], bn[1], bn[2], bn[3]
        inv_std = 1.0 / np.sqrt(var + eps)
        for oc in range(OC):
            for oh in range(output_height):
                for ow in range(output_width):
                    acc = 0.0
                    ih = oh * stride
                    iw = ow * stride
                    if 0 <= ih < H and 0 <= iw < W:
                        for ic in range(IC):
                            acc += src[ih, iw, ic] * conv[oc, ic]
                    out[oh, ow, oc] = gamma[oc] * (acc - mean[oc]) * inv_std[oc] + beta[oc]
        return out

    t1 = conv3x3_bn(x, conv3x3, bn3)
    # Round intermediate to bf16 to match what the kernel stores
    t1 = t1.astype(bfloat16).astype(np.float32)

    t2 = conv1x1_bn(x, conv1x1, bn1)
    t2 = t2.astype(bfloat16).astype(np.float32)

    t3 = _silu_tanh((t1 + t2).astype(bfloat16).astype(np.float32))
    t3 = t3.astype(bfloat16).astype(np.float32)

    t4 = conv3x3_bn(t3, conv2, bn2)
    # Stage 2 uses the simple-sigmoid scalar SiLU (matches kernel stage 2)
    t4 = _silu_simple(t4)
    t4 = t4.astype(bfloat16).astype(np.float32)

    if residual and in_channels == out_channels and height == output_height and width == output_width:
        out = (x.astype(np.float32) + t4).astype(bfloat16).astype(np.float32)
    else:
        out = t4

    return out


def run_with_xrt(xclbin_path: Path, insts_path: Path, args):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    H, W, IC, OC = args.height, args.width, args.in_channels, args.out_channels
    stride, padding, residual = args.stride, args.padding, args.residual

    output_height = (H + 2 * padding - 3) // stride + 1
    output_width = (W + 2 * padding - 3) // stride + 1
    input_size = H * W * IC
    output_size = output_height * output_width * OC

    rng = np.random.default_rng(42)
    # Inputs (smaller magnitude → keeps BN stable)
    input_f32 = (rng.standard_normal(input_size).astype(np.float32) * 0.5)
    input_bf16 = input_f32.astype(bfloat16)

    # Weights
    conv3x3_w_sz = OC * IC * 9
    conv1x1_w_sz = OC * IC
    bn_sz = 4 * OC
    conv2_w_sz = OC * OC * 9

    conv3x3_w = rng.standard_normal(conv3x3_w_sz).astype(np.float32) * 0.3
    bn3_g = rng.standard_normal(OC).astype(np.float32) * 0.5 + 1.0
    bn3_b = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn3_m = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn3_v = np.abs(rng.standard_normal(OC).astype(np.float32)) + 0.5

    conv1x1_w = rng.standard_normal(conv1x1_w_sz).astype(np.float32) * 0.3
    bn1_g = rng.standard_normal(OC).astype(np.float32) * 0.5 + 1.0
    bn1_b = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn1_m = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn1_v = np.abs(rng.standard_normal(OC).astype(np.float32)) + 0.5

    conv2_w = rng.standard_normal(conv2_w_sz).astype(np.float32) * 0.3
    bn2_g = rng.standard_normal(OC).astype(np.float32) * 0.5 + 1.0
    bn2_b = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn2_m = rng.standard_normal(OC).astype(np.float32) * 0.1
    bn2_v = np.abs(rng.standard_normal(OC).astype(np.float32)) + 0.5

    weights_flat = np.concatenate([
        conv3x3_w,
        bn3_g, bn3_b, bn3_m, bn3_v,
        conv1x1_w,
        bn1_g, bn1_b, bn1_m, bn1_v,
        conv2_w,
        bn2_g, bn2_b, bn2_m, bn2_v,
    ]).astype(bfloat16)

    in_input = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    in_weights = iron.tensor(weights_flat.view(np.uint16), dtype=np.uint16)
    out_tensor = iron.zeros(output_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_input, in_weights, out_tensor])

    actual_u16 = out_tensor.numpy()
    actual_f32 = np.array(actual_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = numpy_reference(
        input_bf16, weights_flat, H, W, IC, OC,
        stride, padding, residual,
    ).reshape(-1)

    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()

    try:
        print(
            f"[1/3] Building IRON program "
            f"(H={args.height} W={args.width} IC={args.in_channels} "
            f"OC={args.out_channels} stride={args.stride} "
            f"padding={args.padding} residual={args.residual})"
        )
        module = build_mlir_module(
            device,
            args.height, args.width, args.in_channels, args.out_channels,
            args.stride, args.padding, args.residual,
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
        actual, expected = run_with_xrt(xclbin_path, insts_path, args)
        print(f"      Output (first 8):   {actual[:8]}")
        print(f"      Expected (first 8): {expected[:8]}")
        max_abs = float(np.max(np.abs(actual - expected)))
        mean_abs = float(np.mean(np.abs(actual - expected)))
        print(f"      Max abs diff:  {max_abs:.6f}")
        print(f"      Mean abs diff: {mean_abs:.6f}")

        if max_abs < args.tolerance:
            print("PASS!")
            return 0
        print(f"FAILED: max abs diff {max_abs:.6f} >= tolerance {args.tolerance}")
        # Show a few mismatches
        diffs = np.abs(actual - expected)
        worst = np.argsort(diffs)[-5:][::-1]
        for i in worst:
            print(f"        [{i}] got {actual[i]:.6f}, expected {expected[i]:.6f} (|diff|={diffs[i]:.6f})")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
