#!/usr/bin/env python3
# sppelan_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 16 --out-channels 16 --work-dir ./sppelan_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 SPPELAN (Spatial Pyramid Pooling + ELAN) layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/sppelan/{aie2.py,sppelan_bf16.cc} that
replaces the external C++ kernel with an inline PythoC kernel.

Architecture (bfloat16, HWC layout):

    Input -> Conv1 (1x1) + BN + SiLU -> f0
                          |
                       MaxPool(5x5, s=1, p=2) -> f1
                          |
                       MaxPool(5x5, s=1, p=2) -> f2
                          |
                       MaxPool(5x5, s=1, p=2) -> f3
                          |
                  Concat[f0, f1, f2, f3] (4-way along channel)
                          |
                       Conv5 (1x1) + BN + SiLU -> Output

BatchNorm collapse:
    PyTorch BN is gamma*(x-mean)*inv_std + beta. To keep the on-device
    kernel small and avoid sqrt, we collapse this on the host to a per-
    channel affine y = w*x + b with
        w = gamma * inv_std,  inv_std = 1/sqrt(var + eps)
        b = beta - w * mean
    The device-side kernel then implements plain BN as w*x + b followed
    by SiLU using the LUT-based tanh approximation (getTanhBf16).

MaxPool uses scalar loops with explicit boundary checks; the spatial
sizes used by SPPELAN in MDV6 are small (H=W=8 typically) so we don't
need a vector implementation.

Channel concat is a flat scalar copy over the 4 features.
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

from pythoc import ptr, i32, bf16
from pythoc.aie import (
    aie_vector,
    broadcast,
    extract_elem,
    getTanhBf16,
)
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "sppelan_pythoc_build"


# --- PythoC kernel: SPPELAN -------------------------------------------
#
# Signature mirrors the original sppelan_bf16 C kernel:
#   input            -- (H, W, in_channels) bf16, HWC
#   weights_and_bn   -- packed: [conv1_w, conv1_bn_w, conv1_bn_b,
#                                conv5_w, conv5_bn_w, conv5_bn_b]
#   output           -- (H, W, out_channels) bf16
#   conv1_output     -- (H, W, neck_channels) scratch buffer (f0)
#   pool1_output     -- (H, W, neck_channels) scratch buffer (f1)
#   pool2_output     -- (H, W, neck_channels) scratch buffer (f2)
#   pool3_output     -- (H, W, neck_channels) scratch buffer (f3)
#   concat_buffer    -- (H, W, 4*neck_channels) scratch buffer
#   height, width    -- spatial dimensions
#   in_channels, out_channels, neck_channels  -- channel counts
#   kernel_size      -- maxpool kernel (typically 5)
#   stride           -- maxpool stride  (typically 1)
#   padding          -- maxpool padding (typically 2)
#
# The weight block has been pre-collapsed on the host so that the device
# only needs the affine BN form (w*x + b).  Layout:
#   conv1_w (neck * in_channels)
#   conv1_bn_w (neck)            ## per-channel: gamma * inv_std
#   conv1_bn_b (neck)            ## per-channel: beta - w*mean
#   conv5_w (out_channels * 4*neck)
#   conv5_bn_w (out_channels)
#   conv5_bn_b (out_channels)


@aie_kernel
def sppelan_bf16_kernel(
    input: ptr[bf16, True],
    weights_and_bn: ptr[bf16, True],
    output: ptr[bf16, True],
    conv1_output: ptr[bf16, True],
    pool1_output: ptr[bf16, True],
    pool2_output: ptr[bf16, True],
    pool3_output: ptr[bf16, True],
    concat_buffer: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
    neck_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
):
    event0()

    spatial: i32 = height * width
    concat_channels: i32 = 4 * neck_channels

    # ----- Carve up weights_and_bn -------------------------------------
    conv1_w_size: i32 = neck_channels * in_channels
    conv1_w: ptr[bf16] = weights_and_bn
    conv1_bn_w: ptr[bf16] = conv1_w + conv1_w_size
    conv1_bn_b: ptr[bf16] = conv1_bn_w + neck_channels

    conv5_w_size: i32 = out_channels * concat_channels
    conv5_w: ptr[bf16] = conv1_bn_b + neck_channels
    conv5_bn_w: ptr[bf16] = conv5_w + conv5_w_size
    conv5_bn_b: ptr[bf16] = conv5_bn_w + out_channels

    # SiLU constants (vectors) for the C-axis-vectorised activation path.
    half_v: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(0.5))
    one_v: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.0))
    half_s: bf16 = 0.5
    one_s: bf16 = 1.0

    # ===================================================================
    # Stage 1: Conv1x1 + BN + SiLU  (in_channels -> neck_channels)
    # ===================================================================
    # Scalar conv1x1 accumulator with affine BN + LUT SiLU per output channel.
    # neck_channels may be < 16, so we use a scalar BN+SiLU path here.
    hw: i32 = 0
    while hw < spatial:
        oc: i32 = 0
        while oc < neck_channels:
            acc: bf16 = bf16(0.0)
            ic: i32 = 0
            while ic < in_channels:
                xv: bf16 = input[hw * in_channels + ic]
                wv: bf16 = conv1_w[oc * in_channels + ic]
                acc = acc + xv * wv
                ic = ic + 1
            bnw: bf16 = conv1_bn_w[oc]
            bnb: bf16 = conv1_bn_b[oc]
            y: bf16 = bnw * acc + bnb
            # SiLU(y) = y * 0.5 * (1 + tanh(0.5*y))  (LUT tanh)
            # Broadcast scalar y*0.5 to a 16-lane vector, run LUT tanh,
            # then extract lane 0 back to scalar.
            half_y_vec: aie_vector[bf16, 16] = broadcast(bf16, 16, y * half_s)
            t_vec: aie_vector[bf16, 16] = getTanhBf16(half_y_vec)
            t_s: bf16 = extract_elem(t_vec, 0)
            sig: bf16 = half_s * (one_s + t_s)
            silu: bf16 = y * sig
            conv1_output[hw * neck_channels + oc] = silu
            oc = oc + 1
        hw = hw + 1

    # ===================================================================
    # Stage 2-4: 3x MaxPool 5x5, stride=1, padding=2  on neck_channels
    # ===================================================================
    # With stride=1, pad=(k-1)/2 the spatial dims are preserved.
    out_h: i32 = (height + 2 * padding - kernel_size) // stride + 1
    out_w: i32 = (width + 2 * padding - kernel_size) // stride + 1
    neg_huge: bf16 = bf16(-1.0e30)  # below the dynamic range of any realistic activation

    # Pool 1: conv1_output -> pool1_output
    c1: i32 = 0
    while c1 < neck_channels:
        oh1: i32 = 0
        while oh1 < out_h:
            ow1: i32 = 0
            while ow1 < out_w:
                mx1: bf16 = neg_huge
                kh1: i32 = 0
                while kh1 < kernel_size:
                    kw1: i32 = 0
                    while kw1 < kernel_size:
                        ih1: i32 = oh1 * stride + kh1 - padding
                        iw1: i32 = ow1 * stride + kw1 - padding
                        if ih1 >= 0 and ih1 < height and iw1 >= 0 and iw1 < width:
                            v1: bf16 = conv1_output[(ih1 * width + iw1) * neck_channels + c1]
                            if v1 > mx1:
                                mx1 = v1
                        kw1 = kw1 + 1
                    kh1 = kh1 + 1
                pool1_output[(oh1 * out_w + ow1) * neck_channels + c1] = mx1
                ow1 = ow1 + 1
            oh1 = oh1 + 1
        c1 = c1 + 1

    # Pool 2: pool1_output -> pool2_output
    c2: i32 = 0
    while c2 < neck_channels:
        oh2: i32 = 0
        while oh2 < out_h:
            ow2: i32 = 0
            while ow2 < out_w:
                mx2: bf16 = neg_huge
                kh2: i32 = 0
                while kh2 < kernel_size:
                    kw2: i32 = 0
                    while kw2 < kernel_size:
                        ih2: i32 = oh2 * stride + kh2 - padding
                        iw2: i32 = ow2 * stride + kw2 - padding
                        if ih2 >= 0 and ih2 < height and iw2 >= 0 and iw2 < width:
                            v2: bf16 = pool1_output[(ih2 * width + iw2) * neck_channels + c2]
                            if v2 > mx2:
                                mx2 = v2
                        kw2 = kw2 + 1
                    kh2 = kh2 + 1
                pool2_output[(oh2 * out_w + ow2) * neck_channels + c2] = mx2
                ow2 = ow2 + 1
            oh2 = oh2 + 1
        c2 = c2 + 1

    # Pool 3: pool2_output -> pool3_output
    c3: i32 = 0
    while c3 < neck_channels:
        oh3: i32 = 0
        while oh3 < out_h:
            ow3: i32 = 0
            while ow3 < out_w:
                mx3: bf16 = neg_huge
                kh3: i32 = 0
                while kh3 < kernel_size:
                    kw3: i32 = 0
                    while kw3 < kernel_size:
                        ih3: i32 = oh3 * stride + kh3 - padding
                        iw3: i32 = ow3 * stride + kw3 - padding
                        if ih3 >= 0 and ih3 < height and iw3 >= 0 and iw3 < width:
                            v3: bf16 = pool2_output[(ih3 * width + iw3) * neck_channels + c3]
                            if v3 > mx3:
                                mx3 = v3
                        kw3 = kw3 + 1
                    kh3 = kh3 + 1
                pool3_output[(oh3 * out_w + ow3) * neck_channels + c3] = mx3
                ow3 = ow3 + 1
            oh3 = oh3 + 1
        c3 = c3 + 1

    # ===================================================================
    # Stage 5: 4-way Concat [f0, f1, f2, f3]  along channel axis
    # ===================================================================
    hw = 0
    while hw < spatial:
        out_base: i32 = hw * concat_channels
        # f0
        cc: i32 = 0
        while cc < neck_channels:
            concat_buffer[out_base + cc] = conv1_output[hw * neck_channels + cc]
            cc = cc + 1
        # f1
        cc = 0
        while cc < neck_channels:
            concat_buffer[out_base + neck_channels + cc] = pool1_output[hw * neck_channels + cc]
            cc = cc + 1
        # f2
        cc = 0
        while cc < neck_channels:
            concat_buffer[out_base + 2 * neck_channels + cc] = pool2_output[hw * neck_channels + cc]
            cc = cc + 1
        # f3
        cc = 0
        while cc < neck_channels:
            concat_buffer[out_base + 3 * neck_channels + cc] = pool3_output[hw * neck_channels + cc]
            cc = cc + 1
        hw = hw + 1

    # ===================================================================
    # Stage 6: Conv1x1 + BN + SiLU  (4*neck -> out_channels)
    # ===================================================================
    hw = 0
    while hw < spatial:
        oc: i32 = 0
        while oc < out_channels:
            acc: bf16 = bf16(0.0)
            ic: i32 = 0
            while ic < concat_channels:
                xv2: bf16 = concat_buffer[hw * concat_channels + ic]
                wv2: bf16 = conv5_w[oc * concat_channels + ic]
                acc = acc + xv2 * wv2
                ic = ic + 1
            bnw2: bf16 = conv5_bn_w[oc]
            bnb2: bf16 = conv5_bn_b[oc]
            y2: bf16 = bnw2 * acc + bnb2
            half_y_vec2: aie_vector[bf16, 16] = broadcast(bf16, 16, y2 * half_s)
            t_vec2: aie_vector[bf16, 16] = getTanhBf16(half_y_vec2)
            t_s2: bf16 = extract_elem(t_vec2, 0)
            sig2: bf16 = half_s * (one_s + t_s2)
            silu2: bf16 = y2 * sig2
            output[hw * out_channels + oc] = silu2
            oc = oc + 1
        hw = hw + 1

    event1()


# --- CLI --------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 SPPELAN layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", "-ic", type=int, default=16)
    parser.add_argument("--out-channels", "-oc", type=int, default=16)
    parser.add_argument(
        "--neck-channels", "-nc", type=int, default=None,
        help="Intermediate channels (default in_channels // 2)",
    )
    parser.add_argument("--kernel-size", "-k", type=int, default=5)
    parser.add_argument("--stride", "-s", type=int, default=1)
    parser.add_argument("--padding", "-p", type=int, default=2)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# --- MLIR / IRON construction ---------------------------------------


def build_mlir_module(device, height, width, in_channels, out_channels,
                      neck_channels, kernel_size, stride, padding):
    concat_channels = 4 * neck_channels

    input_size = height * width * in_channels
    conv1_size = height * width * neck_channels
    pool_size = height * width * neck_channels  # stride=1, pad=2 keeps spatial
    concat_size = height * width * concat_channels
    output_size = height * width * out_channels

    conv1_w_size = neck_channels * in_channels
    conv1_bn_size = 2 * neck_channels  # affine: [bn_w, bn_b]
    conv5_w_size = out_channels * concat_channels
    conv5_bn_size = 2 * out_channels
    total_weight_size = (
        conv1_w_size + conv1_bn_size + conv5_w_size + conv5_bn_size
    )

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    pool_ty = np.ndarray[(pool_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        sppelan_bf16_kernel,
        [
            input_ty,    # input
            weight_ty,   # weights_and_bn
            output_ty,   # output
            conv1_ty,    # conv1_output
            pool_ty,     # pool1_output
            pool_ty,     # pool2_output
            pool_ty,     # pool3_output
            concat_ty,   # concat_buffer
            np.int32,    # height
            np.int32,    # width
            np.int32,    # in_channels
            np.int32,    # out_channels
            np.int32,    # neck_channels
            np.int32,    # kernel_size
            np.int32,    # stride
            np.int32,    # padding
        ],
        extra_globals={"getTanhBf16": getTanhBf16},
    )

    of_in = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_wts = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_out = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    conv1_buf = Buffer(conv1_ty, name="conv1_output")
    pool1_buf = Buffer(pool_ty, name="pool1_output")
    pool2_buf = Buffer(pool_ty, name="pool2_output")
    pool3_buf = Buffer(pool_ty, name="pool3_output")
    concat_buf = Buffer(concat_ty, name="concat_buffer")

    def core_fn(of_in, of_wts, of_out, kernel,
                conv1_buf, pool1_buf, pool2_buf, pool3_buf, concat_buf):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(
            elem_in,
            elem_wts,
            elem_out,
            conv1_buf,
            pool1_buf,
            pool2_buf,
            pool3_buf,
            concat_buf,
            height,
            width,
            in_channels,
            out_channels,
            neck_channels,
            kernel_size,
            stride,
            padding,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_in.cons(),
            of_wts.cons(),
            of_out.prod(),
            kernel,
            conv1_buf,
            pool1_buf,
            pool2_buf,
            pool3_buf,
            concat_buf,
        ],
        stack_size=4096,
    )

    def sequence(I, W, O, of_in_prod, of_wts_prod, of_out_cons):
        of_in_prod.fill(I)
        of_wts_prod.fill(W)
        of_out_cons.drain(O, wait=True)

    rt = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_in.prod(), of_wts.prod(), of_out.cons()],
    )

    module = Program(device, rt, workers=[worker]).resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# --- Numpy reference ---------------------------------------------------


def _conv1x1_bn_silu_ref(x_hwc_f32, w_oc_ic_f32, bn_w_f32, bn_b_f32):
    """x: (H,W,IC) -> (H,W,OC), all fp32; BN already collapsed to affine."""
    H, W, IC = x_hwc_f32.shape
    OC = w_oc_ic_f32.shape[0]
    # conv1x1 == matmul on the channel axis: y[h,w,oc] = sum_ic x*w
    y = x_hwc_f32 @ w_oc_ic_f32.T  # (H,W,OC)
    # round per-output bf16
    y = y.astype(bfloat16).astype(np.float32)
    # BN affine: y = bn_w * y + bn_b
    y = y * bn_w_f32 + bn_b_f32
    y = y.astype(bfloat16).astype(np.float32)
    # SiLU = y * sigmoid(y); use math tanh which matches the LUT tanh closely
    sig = 0.5 * (1.0 + np.tanh(0.5 * y))
    out = y * sig
    return out.astype(bfloat16).astype(np.float32)


def _maxpool_ref(x_hwc_f32, k, s, p):
    H, W, C = x_hwc_f32.shape
    out_h = (H + 2 * p - k) // s + 1
    out_w = (W + 2 * p - k) // s + 1
    y = np.full((out_h, out_w, C), -3.4e38, dtype=np.float32)
    for oh in range(out_h):
        for ow in range(out_w):
            for kh in range(k):
                for kw in range(k):
                    ih = oh * s + kh - p
                    iw = ow * s + kw - p
                    if 0 <= ih < H and 0 <= iw < W:
                        y[oh, ow] = np.maximum(y[oh, ow], x_hwc_f32[ih, iw])
    return y.astype(bfloat16).astype(np.float32)


def numpy_reference(
    input_bf16, conv1_w_bf16, conv1_bn_w_bf16, conv1_bn_b_bf16,
    conv5_w_bf16, conv5_bn_w_bf16, conv5_bn_b_bf16,
    height, width, in_channels, out_channels, neck_channels,
    kernel_size, stride, padding,
):
    x_hwc = input_bf16.reshape(height, width, in_channels).astype(np.float32)
    cw1 = conv1_w_bf16.reshape(neck_channels, in_channels).astype(np.float32)
    cw5 = conv5_w_bf16.reshape(out_channels, 4 * neck_channels).astype(np.float32)

    f0 = _conv1x1_bn_silu_ref(
        x_hwc, cw1,
        conv1_bn_w_bf16.astype(np.float32),
        conv1_bn_b_bf16.astype(np.float32),
    )
    f1 = _maxpool_ref(f0, kernel_size, stride, padding)
    f2 = _maxpool_ref(f1, kernel_size, stride, padding)
    f3 = _maxpool_ref(f2, kernel_size, stride, padding)
    concat = np.concatenate([f0, f1, f2, f3], axis=-1)
    out = _conv1x1_bn_silu_ref(
        concat, cw5,
        conv5_bn_w_bf16.astype(np.float32),
        conv5_bn_b_bf16.astype(np.float32),
    )
    return out.reshape(-1)


# --- Compile & run ----------------------------------------------------


def run_with_xrt(xclbin_path, insts_path,
                 height, width, in_channels, out_channels, neck_channels,
                 kernel_size, stride, padding):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    input_size = height * width * in_channels
    output_size = height * width * out_channels
    concat_channels = 4 * neck_channels

    rng = np.random.default_rng(42)

    # Random bf16 tensors
    input_bf16 = rng.standard_normal(input_size).astype(np.float32).astype(bfloat16)

    # Pretend conv weights start as PyTorch-style gaussian / scale.
    # Use small magnitude to keep accumulator within bf16 range for 8x8 H,W.
    conv1_w_bf16 = (0.1 * rng.standard_normal(neck_channels * in_channels).astype(np.float32)).astype(bfloat16)
    conv5_w_bf16 = (0.1 * rng.standard_normal(out_channels * concat_channels).astype(np.float32)).astype(bfloat16)

    # Pretend PyTorch BN params (gamma, beta, mean, var)
    gamma1 = rng.uniform(0.5, 1.5, neck_channels).astype(np.float32)
    beta1 = rng.standard_normal(neck_channels).astype(np.float32) * 0.1
    mean1 = rng.standard_normal(neck_channels).astype(np.float32) * 0.1
    var1 = rng.uniform(0.5, 1.5, neck_channels).astype(np.float32)

    gamma5 = rng.uniform(0.5, 1.5, out_channels).astype(np.float32)
    beta5 = rng.standard_normal(out_channels).astype(np.float32) * 0.1
    mean5 = rng.standard_normal(out_channels).astype(np.float32) * 0.1
    var5 = rng.uniform(0.5, 1.5, out_channels).astype(np.float32)

    eps = 1e-3
    inv_std1 = 1.0 / np.sqrt(var1 + eps)
    inv_std5 = 1.0 / np.sqrt(var5 + eps)

    # Collapse PyTorch BN to affine y = w*x + b
    bn_w1_f32 = gamma1 * inv_std1
    bn_b1_f32 = beta1 - bn_w1_f32 * mean1
    bn_w5_f32 = gamma5 * inv_std5
    bn_b5_f32 = beta5 - bn_w5_f32 * mean5

    conv1_bn_w_bf16 = bn_w1_f32.astype(bfloat16)
    conv1_bn_b_bf16 = bn_b1_f32.astype(bfloat16)
    conv5_bn_w_bf16 = bn_w5_f32.astype(bfloat16)
    conv5_bn_b_bf16 = bn_b5_f32.astype(bfloat16)

    weights_packed_bf16 = np.concatenate([
        conv1_w_bf16,
        conv1_bn_w_bf16,
        conv1_bn_b_bf16,
        conv5_w_bf16,
        conv5_bn_w_bf16,
        conv5_bn_b_bf16,
    ])

    in_input = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    in_weights = iron.tensor(weights_packed_bf16.view(np.uint16), dtype=np.uint16)
    out_tensor = iron.zeros(output_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_input, in_weights, out_tensor])

    out_u16 = out_tensor.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = numpy_reference(
        input_bf16, conv1_w_bf16, conv1_bn_w_bf16, conv1_bn_b_bf16,
        conv5_w_bf16, conv5_bn_w_bf16, conv5_bn_b_bf16,
        height, width, in_channels, out_channels, neck_channels,
        kernel_size, stride, padding,
    )
    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()
    neck_channels = args.neck_channels or (args.in_channels // 2)

    try:
        print(
            f"[1/3] Building IRON program "
            f"(H={args.height}, W={args.width}, "
            f"IC={args.in_channels}, OC={args.out_channels}, NC={neck_channels})"
        )
        module = build_mlir_module(
            device, args.height, args.width,
            args.in_channels, args.out_channels, neck_channels,
            args.kernel_size, args.stride, args.padding,
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
            args.height, args.width,
            args.in_channels, args.out_channels, neck_channels,
            args.kernel_size, args.stride, args.padding,
        )
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        # SiLU uses a LUT tanh approximation in two fused conv blocks;
        # allow a moderate tolerance similar to batchnorm_silu.
        rtol = 5e-2
        atol = 1e-1
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
