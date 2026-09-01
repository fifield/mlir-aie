#!/usr/bin/env python3
# repncsp_elan_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 16 --out-channels 16 --part-channels 16 --process-channels 8 --work-dir ./repncsp_elan_pythoc_small_build | FileCheck %s
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 32 --out-channels 32 --part-channels 32 --process-channels 16 --work-dir ./repncsp_elan_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 RepNCSPELAN layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/repncsp_elan/{aie2.py,repncsp_elan_bf16.cc}
that replaces the external C++ kernel with inline PythoC kernels.

The RepNCSPELAN block (most complex layer in MDV6) is composed of:
    Input -> Conv1 (1x1+BN+SiLU) -> split into [x1, x2]
                                         |
                                      x2 -> RepNCSP -> Conv3x3 -> x3
                                                            |
                                                        x3 -> RepNCSP -> Conv3x3 -> x4
                                         |              |             |
                              Concat [x1, x2, x3, x4] (4-way)
                                         |
                                    Conv4 (1x1+BN+SiLU) -> Output

This port keeps the same scalar arithmetic as the C++ implementation
(`fast_sqrt`/`fast_sigmoid` analogues realized through the AIE2P `invsqrt`
intrinsic and a small inlined sigmoid). Inner accumulation is done in
float32; values are rounded back to bfloat16 at each tensor store.

Layout: HWC (channels innermost), bfloat16 carried as uint16 in
ObjectFifos and numpy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Buffer, Program, Runtime, Worker
import aie.iron as iron
from aie.utils.compile import compile_mlir_module
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import ptr, i32, f32, bf16
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

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "repncsp_elan_pythoc_build"


# =============================================================================
# PythoC kernel helpers
#
# All helpers are decorated with @aie_kernel and passed via PythocKernel(
# helpers=[...]). They are compiled into the same LLVM module as the main
# kernel so the main kernel can call them by name.
# =============================================================================


@aie_kernel
def conv1x1_bn_helper(
    inp: ptr[bf16, True],
    weights: ptr[bf16, True],
    bn_w: ptr[bf16, True],
    bn_b: ptr[bf16, True],
    bn_m: ptr[bf16, True],
    bn_v: ptr[bf16, True],
    out: ptr[bf16, True],
    H: i32,
    W: i32,
    IC: i32,
    OC: i32,
    do_silu: i32,
):
    """1x1 conv + per-channel BatchNorm, optional fast-sigmoid SiLU.

    do_silu == 1 applies fast_sigmoid SiLU (`x * (0.5 + x/(2*(1+|x|)))`).
    """
    bn_eps: f32 = 0.001
    one: f32 = 1.0
    two: f32 = 2.0
    half: f32 = 0.5
    oc: i32 = 0
    while oc < OC:
        gamma: f32 = f32(bn_w[oc])
        beta: f32 = f32(bn_b[oc])
        mean: f32 = f32(bn_m[oc])
        var: f32 = f32(bn_v[oc])
        inv_std: f32 = invsqrt(var + bn_eps)
        h: i32 = 0
        while h < H:
            w: i32 = 0
            while w < W:
                s: f32 = 0.0
                ic: i32 = 0
                while ic < IC:
                    in_idx: i32 = (h * W + w) * IC + ic
                    wt_idx: i32 = oc * IC + ic
                    s = s + f32(inp[in_idx]) * f32(weights[wt_idx])
                    ic = ic + 1
                bn_out: f32 = gamma * (s - mean) * inv_std + beta
                if do_silu == 1:
                    ax: f32 = bn_out
                    if ax < 0.0:
                        ax = 0.0 - ax
                    sigm: f32 = half + bn_out / (two * (one + ax))
                    bn_out = bn_out * sigm
                o_idx: i32 = (h * W + w) * OC + oc
                out[o_idx] = bf16(bn_out)
                w = w + 1
            h = h + 1
        oc = oc + 1


@aie_kernel
def conv3x3_bn_helper(
    inp: ptr[bf16, True],
    weights: ptr[bf16, True],
    bn_w: ptr[bf16, True],
    bn_b: ptr[bf16, True],
    bn_m: ptr[bf16, True],
    bn_v: ptr[bf16, True],
    out: ptr[bf16, True],
    H: i32,
    W: i32,
    IC: i32,
    OC: i32,
    padding: i32,
    do_silu: i32,
):
    """3x3 conv + per-channel BatchNorm, optional fast-sigmoid SiLU.

    HWC layout, zero-padding. do_silu == 1 applies fast_sigmoid SiLU.
    """
    bn_eps: f32 = 0.001
    one: f32 = 1.0
    two: f32 = 2.0
    half: f32 = 0.5
    oc: i32 = 0
    while oc < OC:
        gamma: f32 = f32(bn_w[oc])
        beta: f32 = f32(bn_b[oc])
        mean: f32 = f32(bn_m[oc])
        var: f32 = f32(bn_v[oc])
        inv_std: f32 = invsqrt(var + bn_eps)
        h: i32 = 0
        while h < H:
            w: i32 = 0
            while w < W:
                s: f32 = 0.0
                ic: i32 = 0
                while ic < IC:
                    kh: i32 = 0
                    while kh < 3:
                        kw: i32 = 0
                        while kw < 3:
                            ih: i32 = h + kh - padding
                            iw: i32 = w + kw - padding
                            if ih >= 0:
                                if ih < H:
                                    if iw >= 0:
                                        if iw < W:
                                            in_idx: i32 = (ih * W + iw) * IC + ic
                                            wt_idx: i32 = (
                                                (oc * IC + ic) * 3 + kh
                                            ) * 3 + kw
                                            s = s + f32(inp[in_idx]) * f32(
                                                weights[wt_idx]
                                            )
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1
                bn_out: f32 = gamma * (s - mean) * inv_std + beta
                if do_silu == 1:
                    ax: f32 = bn_out
                    if ax < 0.0:
                        ax = 0.0 - ax
                    sigm: f32 = half + bn_out / (two * (one + ax))
                    bn_out = bn_out * sigm
                o_idx: i32 = (h * W + w) * OC + oc
                out[o_idx] = bf16(bn_out)
                w = w + 1
            h = h + 1
        oc = oc + 1


@aie_kernel
def concat_2way_helper(
    x1: ptr[bf16, True],
    x2: ptr[bf16, True],
    out: ptr[bf16, True],
    H: i32,
    W: i32,
    C1: i32,
    C2: i32,
):
    """Concatenate two HWC tensors along the channel axis."""
    total: i32 = C1 + C2
    h: i32 = 0
    while h < H:
        w: i32 = 0
        while w < W:
            sp: i32 = h * W + w
            off: i32 = sp * total
            c: i32 = 0
            while c < C1:
                out[off + c] = x1[sp * C1 + c]
                c = c + 1
            c = 0
            while c < C2:
                out[off + C1 + c] = x2[sp * C2 + c]
                c = c + 1
            w = w + 1
        h = h + 1


@aie_kernel
def concat_4way_helper(
    x1: ptr[bf16, True],
    x2: ptr[bf16, True],
    x3: ptr[bf16, True],
    x4: ptr[bf16, True],
    out: ptr[bf16, True],
    H: i32,
    W: i32,
    C1: i32,
    C2: i32,
    C3: i32,
    C4: i32,
):
    """Concatenate four HWC tensors along the channel axis."""
    total: i32 = C1 + C2 + C3 + C4
    h: i32 = 0
    while h < H:
        w: i32 = 0
        while w < W:
            sp: i32 = h * W + w
            off: i32 = sp * total
            c: i32 = 0
            while c < C1:
                out[off + c] = x1[sp * C1 + c]
                c = c + 1
            c = 0
            while c < C2:
                out[off + C1 + c] = x2[sp * C2 + c]
                c = c + 1
            c = 0
            while c < C3:
                out[off + C1 + C2 + c] = x3[sp * C3 + c]
                c = c + 1
            c = 0
            while c < C4:
                out[off + C1 + C2 + C3 + c] = x4[sp * C4 + c]
                c = c + 1
            w = w + 1
        h = h + 1


@aie_kernel
def add_silu_helper(
    a: ptr[bf16, True], b: ptr[bf16, True], out: ptr[bf16, True], n: i32
):
    """out[i] = silu(a[i] + b[i]) using the LUT-based tanh sigmoid.

    sigmoid(z) ~= 0.5 * (1 + tanh(0.5 * z)); SiLU = z * sigmoid(z).
    Processes 16 bf16 elements per iteration.
    """
    vec: i32 = 16
    half_v: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(0.5))
    one_v: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.0))
    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pO: ptr[bf16] = out
    i: i32 = 0
    while i < n:
        va: aie_vector[bf16, 16] = load_v(pA, 16)
        vb: aie_vector[bf16, 16] = load_v(pB, 16)
        s: aie_vector[bf16, 16] = vector_add(va, vb)
        s_half: aie_vector[bf16, 16] = vector_mul(s, half_v)
        t: aie_vector[bf16, 16] = getTanhBf16(s_half)
        one_plus: aie_vector[bf16, 16] = vector_add(one_v, t)
        sig: aie_vector[bf16, 16] = vector_mul(half_v, one_plus)
        res: aie_vector[bf16, 16] = vector_mul(s, sig)
        store_v(pO, res)
        pA = pA + vec
        pB = pB + vec
        pO = pO + vec
        i = i + vec


@aie_kernel
def add_residual_helper(
    a: ptr[bf16, True], b: ptr[bf16, True], out: ptr[bf16, True], n: i32
):
    """out[i] = a[i] + b[i] (no activation). 16-wide bf16 vectors."""
    vec: i32 = 16
    pA: ptr[bf16] = a
    pB: ptr[bf16] = b
    pO: ptr[bf16] = out
    i: i32 = 0
    while i < n:
        va: aie_vector[bf16, 16] = load_v(pA, 16)
        vb: aie_vector[bf16, 16] = load_v(pB, 16)
        store_v(pO, vector_add(va, vb))
        pA = pA + vec
        pB = pB + vec
        pO = pO + vec
        i = i + vec


@aie_kernel
def copy_helper(src: ptr[bf16, True], dst: ptr[bf16, True], n: i32):
    """dst[i] = src[i]. 16-wide bf16 vector copy."""
    vec: i32 = 16
    pS: ptr[bf16] = src
    pD: ptr[bf16] = dst
    i: i32 = 0
    while i < n:
        v: aie_vector[bf16, 16] = load_v(pS, 16)
        store_v(pD, v)
        pS = pS + vec
        pD = pD + vec
        i = i + vec


# =============================================================================
# Main RepNCSPELAN kernel
#
# Mirrors repncsp_elan_bf16_scalar from repncsp_elan_bf16.cc. Buffers
# and weights are passed in as raw pointers; the kernel does its own
# pointer arithmetic on the packed weight blob.
# =============================================================================


@aie_kernel
def repncsp_elan_bf16_kernel(
    inp: ptr[bf16, True],
    weights: ptr[bf16, True],
    out: ptr[bf16, True],
    # Main stage scratch buffers
    conv1_output: ptr[bf16, True],
    x3_repncsp_out: ptr[bf16, True],
    x3_conv_out: ptr[bf16, True],
    x4_repncsp_out: ptr[bf16, True],
    x4_conv_out: ptr[bf16, True],
    concat_buffer: ptr[bf16, True],
    # RepNCSP shared scratch (rn1 and rn2 reuse the same physical buffers)
    rn_conv1_out: ptr[bf16, True],
    rn_bottleneck_out: ptr[bf16, True],
    rn_conv2_out: ptr[bf16, True],
    rn_concat: ptr[bf16, True],
    rn_bn_input_copy: ptr[bf16, True],
    rn_bn_temp1: ptr[bf16, True],
    rn_bn_temp2: ptr[bf16, True],
    rn_bn_temp3: ptr[bf16, True],
    rn_bn_temp4: ptr[bf16, True],
    # Dimensions
    H: i32,
    W: i32,
    IC: i32,
    OC: i32,
    PC: i32,  # part_channels
    PR: i32,  # process_channels
):
    event0()

    half_part: i32 = PC // 2
    concat_channels: i32 = PC + 2 * PR
    rn_neck: i32 = PR // 2  # both rn1 and rn2 use the same neck size
    rn_neck_size: i32 = H * W * rn_neck

    # -----------------------------------------------------------------
    # Weight pointer extraction (matches repncsp_elan_bf16.cc layout)
    # -----------------------------------------------------------------
    p: ptr[bf16] = weights

    # Conv1 (1x1): IC -> PC
    conv1_w: ptr[bf16] = p
    p = p + PC * IC
    conv1_bn_w: ptr[bf16] = p
    p = p + PC
    conv1_bn_b: ptr[bf16] = p
    p = p + PC
    conv1_bn_m: ptr[bf16] = p
    p = p + PC
    conv1_bn_v: ptr[bf16] = p
    p = p + PC

    # RepNCSP #1 weights ----
    rn1_conv1_w: ptr[bf16] = p
    p = p + rn_neck * half_part
    rn1_conv1_bn_w: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv1_bn_b: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv1_bn_m: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv1_bn_v: ptr[bf16] = p
    p = p + rn_neck

    rn1_bn_conv3x3_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck * 9
    rn1_bn_bn3x3_w: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn3x3_b: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn3x3_m: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn3x3_v: ptr[bf16] = p
    p = p + rn_neck

    rn1_bn_conv1x1_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck
    rn1_bn_bn1x1_w: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn1x1_b: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn1x1_m: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn1x1_v: ptr[bf16] = p
    p = p + rn_neck

    rn1_bn_conv2_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck * 9
    rn1_bn_bn2_w: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn2_b: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn2_m: ptr[bf16] = p
    p = p + rn_neck
    rn1_bn_bn2_v: ptr[bf16] = p
    p = p + rn_neck

    rn1_conv2_w: ptr[bf16] = p
    p = p + rn_neck * half_part
    rn1_conv2_bn_w: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv2_bn_b: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv2_bn_m: ptr[bf16] = p
    p = p + rn_neck
    rn1_conv2_bn_v: ptr[bf16] = p
    p = p + rn_neck

    rn1_conv3_w: ptr[bf16] = p
    p = p + PR * 2 * rn_neck
    rn1_conv3_bn_w: ptr[bf16] = p
    p = p + PR
    rn1_conv3_bn_b: ptr[bf16] = p
    p = p + PR
    rn1_conv3_bn_m: ptr[bf16] = p
    p = p + PR
    rn1_conv3_bn_v: ptr[bf16] = p
    p = p + PR

    # Conv3x3 #1 ----
    conv3x3_1_w: ptr[bf16] = p
    p = p + PR * PR * 9
    conv3x3_1_bn_w: ptr[bf16] = p
    p = p + PR
    conv3x3_1_bn_b: ptr[bf16] = p
    p = p + PR
    conv3x3_1_bn_m: ptr[bf16] = p
    p = p + PR
    conv3x3_1_bn_v: ptr[bf16] = p
    p = p + PR

    # RepNCSP #2 weights ----
    rn2_conv1_w: ptr[bf16] = p
    p = p + rn_neck * PR
    rn2_conv1_bn_w: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv1_bn_b: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv1_bn_m: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv1_bn_v: ptr[bf16] = p
    p = p + rn_neck

    rn2_bn_conv3x3_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck * 9
    rn2_bn_bn3x3_w: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn3x3_b: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn3x3_m: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn3x3_v: ptr[bf16] = p
    p = p + rn_neck

    rn2_bn_conv1x1_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck
    rn2_bn_bn1x1_w: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn1x1_b: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn1x1_m: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn1x1_v: ptr[bf16] = p
    p = p + rn_neck

    rn2_bn_conv2_w: ptr[bf16] = p
    p = p + rn_neck * rn_neck * 9
    rn2_bn_bn2_w: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn2_b: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn2_m: ptr[bf16] = p
    p = p + rn_neck
    rn2_bn_bn2_v: ptr[bf16] = p
    p = p + rn_neck

    rn2_conv2_w: ptr[bf16] = p
    p = p + rn_neck * PR
    rn2_conv2_bn_w: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv2_bn_b: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv2_bn_m: ptr[bf16] = p
    p = p + rn_neck
    rn2_conv2_bn_v: ptr[bf16] = p
    p = p + rn_neck

    rn2_conv3_w: ptr[bf16] = p
    p = p + PR * 2 * rn_neck
    rn2_conv3_bn_w: ptr[bf16] = p
    p = p + PR
    rn2_conv3_bn_b: ptr[bf16] = p
    p = p + PR
    rn2_conv3_bn_m: ptr[bf16] = p
    p = p + PR
    rn2_conv3_bn_v: ptr[bf16] = p
    p = p + PR

    # Conv3x3 #2 ----
    conv3x3_2_w: ptr[bf16] = p
    p = p + PR * PR * 9
    conv3x3_2_bn_w: ptr[bf16] = p
    p = p + PR
    conv3x3_2_bn_b: ptr[bf16] = p
    p = p + PR
    conv3x3_2_bn_m: ptr[bf16] = p
    p = p + PR
    conv3x3_2_bn_v: ptr[bf16] = p
    p = p + PR

    # Conv4 (1x1): concat_channels -> OC ----
    conv4_w: ptr[bf16] = p
    p = p + OC * concat_channels
    conv4_bn_w: ptr[bf16] = p
    p = p + OC
    conv4_bn_b: ptr[bf16] = p
    p = p + OC
    conv4_bn_m: ptr[bf16] = p
    p = p + OC
    conv4_bn_v: ptr[bf16] = p
    p = p + OC

    # -----------------------------------------------------------------
    # Stage 1: Conv1 (1x1+BN+SiLU) -> split into [x1, x2]
    # -----------------------------------------------------------------
    conv1x1_bn_helper(
        inp, conv1_w, conv1_bn_w, conv1_bn_b, conv1_bn_m, conv1_bn_v,
        conv1_output, H, W, IC, PC, 1,
    )
    x1: ptr[bf16] = conv1_output
    x2: ptr[bf16] = conv1_output + H * W * half_part

    # -----------------------------------------------------------------
    # Stage 2: RepNCSP #1 (x2 -> x3_repncsp_out)
    # -----------------------------------------------------------------
    # RepNCSP #1 - Conv1 (1x1+BN+SiLU)
    conv1x1_bn_helper(
        x2, rn1_conv1_w, rn1_conv1_bn_w, rn1_conv1_bn_b,
        rn1_conv1_bn_m, rn1_conv1_bn_v,
        rn_conv1_out, H, W, half_part, rn_neck, 1,
    )

    # Save residual input
    copy_helper(rn_conv1_out, rn_bn_input_copy, rn_neck_size)

    # Bottleneck - RepConv branch 1: Conv3x3 + BN (no SiLU)
    conv3x3_bn_helper(
        rn_conv1_out, rn1_bn_conv3x3_w, rn1_bn_bn3x3_w, rn1_bn_bn3x3_b,
        rn1_bn_bn3x3_m, rn1_bn_bn3x3_v,
        rn_bn_temp1, H, W, rn_neck, rn_neck, 1, 0,
    )

    # Bottleneck - RepConv branch 2: Conv1x1 + BN (no SiLU)
    conv1x1_bn_helper(
        rn_conv1_out, rn1_bn_conv1x1_w, rn1_bn_bn1x1_w, rn1_bn_bn1x1_b,
        rn1_bn_bn1x1_m, rn1_bn_bn1x1_v,
        rn_bn_temp2, H, W, rn_neck, rn_neck, 0,
    )

    # Bottleneck - Add + SiLU (complete RepConv)
    add_silu_helper(rn_bn_temp1, rn_bn_temp2, rn_bn_temp3, rn_neck_size)

    # Bottleneck - Conv2 (3x3) + BN + SiLU
    conv3x3_bn_helper(
        rn_bn_temp3, rn1_bn_conv2_w, rn1_bn_bn2_w, rn1_bn_bn2_b,
        rn1_bn_bn2_m, rn1_bn_bn2_v,
        rn_bn_temp4, H, W, rn_neck, rn_neck, 1, 1,
    )

    # Bottleneck - Residual add (no activation)
    add_residual_helper(
        rn_bn_input_copy, rn_bn_temp4, rn_bottleneck_out, rn_neck_size
    )

    # RepNCSP #1 - Conv2 (1x1+BN+SiLU) bypass
    conv1x1_bn_helper(
        x2, rn1_conv2_w, rn1_conv2_bn_w, rn1_conv2_bn_b,
        rn1_conv2_bn_m, rn1_conv2_bn_v,
        rn_conv2_out, H, W, half_part, rn_neck, 1,
    )

    # Concat [bottleneck_out, conv2_out]
    concat_2way_helper(
        rn_bottleneck_out, rn_conv2_out, rn_concat, H, W, rn_neck, rn_neck
    )

    # Conv3 (1x1+BN+SiLU) merge -> x3_repncsp_out
    conv1x1_bn_helper(
        rn_concat, rn1_conv3_w, rn1_conv3_bn_w, rn1_conv3_bn_b,
        rn1_conv3_bn_m, rn1_conv3_bn_v,
        x3_repncsp_out, H, W, 2 * rn_neck, PR, 1,
    )

    # -----------------------------------------------------------------
    # Stage 3: Conv3x3 #1 (x3_repncsp_out -> x3_conv_out)
    # -----------------------------------------------------------------
    conv3x3_bn_helper(
        x3_repncsp_out, conv3x3_1_w, conv3x3_1_bn_w, conv3x3_1_bn_b,
        conv3x3_1_bn_m, conv3x3_1_bn_v,
        x3_conv_out, H, W, PR, PR, 1, 1,
    )

    # -----------------------------------------------------------------
    # Stage 4: RepNCSP #2 (x3_conv_out -> x4_repncsp_out)
    # -----------------------------------------------------------------
    conv1x1_bn_helper(
        x3_conv_out, rn2_conv1_w, rn2_conv1_bn_w, rn2_conv1_bn_b,
        rn2_conv1_bn_m, rn2_conv1_bn_v,
        rn_conv1_out, H, W, PR, rn_neck, 1,
    )

    copy_helper(rn_conv1_out, rn_bn_input_copy, rn_neck_size)

    conv3x3_bn_helper(
        rn_conv1_out, rn2_bn_conv3x3_w, rn2_bn_bn3x3_w, rn2_bn_bn3x3_b,
        rn2_bn_bn3x3_m, rn2_bn_bn3x3_v,
        rn_bn_temp1, H, W, rn_neck, rn_neck, 1, 0,
    )

    conv1x1_bn_helper(
        rn_conv1_out, rn2_bn_conv1x1_w, rn2_bn_bn1x1_w, rn2_bn_bn1x1_b,
        rn2_bn_bn1x1_m, rn2_bn_bn1x1_v,
        rn_bn_temp2, H, W, rn_neck, rn_neck, 0,
    )

    add_silu_helper(rn_bn_temp1, rn_bn_temp2, rn_bn_temp3, rn_neck_size)

    conv3x3_bn_helper(
        rn_bn_temp3, rn2_bn_conv2_w, rn2_bn_bn2_w, rn2_bn_bn2_b,
        rn2_bn_bn2_m, rn2_bn_bn2_v,
        rn_bn_temp4, H, W, rn_neck, rn_neck, 1, 1,
    )

    add_residual_helper(
        rn_bn_input_copy, rn_bn_temp4, rn_bottleneck_out, rn_neck_size
    )

    conv1x1_bn_helper(
        x3_conv_out, rn2_conv2_w, rn2_conv2_bn_w, rn2_conv2_bn_b,
        rn2_conv2_bn_m, rn2_conv2_bn_v,
        rn_conv2_out, H, W, PR, rn_neck, 1,
    )

    concat_2way_helper(
        rn_bottleneck_out, rn_conv2_out, rn_concat, H, W, rn_neck, rn_neck
    )

    conv1x1_bn_helper(
        rn_concat, rn2_conv3_w, rn2_conv3_bn_w, rn2_conv3_bn_b,
        rn2_conv3_bn_m, rn2_conv3_bn_v,
        x4_repncsp_out, H, W, 2 * rn_neck, PR, 1,
    )

    # -----------------------------------------------------------------
    # Stage 5: Conv3x3 #2 (x4_repncsp_out -> x4_conv_out)
    # -----------------------------------------------------------------
    conv3x3_bn_helper(
        x4_repncsp_out, conv3x3_2_w, conv3x3_2_bn_w, conv3x3_2_bn_b,
        conv3x3_2_bn_m, conv3x3_2_bn_v,
        x4_conv_out, H, W, PR, PR, 1, 1,
    )

    # -----------------------------------------------------------------
    # Stage 6: 4-way concat [x1, x2, x3_conv_out, x4_conv_out]
    # -----------------------------------------------------------------
    concat_4way_helper(
        x1, x2, x3_conv_out, x4_conv_out, concat_buffer,
        H, W, half_part, half_part, PR, PR,
    )

    # -----------------------------------------------------------------
    # Stage 7: Conv4 (1x1+BN+SiLU) -> output
    # -----------------------------------------------------------------
    conv1x1_bn_helper(
        concat_buffer, conv4_w, conv4_bn_w, conv4_bn_b,
        conv4_bn_m, conv4_bn_v,
        out, H, W, concat_channels, OC, 1,
    )

    event1()


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 RepNCSPELAN layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--part-channels", type=int, default=32)
    parser.add_argument(
        "--process-channels", type=int, default=None,
        help="Default: part-channels // 2",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# =============================================================================
# MLIR / IRON construction
# =============================================================================


def compute_weight_size(in_c, out_c, part_c, proc_c):
    half_part = part_c // 2
    rn_neck = proc_c // 2
    concat_channels = part_c + 2 * proc_c

    # Conv1
    sz = part_c * in_c + 4 * part_c
    # RepNCSP #1
    sz += rn_neck * half_part + 4 * rn_neck  # rn1.conv1
    sz += rn_neck * rn_neck * 9 + 4 * rn_neck  # rn1 bn.conv3x3
    sz += rn_neck * rn_neck + 4 * rn_neck  # rn1 bn.conv1x1
    sz += rn_neck * rn_neck * 9 + 4 * rn_neck  # rn1 bn.conv2 (3x3)
    sz += rn_neck * half_part + 4 * rn_neck  # rn1.conv2 (1x1)
    sz += proc_c * 2 * rn_neck + 4 * proc_c  # rn1.conv3 (1x1)
    # Conv3x3 #1
    sz += proc_c * proc_c * 9 + 4 * proc_c
    # RepNCSP #2 (in_c=proc_c)
    sz += rn_neck * proc_c + 4 * rn_neck
    sz += rn_neck * rn_neck * 9 + 4 * rn_neck
    sz += rn_neck * rn_neck + 4 * rn_neck
    sz += rn_neck * rn_neck * 9 + 4 * rn_neck
    sz += rn_neck * proc_c + 4 * rn_neck
    sz += proc_c * 2 * rn_neck + 4 * proc_c
    # Conv3x3 #2
    sz += proc_c * proc_c * 9 + 4 * proc_c
    # Conv4
    sz += out_c * concat_channels + 4 * out_c
    return sz


def build_mlir_module(device, height, width, in_c, out_c, part_c, proc_c):
    half_part = part_c // 2
    rn_neck = proc_c // 2
    concat_channels = part_c + 2 * proc_c

    input_size = height * width * in_c
    output_size = height * width * out_c
    conv1_size = height * width * part_c
    repncsp_size = height * width * proc_c
    concat_size = height * width * concat_channels
    rn_neck_size = height * width * rn_neck
    rn_concat_size = height * width * 2 * rn_neck

    total_weight_size = compute_weight_size(in_c, out_c, part_c, proc_c)

    # IRON / numpy types (bf16 carried as uint16)
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(total_weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    repncsp_ty = np.ndarray[(repncsp_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]
    rn_neck_ty = np.ndarray[(rn_neck_size,), np.dtype[np.uint16]]
    rn_concat_ty = np.ndarray[(rn_concat_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        repncsp_elan_bf16_kernel,
        [
            input_ty,      # inp
            weight_ty,     # weights
            output_ty,     # out
            conv1_ty,      # conv1_output
            repncsp_ty,    # x3_repncsp_out
            repncsp_ty,    # x3_conv_out
            repncsp_ty,    # x4_repncsp_out
            repncsp_ty,    # x4_conv_out
            concat_ty,     # concat_buffer
            rn_neck_ty,    # rn_conv1_out
            rn_neck_ty,    # rn_bottleneck_out
            rn_neck_ty,    # rn_conv2_out
            rn_concat_ty,  # rn_concat
            rn_neck_ty,    # rn_bn_input_copy
            rn_neck_ty,    # rn_bn_temp1
            rn_neck_ty,    # rn_bn_temp2
            rn_neck_ty,    # rn_bn_temp3
            rn_neck_ty,    # rn_bn_temp4
            np.int32,      # H
            np.int32,      # W
            np.int32,      # IC
            np.int32,      # OC
            np.int32,      # PC
            np.int32,      # PR
        ],
        extra_globals={
            "invsqrt": invsqrt,
            "getTanhBf16": getTanhBf16,
        },
        helpers=[
            conv1x1_bn_helper,
            conv3x3_bn_helper,
            concat_2way_helper,
            concat_4way_helper,
            add_silu_helper,
            add_residual_helper,
            copy_helper,
        ],
    )

    of_in = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_wts = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_out = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Local scratch buffers. rn1 and rn2 share the same physical buffers
    # (sequential execution, same neck size).
    conv1_buf = Buffer(conv1_ty, name="conv1_output")
    x3_repncsp_buf = Buffer(repncsp_ty, name="x3x4_repncsp_out")
    x3_conv_buf = Buffer(repncsp_ty, name="x3x4_conv_out")
    concat_buf = Buffer(concat_ty, name="concat_buffer")
    rn_conv1_buf = Buffer(rn_neck_ty, name="rn_conv1_out")
    rn_bottleneck_buf = Buffer(rn_neck_ty, name="rn_bottleneck_out")
    rn_conv2_buf = Buffer(rn_neck_ty, name="rn_conv2_out")
    rn_concat_buf = Buffer(rn_concat_ty, name="rn_concat")
    rn_in_copy_buf = Buffer(rn_neck_ty, name="rn_bn_input_copy")
    rn_t1_buf = Buffer(rn_neck_ty, name="rn_bn_temp1")
    rn_t2_buf = Buffer(rn_neck_ty, name="rn_bn_temp2")
    rn_t3_buf = Buffer(rn_neck_ty, name="rn_bn_temp3")
    rn_t4_buf = Buffer(rn_neck_ty, name="rn_bn_temp4")

    def core_fn(
        of_in, of_wts, of_out, kernel,
        conv1_b, x3_rn_b, x3_cv_b, concat_b,
        rn_c1, rn_bn, rn_c2, rn_cat,
        rn_ic, rn_t1, rn_t2, rn_t3, rn_t4,
    ):
        elem_in = of_in.acquire(1)
        elem_w = of_wts.acquire(1)
        elem_o = of_out.acquire(1)
        # x3 and x4 share buffers (sequential execution).
        kernel(
            elem_in, elem_w, elem_o,
            conv1_b,
            x3_rn_b,        # x3_repncsp_out
            x3_cv_b,        # x3_conv_out
            x3_rn_b,        # x4_repncsp_out (shared)
            x3_cv_b,        # x4_conv_out    (shared)
            concat_b,
            rn_c1, rn_bn, rn_c2, rn_cat,
            rn_ic, rn_t1, rn_t2, rn_t3, rn_t4,
            height, width, in_c, out_c, part_c, proc_c,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_in.cons(), of_wts.cons(), of_out.prod(), kernel,
            conv1_buf, x3_repncsp_buf, x3_conv_buf, concat_buf,
            rn_conv1_buf, rn_bottleneck_buf, rn_conv2_buf, rn_concat_buf,
            rn_in_copy_buf, rn_t1_buf, rn_t2_buf, rn_t3_buf, rn_t4_buf,
        ],
        stack_size=4096,
    )

    def sequence(I, W, O, of_in_prod, of_wts_prod, of_out_cons):
        of_in_prod.fill(I)
        of_wts_prod.fill(W)
        of_out_cons.drain(O, wait=True)

    runtime = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_in.prod(), of_wts.prod(), of_out.cons()],
    )

    program = Program(device, runtime, workers=[worker])
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# =============================================================================
# Reference implementation (numpy, matching the on-device scalar math)
# =============================================================================


def _fast_sigmoid(x):
    """Approximation used by conv1x1/conv3x3 bn_silu kernels (scalar)."""
    return 0.5 + x / (2.0 * (1.0 + np.abs(x)))


def _tanh_sigmoid(x):
    """Approximation used by add_silu_helper (vector tanh path)."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _silu(x):
    """SiLU used by conv kernels (fast_sigmoid path)."""
    return x * _fast_sigmoid(x)


def _silu_tanh(x):
    """SiLU used by add_silu_helper (LUT-tanh sigmoid path)."""
    return x * _tanh_sigmoid(x)


def _conv1x1_bn_silu(inp_hwc, w, bn_w, bn_b, bn_m, bn_v, H, W, IC, OC, silu=True):
    eps = 1e-3
    inv_std = 1.0 / np.sqrt(bn_v + eps)
    out = np.zeros((H, W, OC), dtype=np.float32)
    # w shape: (OC, IC) (weight_idx = oc*IC + ic)
    w_mat = w.reshape(OC, IC).astype(np.float32)
    x = inp_hwc.reshape(H, W, IC).astype(np.float32)
    # s[h,w,oc] = sum_ic x[h,w,ic] * w_mat[oc,ic]
    s = np.einsum("hwi,oi->hwo", x, w_mat)
    bn_out = bn_w * (s - bn_m) * inv_std + bn_b
    bn_out = bn_out.astype(bfloat16).astype(np.float32)
    if silu:
        bn_out = _silu(bn_out)
        bn_out = bn_out.astype(bfloat16).astype(np.float32)
    return bn_out


def _conv3x3_bn(inp_hwc, w, bn_w, bn_b, bn_m, bn_v, H, W, IC, OC, padding, silu):
    eps = 1e-3
    inv_std = 1.0 / np.sqrt(bn_v + eps)
    x = inp_hwc.reshape(H, W, IC).astype(np.float32)
    w_arr = w.reshape(OC, IC, 3, 3).astype(np.float32)
    out = np.zeros((H, W, OC), dtype=np.float32)
    for oc in range(OC):
        gamma = float(bn_w[oc]); beta = float(bn_b[oc])
        mean = float(bn_m[oc]); std_inv = float(inv_std[oc])
        for h in range(H):
            for w_ in range(W):
                s = 0.0
                for ic in range(IC):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - padding
                            iw = w_ + kw - padding
                            if 0 <= ih < H and 0 <= iw < W:
                                s += x[ih, iw, ic] * w_arr[oc, ic, kh, kw]
                bn_out = gamma * (s - mean) * std_inv + beta
                out[h, w_, oc] = bn_out
    out = out.astype(bfloat16).astype(np.float32)
    if silu:
        out = _silu(out)
        out = out.astype(bfloat16).astype(np.float32)
    return out


def _concat_channels(*tensors):
    return np.concatenate(tensors, axis=-1)


def _split_weights(weights, IC, OC, PC, PR):
    """Split weights blob into the same named pointers as the kernel."""
    half_part = PC // 2
    rn_neck = PR // 2
    concat_channels = PC + 2 * PR
    p = 0

    def take(n):
        nonlocal p
        r = weights[p:p + n]
        p += n
        return r

    out = {}
    out["conv1_w"] = take(PC * IC)
    out["conv1_bn_w"] = take(PC); out["conv1_bn_b"] = take(PC)
    out["conv1_bn_m"] = take(PC); out["conv1_bn_v"] = take(PC)

    for prefix in ("rn1_", "rn2_"):
        nin = half_part if prefix == "rn1_" else PR
        out[f"{prefix}conv1_w"] = take(rn_neck * nin)
        out[f"{prefix}conv1_bn_w"] = take(rn_neck); out[f"{prefix}conv1_bn_b"] = take(rn_neck)
        out[f"{prefix}conv1_bn_m"] = take(rn_neck); out[f"{prefix}conv1_bn_v"] = take(rn_neck)

        out[f"{prefix}bn_conv3x3_w"] = take(rn_neck * rn_neck * 9)
        out[f"{prefix}bn_bn3x3_w"] = take(rn_neck); out[f"{prefix}bn_bn3x3_b"] = take(rn_neck)
        out[f"{prefix}bn_bn3x3_m"] = take(rn_neck); out[f"{prefix}bn_bn3x3_v"] = take(rn_neck)

        out[f"{prefix}bn_conv1x1_w"] = take(rn_neck * rn_neck)
        out[f"{prefix}bn_bn1x1_w"] = take(rn_neck); out[f"{prefix}bn_bn1x1_b"] = take(rn_neck)
        out[f"{prefix}bn_bn1x1_m"] = take(rn_neck); out[f"{prefix}bn_bn1x1_v"] = take(rn_neck)

        out[f"{prefix}bn_conv2_w"] = take(rn_neck * rn_neck * 9)
        out[f"{prefix}bn_bn2_w"] = take(rn_neck); out[f"{prefix}bn_bn2_b"] = take(rn_neck)
        out[f"{prefix}bn_bn2_m"] = take(rn_neck); out[f"{prefix}bn_bn2_v"] = take(rn_neck)

        out[f"{prefix}conv2_w"] = take(rn_neck * nin)
        out[f"{prefix}conv2_bn_w"] = take(rn_neck); out[f"{prefix}conv2_bn_b"] = take(rn_neck)
        out[f"{prefix}conv2_bn_m"] = take(rn_neck); out[f"{prefix}conv2_bn_v"] = take(rn_neck)

        out[f"{prefix}conv3_w"] = take(PR * 2 * rn_neck)
        out[f"{prefix}conv3_bn_w"] = take(PR); out[f"{prefix}conv3_bn_b"] = take(PR)
        out[f"{prefix}conv3_bn_m"] = take(PR); out[f"{prefix}conv3_bn_v"] = take(PR)

        if prefix == "rn1_":
            out["conv3x3_1_w"] = take(PR * PR * 9)
            out["conv3x3_1_bn_w"] = take(PR); out["conv3x3_1_bn_b"] = take(PR)
            out["conv3x3_1_bn_m"] = take(PR); out["conv3x3_1_bn_v"] = take(PR)

    out["conv3x3_2_w"] = take(PR * PR * 9)
    out["conv3x3_2_bn_w"] = take(PR); out["conv3x3_2_bn_b"] = take(PR)
    out["conv3x3_2_bn_m"] = take(PR); out["conv3x3_2_bn_v"] = take(PR)

    out["conv4_w"] = take(OC * concat_channels)
    out["conv4_bn_w"] = take(OC); out["conv4_bn_b"] = take(OC)
    out["conv4_bn_m"] = take(OC); out["conv4_bn_v"] = take(OC)

    return out


def reference_forward(input_hwc_bf16, weights_bf16, H, W, IC, OC, PC, PR):
    """Numpy reference mirroring the on-device scalar math (HWC layout)."""
    half_part = PC // 2
    rn_neck = PR // 2

    w = _split_weights(weights_bf16.astype(np.float32), IC, OC, PC, PR)

    # Stage 1
    conv1 = _conv1x1_bn_silu(
        input_hwc_bf16, w["conv1_w"],
        w["conv1_bn_w"], w["conv1_bn_b"], w["conv1_bn_m"], w["conv1_bn_v"],
        H, W, IC, PC, silu=True,
    )
    x1 = conv1[:, :, :half_part]
    x2 = conv1[:, :, half_part:]

    def run_repncsp(x_in, IN_C, p):
        # Conv1
        c1 = _conv1x1_bn_silu(
            x_in, w[f"{p}conv1_w"],
            w[f"{p}conv1_bn_w"], w[f"{p}conv1_bn_b"],
            w[f"{p}conv1_bn_m"], w[f"{p}conv1_bn_v"],
            H, W, IN_C, rn_neck, silu=True,
        )
        # Bottleneck branch 1 (3x3 + BN, no SiLU)
        b1 = _conv3x3_bn(
            c1, w[f"{p}bn_conv3x3_w"],
            w[f"{p}bn_bn3x3_w"], w[f"{p}bn_bn3x3_b"],
            w[f"{p}bn_bn3x3_m"], w[f"{p}bn_bn3x3_v"],
            H, W, rn_neck, rn_neck, padding=1, silu=False,
        )
        # Bottleneck branch 2 (1x1 + BN, no SiLU)
        b2 = _conv1x1_bn_silu(
            c1, w[f"{p}bn_conv1x1_w"],
            w[f"{p}bn_bn1x1_w"], w[f"{p}bn_bn1x1_b"],
            w[f"{p}bn_bn1x1_m"], w[f"{p}bn_bn1x1_v"],
            H, W, rn_neck, rn_neck, silu=False,
        )
        # add + silu (vector-tanh sigmoid path, matches add_silu_helper)
        t3 = _silu_tanh((b1 + b2).astype(bfloat16).astype(np.float32))
        t3 = t3.astype(bfloat16).astype(np.float32)
        # bottleneck conv2 (3x3 + BN + SiLU)
        t4 = _conv3x3_bn(
            t3, w[f"{p}bn_conv2_w"],
            w[f"{p}bn_bn2_w"], w[f"{p}bn_bn2_b"],
            w[f"{p}bn_bn2_m"], w[f"{p}bn_bn2_v"],
            H, W, rn_neck, rn_neck, padding=1, silu=True,
        )
        # Residual (no activation)
        b_out = (c1 + t4).astype(bfloat16).astype(np.float32)
        # bypass conv2
        c2 = _conv1x1_bn_silu(
            x_in, w[f"{p}conv2_w"],
            w[f"{p}conv2_bn_w"], w[f"{p}conv2_bn_b"],
            w[f"{p}conv2_bn_m"], w[f"{p}conv2_bn_v"],
            H, W, IN_C, rn_neck, silu=True,
        )
        # concat + merge
        cat = _concat_channels(b_out, c2)
        merged = _conv1x1_bn_silu(
            cat, w[f"{p}conv3_w"],
            w[f"{p}conv3_bn_w"], w[f"{p}conv3_bn_b"],
            w[f"{p}conv3_bn_m"], w[f"{p}conv3_bn_v"],
            H, W, 2 * rn_neck, PR, silu=True,
        )
        return merged

    # Stage 2: RepNCSP #1 on x2
    x3_rn = run_repncsp(x2, half_part, "rn1_")

    # Stage 3: Conv3x3 #1
    x3_conv = _conv3x3_bn(
        x3_rn, w["conv3x3_1_w"],
        w["conv3x3_1_bn_w"], w["conv3x3_1_bn_b"],
        w["conv3x3_1_bn_m"], w["conv3x3_1_bn_v"],
        H, W, PR, PR, padding=1, silu=True,
    )

    # Stage 4: RepNCSP #2 on x3_conv
    x4_rn = run_repncsp(x3_conv, PR, "rn2_")

    # Stage 5: Conv3x3 #2
    x4_conv = _conv3x3_bn(
        x4_rn, w["conv3x3_2_w"],
        w["conv3x3_2_bn_w"], w["conv3x3_2_bn_b"],
        w["conv3x3_2_bn_m"], w["conv3x3_2_bn_v"],
        H, W, PR, PR, padding=1, silu=True,
    )

    # Stage 6: 4-way concat
    cat = _concat_channels(x1, x2, x3_conv, x4_conv)

    # Stage 7: Conv4
    final = _conv1x1_bn_silu(
        cat, w["conv4_w"],
        w["conv4_bn_w"], w["conv4_bn_b"],
        w["conv4_bn_m"], w["conv4_bn_v"],
        H, W, PC + 2 * PR, OC, silu=True,
    )

    return final.reshape(-1)


# =============================================================================
# XRT run
# =============================================================================


def run_with_xrt(xclbin_path, insts_path, args):
    H, W = args.height, args.width
    IC, OC = args.in_channels, args.out_channels
    PC = args.part_channels
    PR = args.process_channels if args.process_channels is not None else PC // 2

    input_size = H * W * IC
    output_size = H * W * OC
    total_w = compute_weight_size(IC, OC, PC, PR)

    rng = np.random.default_rng(42)
    # Use small magnitudes so the post-conv values stay in a sensible range.
    inp_f32 = rng.standard_normal(input_size).astype(np.float32) * 0.5
    wts_f32 = rng.standard_normal(total_w).astype(np.float32) * 0.2

    # BN running_var must be positive; sprinkle positive values where the
    # weights blob holds variances. Easiest: take |x| + 0.5 for all weights
    # used for variances. We don't have an easy map, so we just clip the
    # entire blob via abs() for variances by re-deriving expected indices
    # would be invasive. Instead, set all BN variances to a positive value
    # via _split_weights and then re-pack.
    w_split = _split_weights(wts_f32, IC, OC, PC, PR)
    for k, v in w_split.items():
        if k.endswith("_bn_v") or k.endswith("_bn3x3_v") or k.endswith("_bn1x1_v") or k.endswith("_bn2_v"):
            v[:] = np.abs(v) + 0.5
    # Rebuild the linear blob (_split_weights returned views into wts_f32,
    # so mutations above already persist).

    inp_bf16 = inp_f32.astype(bfloat16)
    wts_bf16 = wts_f32.astype(bfloat16)

    in_tensor = iron.tensor(inp_bf16.view(np.uint16), dtype=np.uint16)
    w_tensor = iron.tensor(wts_bf16.view(np.uint16), dtype=np.uint16)
    out_tensor = iron.zeros(output_size, dtype=np.uint16)

    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)
    DefaultNPURuntime.run(handle, [in_tensor, w_tensor, out_tensor])

    out_u16 = out_tensor.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = reference_forward(
        inp_bf16.reshape(H, W, IC), wts_bf16, H, W, IC, OC, PC, PR
    )
    return actual_f32, expected_f32


# =============================================================================
# Main driver
# =============================================================================


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.process_channels is None:
        args.process_channels = args.part_channels // 2

    device = NPU2Col1()

    try:
        print(
            f"[1/3] Building IRON program "
            f"(H={args.height}, W={args.width}, IC={args.in_channels}, "
            f"OC={args.out_channels}, PC={args.part_channels}, "
            f"PR={args.process_channels})"
        )
        module = build_mlir_module(
            device, args.height, args.width,
            args.in_channels, args.out_channels,
            args.part_channels, args.process_channels,
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
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        # The on-device scalar math uses the AIE2P invsqrt approximation
        # (a few ULPs of error vs. 1/sqrt), plus the C++ `fast_sigmoid`
        # rational approximation. With ~12 BN sqrt and ~10 sigmoid ops on
        # the critical path, a 0.30 absolute tolerance (matching the
        # original test.py) is appropriate.
        atol = 0.30
        rtol = 0.30
        if np.allclose(actual, expected, rtol=rtol, atol=atol):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=rtol, atol=atol)
        print(f"FAILED: {int(mism.sum())}/{len(actual)} mismatches")
        for i in np.where(mism)[0][:8]:
            print(f"        [{i}] got {actual[i]}, expected {expected[i]}")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
