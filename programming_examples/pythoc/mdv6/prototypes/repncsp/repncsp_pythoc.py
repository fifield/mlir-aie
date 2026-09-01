#!/usr/bin/env python3
# repncsp_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 16 --out-channels 16 --work-dir ./repncsp_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 RepNCSP layer as a single-file PythoC + IRON example.

Port of ``programming_examples/ml/mdv6/repncsp/{aie2.py,repncsp_bf16.cc}``
that replaces the external C++ kernel with an inline PythoC kernel.

RepNCSP architecture (kernel_size=1, csp_expand=0.5, repeat_num=1):

    Input ─► Conv1(1x1) + BN + SiLU ─► Bottleneck ─► x1 ─┐
                                                          ├─► Concat ─► Conv3(1x1) + BN + SiLU ─► Output
    Input ─► Conv2(1x1) + BN + SiLU ──────────────────► x2 ─┘

The bottleneck is one RepConv (3x3 + 1x1 parallel branches summed and
SiLU'd) followed by a Conv3x3 + BN + SiLU and a residual.

The original C++ kernel is purely scalar (no SIMD); this port preserves
that structure with the same fast-sqrt / fast-sigmoid approximations so
numerical results match the reference within bf16 precision.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.utils.compile import compile_mlir_module
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import ptr, i32, f32, bf16
from pythoc.aie.profiling import event0, event1
from pythoc.aie.utils import bitcast_i32_to_f32


# ──────────────────────────────────────────────────────────────────────
# Patch: disable LLVM's loop vectorizer in the PythoC compile pipeline.
#
# The default Peano opt -O2 pipeline auto-vectorizes simple bf16 loops
# (e.g. the residual ``out[i] = bf16(f32(a[i]) + f32(b[i]))`` pattern in
# this kernel) into ``fadd <32 x bfloat>``, which AIE2P llc cannot
# legalize — it asserts ``unable to legalize G_FADD <32 x s16>``.
#
# The C++ kernel side-steps the issue because Clang's frontend lowers
# bf16 arithmetic differently, but llvmlite produces native bf16 fadd
# that opt happily packs into a vector.  We patch ``subprocess.run`` so
# the opt invocation gets the extra flag.
# ──────────────────────────────────────────────────────────────────────
import subprocess as _subprocess
_orig_subprocess_run = _subprocess.run


def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args")
    if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and "opt" in str(cmd[0]):
        new_cmd = list(cmd)
        # Insert flags right after the opt binary path.
        new_cmd[1:1] = [
            "-vectorize-loops=false",
            "-vectorize-slp=false",
        ]
        if args:
            args = (new_cmd,) + args[1:]
        else:
            kwargs["args"] = new_cmd
    return _orig_subprocess_run(*args, **kwargs)


_subprocess.run = _patched_subprocess_run

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "repncsp_pythoc_build"


# ────────────────────────────────────────────────────────────────────────
# PythoC helper kernels — direct scalar bf16 port of repncsp_bf16.cc
# ────────────────────────────────────────────────────────────────────────


@aie_kernel
def fast_sqrt(x: f32) -> f32:
    """Newton-Raphson square-root approximation (matches C++ fast_sqrt).

    Equivalent to::

        union { float f; uint32_t i; } conv;
        conv.f = x;
        conv.i = 0x1fbd1df5 + (conv.i >> 1);
        y = 0.5f * (conv.f + x / conv.f);
    """
    if x <= 0.0:
        return f32(0.0)
    bits: i32 = bitcast_i32_to_f32(x, i32)
    bits = 0x1FBD1DF5 + (bits >> 1)
    y: f32 = bitcast_i32_to_f32(bits, f32)
    y = 0.5 * (y + x / y)
    return y


@aie_kernel
def fast_sigmoid(x: f32) -> f32:
    """Fast sigmoid approximation: 0.5 + x / (2*(1+|x|))."""
    abs_x: f32 = x
    if abs_x < 0.0:
        abs_x = -abs_x
    return 0.5 + x / (2.0 * (1.0 + abs_x))


@aie_kernel
def silu_scalar(x: f32) -> f32:
    """SiLU(x) = x * fast_sigmoid(x)."""
    return x * fast_sigmoid(x)


@aie_kernel
def conv1x1_bn_silu(
    inp: ptr[bf16, True],
    weights: ptr[bf16, True],
    bn_weight: ptr[bf16, True],
    bn_bias: ptr[bf16, True],
    bn_mean: ptr[bf16, True],
    bn_var: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
):
    """1x1 conv + BatchNorm + SiLU on bfloat16 HWC tensors."""
    bn_eps: f32 = 1e-3
    oc: i32 = 0
    while oc < out_channels:
        gamma: f32 = f32(bn_weight[oc])
        beta: f32 = f32(bn_bias[oc])
        mean: f32 = f32(bn_mean[oc])
        var: f32 = f32(bn_var[oc])
        inv_std: f32 = 1.0 / fast_sqrt(var + bn_eps)

        h: i32 = 0
        while h < height:
            w: i32 = 0
            while w < width:
                acc: f32 = 0.0
                ic: i32 = 0
                while ic < in_channels:
                    input_idx: i32 = (h * width + w) * in_channels + ic
                    weight_idx: i32 = oc * in_channels + ic
                    acc = acc + f32(inp[input_idx]) * f32(weights[weight_idx])
                    ic = ic + 1
                bn_out: f32 = gamma * (acc - mean) * inv_std + beta
                activated: f32 = silu_scalar(bn_out)
                out_idx: i32 = (h * width + w) * out_channels + oc
                output[out_idx] = bf16(activated)
                w = w + 1
            h = h + 1
        oc = oc + 1


@aie_kernel
def concat_channels(
    x1: ptr[bf16, True],
    x2: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    channels1: i32,
    channels2: i32,
):
    """Concatenate two HWC tensors along the channel axis."""
    h: i32 = 0
    while h < height:
        w: i32 = 0
        while w < width:
            spatial_idx: i32 = h * width + w
            total: i32 = channels1 + channels2
            c: i32 = 0
            while c < channels1:
                src_idx: i32 = spatial_idx * channels1 + c
                dst_idx: i32 = spatial_idx * total + c
                output[dst_idx] = x1[src_idx]
                c = c + 1
            c = 0
            while c < channels2:
                src_idx: i32 = spatial_idx * channels2 + c
                dst_idx: i32 = spatial_idx * total + channels1 + c
                output[dst_idx] = x2[src_idx]
                c = c + 1
            w = w + 1
        h = h + 1


# ────────────────────────────────────────────────────────────────────────
# PythoC main kernel — RepNCSP block
# ────────────────────────────────────────────────────────────────────────


@aie_kernel
def repncsp_bf16_kernel(
    inp: ptr[bf16, True],
    weights_and_bn: ptr[bf16, True],
    output: ptr[bf16, True],
    x1_conv1: ptr[bf16, True],
    x1_bottleneck: ptr[bf16, True],
    x2_conv2: ptr[bf16, True],
    concat_buffer: ptr[bf16, True],
    bn_input_copy: ptr[bf16, True],
    bn_temp1: ptr[bf16, True],
    bn_temp2: ptr[bf16, True],
    bn_temp3: ptr[bf16, True],
    bn_temp4: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
    neck_channels: i32,
):
    """Full RepNCSP block in bfloat16.

    ``neck_channels`` replaces the C++ ``csp_expand`` float argument;
    the host computes ``neck_channels = int(out_channels * csp_expand)``
    and passes it as i32 to keep the kernel signature clean.
    """
    event0()

    bn_eps: f32 = 1e-3
    total_concat_channels: i32 = 2 * neck_channels
    neck_size: i32 = height * width * neck_channels

    # ──── Compute per-section weight offsets (matches C++ pointer math) ────
    conv1_weight_size: i32 = neck_channels * in_channels
    bn_conv3x3_size: i32 = neck_channels * neck_channels * 3 * 3
    bn_conv1x1_size: i32 = neck_channels * neck_channels * 1 * 1
    bn_conv2_size: i32 = neck_channels * neck_channels * 3 * 3
    conv3_weight_size: i32 = out_channels * total_concat_channels

    # Conv1 (1x1) weights + BN params
    conv1_weights_off: i32 = 0
    conv1_bn_weight_off: i32 = conv1_weights_off + conv1_weight_size
    conv1_bn_bias_off: i32 = conv1_bn_weight_off + neck_channels
    conv1_bn_mean_off: i32 = conv1_bn_bias_off + neck_channels
    conv1_bn_var_off: i32 = conv1_bn_mean_off + neck_channels

    # Bottleneck conv3x3 + BN
    bn_conv3x3_weights_off: i32 = conv1_bn_var_off + neck_channels
    bn_bn3x3_weight_off: i32 = bn_conv3x3_weights_off + bn_conv3x3_size
    bn_bn3x3_bias_off: i32 = bn_bn3x3_weight_off + neck_channels
    bn_bn3x3_mean_off: i32 = bn_bn3x3_bias_off + neck_channels
    bn_bn3x3_var_off: i32 = bn_bn3x3_mean_off + neck_channels

    # Bottleneck conv1x1 + BN
    bn_conv1x1_weights_off: i32 = bn_bn3x3_var_off + neck_channels
    bn_bn1x1_weight_off: i32 = bn_conv1x1_weights_off + bn_conv1x1_size
    bn_bn1x1_bias_off: i32 = bn_bn1x1_weight_off + neck_channels
    bn_bn1x1_mean_off: i32 = bn_bn1x1_bias_off + neck_channels
    bn_bn1x1_var_off: i32 = bn_bn1x1_mean_off + neck_channels

    # Bottleneck conv2 (3x3) + BN
    bn_conv2_weights_off: i32 = bn_bn1x1_var_off + neck_channels
    bn_bn2_weight_off: i32 = bn_conv2_weights_off + bn_conv2_size
    bn_bn2_bias_off: i32 = bn_bn2_weight_off + neck_channels
    bn_bn2_mean_off: i32 = bn_bn2_bias_off + neck_channels
    bn_bn2_var_off: i32 = bn_bn2_mean_off + neck_channels

    # Conv2 (1x1) bypass + BN
    conv2_weights_off: i32 = bn_bn2_var_off + neck_channels
    conv2_bn_weight_off: i32 = conv2_weights_off + conv1_weight_size
    conv2_bn_bias_off: i32 = conv2_bn_weight_off + neck_channels
    conv2_bn_mean_off: i32 = conv2_bn_bias_off + neck_channels
    conv2_bn_var_off: i32 = conv2_bn_mean_off + neck_channels

    # Conv3 (1x1) merge + BN
    conv3_weights_off: i32 = conv2_bn_var_off + neck_channels
    conv3_bn_weight_off: i32 = conv3_weights_off + conv3_weight_size
    conv3_bn_bias_off: i32 = conv3_bn_weight_off + out_channels
    conv3_bn_mean_off: i32 = conv3_bn_bias_off + out_channels
    conv3_bn_var_off: i32 = conv3_bn_mean_off + out_channels

    # ──── Stage 1: Conv1 (1x1) + BN + SiLU ────
    conv1x1_bn_silu(
        inp,
        weights_and_bn + conv1_weights_off,
        weights_and_bn + conv1_bn_weight_off,
        weights_and_bn + conv1_bn_bias_off,
        weights_and_bn + conv1_bn_mean_off,
        weights_and_bn + conv1_bn_var_off,
        x1_conv1,
        height,
        width,
        in_channels,
        neck_channels,
    )

    # ──── Stage 2: Bottleneck (RepConv → Conv3x3+BN+SiLU → residual) ────

    # Copy input for residual
    i: i32 = 0
    while i < neck_size:
        bn_input_copy[i] = x1_conv1[i]
        i = i + 1

    # RepConv branch 1: Conv3x3 + BN (no activation)
    oc: i32 = 0
    while oc < neck_channels:
        gamma: f32 = f32(weights_and_bn[bn_bn3x3_weight_off + oc])
        beta: f32 = f32(weights_and_bn[bn_bn3x3_bias_off + oc])
        mean: f32 = f32(weights_and_bn[bn_bn3x3_mean_off + oc])
        var: f32 = f32(weights_and_bn[bn_bn3x3_var_off + oc])
        inv_std: f32 = 1.0 / fast_sqrt(var + bn_eps)

        h: i32 = 0
        while h < height:
            w: i32 = 0
            while w < width:
                acc: f32 = 0.0
                ic: i32 = 0
                while ic < neck_channels:
                    kh: i32 = 0
                    while kh < 3:
                        kw: i32 = 0
                        while kw < 3:
                            ih: i32 = h + kh - 1
                            iw: i32 = w + kw - 1
                            if ih >= 0:
                                if ih < height:
                                    if iw >= 0:
                                        if iw < width:
                                            input_idx: i32 = (ih * width + iw) * neck_channels + ic
                                            weight_idx: i32 = ((oc * neck_channels + ic) * 3 + kh) * 3 + kw
                                            acc = acc + f32(x1_conv1[input_idx]) * f32(
                                                weights_and_bn[bn_conv3x3_weights_off + weight_idx]
                                            )
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1
                bn_out: f32 = gamma * (acc - mean) * inv_std + beta
                temp_idx: i32 = (h * width + w) * neck_channels + oc
                bn_temp1[temp_idx] = bf16(bn_out)
                w = w + 1
            h = h + 1
        oc = oc + 1

    # RepConv branch 2: Conv1x1 + BN (no activation)
    oc = 0
    while oc < neck_channels:
        gamma2: f32 = f32(weights_and_bn[bn_bn1x1_weight_off + oc])
        beta2: f32 = f32(weights_and_bn[bn_bn1x1_bias_off + oc])
        mean2: f32 = f32(weights_and_bn[bn_bn1x1_mean_off + oc])
        var2: f32 = f32(weights_and_bn[bn_bn1x1_var_off + oc])
        inv_std2: f32 = 1.0 / fast_sqrt(var2 + bn_eps)

        h2: i32 = 0
        while h2 < height:
            w2: i32 = 0
            while w2 < width:
                acc2: f32 = 0.0
                ic2: i32 = 0
                while ic2 < neck_channels:
                    in_idx_b: i32 = (h2 * width + w2) * neck_channels + ic2
                    w_idx_b: i32 = oc * neck_channels + ic2
                    acc2 = acc2 + f32(x1_conv1[in_idx_b]) * f32(
                        weights_and_bn[bn_conv1x1_weights_off + w_idx_b]
                    )
                    ic2 = ic2 + 1
                bn_out2: f32 = gamma2 * (acc2 - mean2) * inv_std2 + beta2
                temp_idx2: i32 = (h2 * width + w2) * neck_channels + oc
                bn_temp2[temp_idx2] = bf16(bn_out2)
                w2 = w2 + 1
            h2 = h2 + 1
        oc = oc + 1

    # Add + SiLU (complete RepConv)
    j: i32 = 0
    while j < neck_size:
        s: f32 = f32(bn_temp1[j]) + f32(bn_temp2[j])
        bn_temp3[j] = bf16(silu_scalar(s))
        j = j + 1

    # Bottleneck stage 2: Conv3x3 + BN + SiLU
    oc = 0
    while oc < neck_channels:
        gamma3: f32 = f32(weights_and_bn[bn_bn2_weight_off + oc])
        beta3: f32 = f32(weights_and_bn[bn_bn2_bias_off + oc])
        mean3: f32 = f32(weights_and_bn[bn_bn2_mean_off + oc])
        var3: f32 = f32(weights_and_bn[bn_bn2_var_off + oc])
        inv_std3: f32 = 1.0 / fast_sqrt(var3 + bn_eps)

        h3: i32 = 0
        while h3 < height:
            w3: i32 = 0
            while w3 < width:
                acc3: f32 = 0.0
                ic3: i32 = 0
                while ic3 < neck_channels:
                    kh3: i32 = 0
                    while kh3 < 3:
                        kw3: i32 = 0
                        while kw3 < 3:
                            ih3: i32 = h3 + kh3 - 1
                            iw3: i32 = w3 + kw3 - 1
                            if ih3 >= 0:
                                if ih3 < height:
                                    if iw3 >= 0:
                                        if iw3 < width:
                                            t_idx: i32 = (ih3 * width + iw3) * neck_channels + ic3
                                            w_idx3: i32 = ((oc * neck_channels + ic3) * 3 + kh3) * 3 + kw3
                                            acc3 = acc3 + f32(bn_temp3[t_idx]) * f32(
                                                weights_and_bn[bn_conv2_weights_off + w_idx3]
                                            )
                            kw3 = kw3 + 1
                        kh3 = kh3 + 1
                    ic3 = ic3 + 1
                bn_out3: f32 = gamma3 * (acc3 - mean3) * inv_std3 + beta3
                act3: f32 = silu_scalar(bn_out3)
                temp_idx3: i32 = (h3 * width + w3) * neck_channels + oc
                bn_temp4[temp_idx3] = bf16(act3)
                w3 = w3 + 1
            h3 = h3 + 1
        oc = oc + 1

    # Residual
    k: i32 = 0
    while k < neck_size:
        x1_bottleneck[k] = bf16(f32(bn_input_copy[k]) + f32(bn_temp4[k]))
        k = k + 1

    # ──── Stage 3: Conv2 (1x1) + BN + SiLU (bypass path) ────
    conv1x1_bn_silu(
        inp,
        weights_and_bn + conv2_weights_off,
        weights_and_bn + conv2_bn_weight_off,
        weights_and_bn + conv2_bn_bias_off,
        weights_and_bn + conv2_bn_mean_off,
        weights_and_bn + conv2_bn_var_off,
        x2_conv2,
        height,
        width,
        in_channels,
        neck_channels,
    )

    # ──── Stage 4: Channel-concat [x1_bottleneck, x2_conv2] ────
    concat_channels(
        x1_bottleneck,
        x2_conv2,
        concat_buffer,
        height,
        width,
        neck_channels,
        neck_channels,
    )

    # ──── Stage 5: Conv3 (1x1) + BN + SiLU ────
    conv1x1_bn_silu(
        concat_buffer,
        weights_and_bn + conv3_weights_off,
        weights_and_bn + conv3_bn_weight_off,
        weights_and_bn + conv3_bn_bias_off,
        weights_and_bn + conv3_bn_mean_off,
        weights_and_bn + conv3_bn_var_off,
        output,
        height,
        width,
        total_concat_channels,
        out_channels,
    )

    event1()


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 RepNCSP layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", "-ic", type=int, default=16)
    parser.add_argument("--out-channels", "-oc", type=int, default=16)
    parser.add_argument("--csp-expand", "-e", type=float, default=0.5)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────
# MLIR / IRON construction
# ────────────────────────────────────────────────────────────────────────


def compute_sizes(height, width, in_channels, out_channels, csp_expand):
    neck_channels = int(out_channels * csp_expand)
    concat_channels_n = 2 * neck_channels
    input_size = height * width * in_channels
    output_size = height * width * out_channels
    neck_size = height * width * neck_channels
    concat_size = height * width * concat_channels_n

    conv1_weight_size = neck_channels * in_channels
    bn_conv3x3_size = neck_channels * neck_channels * 3 * 3
    bn_conv1x1_size = neck_channels * neck_channels * 1 * 1
    bn_conv2_size = neck_channels * neck_channels * 3 * 3
    conv3_weight_size = out_channels * concat_channels_n
    conv1_bn_size = 4 * neck_channels
    bn_bn_params = 4 * neck_channels
    conv2_bn_size = 4 * neck_channels
    conv3_bn_size = 4 * out_channels
    bottleneck_weight_size = (
        bn_conv3x3_size + bn_bn_params + bn_conv1x1_size + bn_bn_params
        + bn_conv2_size + bn_bn_params
    )
    total_weight_size = (
        conv1_weight_size + conv1_bn_size
        + bottleneck_weight_size
        + conv1_weight_size + conv2_bn_size  # conv2 has same weight shape as conv1
        + conv3_weight_size + conv3_bn_size
    )

    return {
        "neck_channels": neck_channels,
        "concat_channels": concat_channels_n,
        "input_size": input_size,
        "output_size": output_size,
        "neck_size": neck_size,
        "concat_size": concat_size,
        "total_weight_size": total_weight_size,
    }


def build_mlir_module(device, height, width, in_channels, out_channels, csp_expand):
    sz = compute_sizes(height, width, in_channels, out_channels, csp_expand)
    neck_channels = sz["neck_channels"]

    input_ty = np.ndarray[(sz["input_size"],), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(sz["total_weight_size"],), np.dtype[np.uint16]]
    output_ty = np.ndarray[(sz["output_size"],), np.dtype[np.uint16]]
    neck_ty = np.ndarray[(sz["neck_size"],), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(sz["concat_size"],), np.dtype[np.uint16]]

    # Inline PythoC kernel (compiled at construction time, with helpers prepended)
    kernel = PythocKernel(
        repncsp_bf16_kernel,
        [
            input_ty,    # input
            weight_ty,   # weights_and_bn
            output_ty,   # output
            neck_ty,     # x1_conv1
            neck_ty,     # x1_bottleneck
            neck_ty,     # x2_conv2
            concat_ty,   # concat_buffer
            neck_ty,     # bn_input_copy
            neck_ty,     # bn_temp1
            neck_ty,     # bn_temp2
            neck_ty,     # bn_temp3
            neck_ty,     # bn_temp4
            np.int32,    # height
            np.int32,    # width
            np.int32,    # in_channels
            np.int32,    # out_channels
            np.int32,    # neck_channels
        ],
        helpers=[fast_sqrt, fast_sigmoid, silu_scalar, conv1x1_bn_silu, concat_channels],
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    # Scratch buffers (placed by Worker)
    x1_conv1_buf = Buffer(neck_ty, name="x1_conv1")
    x1_bottleneck_buf = Buffer(neck_ty, name="x1_bottleneck")
    x2_conv2_buf = Buffer(neck_ty, name="x2_conv2")
    concat_buf = Buffer(concat_ty, name="concat_buffer")
    bn_input_copy_buf = Buffer(neck_ty, name="bn_input_copy")
    bn_temp1_buf = Buffer(neck_ty, name="bn_temp1")
    bn_temp2_buf = Buffer(neck_ty, name="bn_temp2")
    bn_temp3_buf = Buffer(neck_ty, name="bn_temp3")
    bn_temp4_buf = Buffer(neck_ty, name="bn_temp4")

    def core_fn(of_in, of_wts, of_out, kernel,
                x1_conv1_b, x1_bn_b, x2_conv2_b, concat_b,
                bn_inp_b, bn_t1, bn_t2, bn_t3, bn_t4):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(
            elem_in, elem_wts, elem_out,
            x1_conv1_b, x1_bn_b, x2_conv2_b, concat_b,
            bn_inp_b, bn_t1, bn_t2, bn_t3, bn_t4,
            height, width, in_channels, out_channels, neck_channels,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_input.cons(), of_weights.cons(), of_output.prod(), kernel,
            x1_conv1_buf, x1_bottleneck_buf, x2_conv2_buf, concat_buf,
            bn_input_copy_buf, bn_temp1_buf, bn_temp2_buf, bn_temp3_buf, bn_temp4_buf,
        ],
        stack_size=4096,
    )

    def sequence(I, W, O, of_input_prod, of_weights_prod, of_output_cons):
        of_input_prod.fill(I)
        of_weights_prod.fill(W)
        of_output_cons.drain(O, wait=True)

    rt = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_input.prod(), of_weights.prod(), of_output.cons()],
    )

    program = Program(device, rt, workers=[worker])
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module, sz


# ────────────────────────────────────────────────────────────────────────
# Numpy reference (matches the scalar C++ kernel bit-for-bit in algorithm)
# ────────────────────────────────────────────────────────────────────────


def _fast_sqrt_np(x):
    """Vectorized numpy version of the C++ fast_sqrt approximation."""
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    bits = xp.view(np.uint32)
    bits = (np.uint32(0x1FBD1DF5) + (bits >> np.uint32(1))).astype(np.uint32)
    y = bits.view(np.float32).copy()
    y = 0.5 * (y + xp / y)
    out[pos] = y
    return out


def _fast_sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 0.5 + x / (2.0 * (1.0 + np.abs(x)))


def _silu_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x * _fast_sigmoid_np(x)


def _conv1x1_bn_silu_np(inp, weights, bnw, bnb, bnm, bnv, H, W, IC, OC):
    """inp: (H, W, IC) float32; weights: (OC, IC); BN params: (OC,)."""
    bn_eps = 1e-3
    out = np.empty((H, W, OC), dtype=np.float32)
    for oc in range(OC):
        gamma = bnw[oc]
        beta = bnb[oc]
        mean = bnm[oc]
        var = bnv[oc]
        inv_std = 1.0 / _fast_sqrt_np(var + bn_eps)
        for h in range(H):
            for w in range(W):
                acc = 0.0
                for ic in range(IC):
                    acc += inp[h, w, ic] * weights[oc, ic]
                bn_out = gamma * (acc - mean) * inv_std + beta
                out[h, w, oc] = _silu_np(bn_out)
    return out


def _conv3x3_bn_np(inp, weights, bnw, bnb, bnm, bnv, H, W, C, with_silu):
    """inp: (H, W, C) float32; weights: (C, C, 3, 3); BN params: (C,)."""
    bn_eps = 1e-3
    out = np.empty((H, W, C), dtype=np.float32)
    for oc in range(C):
        gamma = bnw[oc]
        beta = bnb[oc]
        mean = bnm[oc]
        var = bnv[oc]
        inv_std = 1.0 / _fast_sqrt_np(var + bn_eps)
        for h in range(H):
            for w in range(W):
                acc = 0.0
                for ic in range(C):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = w + kw - 1
                            if 0 <= ih < H and 0 <= iw < W:
                                acc += inp[ih, iw, ic] * weights[oc, ic, kh, kw]
                bn_out = gamma * (acc - mean) * inv_std + beta
                out[h, w, oc] = _silu_np(bn_out) if with_silu else bn_out
    return out


def numpy_reference(input_hwc, weights_uint16, height, width, in_channels, out_channels, csp_expand):
    """Mimic the scalar C++ kernel in numpy (with same approximations)."""
    neck = int(out_channels * csp_expand)
    total_concat = 2 * neck

    # Promote everything to float32 (bf16 is already in input_hwc / weights_uint16).
    inp = input_hwc.view(bfloat16).astype(np.float32).reshape(height, width, in_channels)
    w_bf = weights_uint16.view(bfloat16).astype(np.float32)

    def take(offset, count, shape=None):
        chunk = w_bf[offset:offset + count]
        return chunk if shape is None else chunk.reshape(shape)

    off = 0
    # Conv1
    conv1_w = take(off, neck * in_channels, (neck, in_channels)); off += neck * in_channels
    c1_bw = take(off, neck); off += neck
    c1_bb = take(off, neck); off += neck
    c1_bm = take(off, neck); off += neck
    c1_bv = take(off, neck); off += neck
    # Bottleneck conv3x3
    bn3_w = take(off, neck * neck * 9, (neck, neck, 3, 3)); off += neck * neck * 9
    bn3_bw = take(off, neck); off += neck
    bn3_bb = take(off, neck); off += neck
    bn3_bm = take(off, neck); off += neck
    bn3_bv = take(off, neck); off += neck
    # Bottleneck conv1x1
    bn1_w = take(off, neck * neck, (neck, neck)); off += neck * neck
    bn1_bw = take(off, neck); off += neck
    bn1_bb = take(off, neck); off += neck
    bn1_bm = take(off, neck); off += neck
    bn1_bv = take(off, neck); off += neck
    # Bottleneck conv2 (3x3)
    bn2_w = take(off, neck * neck * 9, (neck, neck, 3, 3)); off += neck * neck * 9
    bn2_bw = take(off, neck); off += neck
    bn2_bb = take(off, neck); off += neck
    bn2_bm = take(off, neck); off += neck
    bn2_bv = take(off, neck); off += neck
    # Conv2 bypass
    conv2_w = take(off, neck * in_channels, (neck, in_channels)); off += neck * in_channels
    c2_bw = take(off, neck); off += neck
    c2_bb = take(off, neck); off += neck
    c2_bm = take(off, neck); off += neck
    c2_bv = take(off, neck); off += neck
    # Conv3
    conv3_w = take(off, out_channels * total_concat, (out_channels, total_concat)); off += out_channels * total_concat
    c3_bw = take(off, out_channels); off += out_channels
    c3_bb = take(off, out_channels); off += out_channels
    c3_bm = take(off, out_channels); off += out_channels
    c3_bv = take(off, out_channels); off += out_channels

    # Stage 1: Conv1 + BN + SiLU
    x1_conv1 = _conv1x1_bn_silu_np(inp, conv1_w, c1_bw, c1_bb, c1_bm, c1_bv,
                                   height, width, in_channels, neck)
    x1_conv1 = x1_conv1.astype(bfloat16).astype(np.float32)

    # Bottleneck
    bn_input_copy = x1_conv1.copy()
    bn_t1 = _conv3x3_bn_np(x1_conv1, bn3_w, bn3_bw, bn3_bb, bn3_bm, bn3_bv,
                            height, width, neck, with_silu=False)
    bn_t1 = bn_t1.astype(bfloat16).astype(np.float32)
    bn_t2_full = np.empty_like(bn_t1)
    for oc in range(neck):
        gamma = bn1_bw[oc]; beta = bn1_bb[oc]; mean = bn1_bm[oc]; var = bn1_bv[oc]
        inv_std = 1.0 / _fast_sqrt_np(var + 1e-3)
        for h in range(height):
            for w in range(width):
                acc = 0.0
                for ic in range(neck):
                    acc += x1_conv1[h, w, ic] * bn1_w[oc, ic]
                bn_t2_full[h, w, oc] = gamma * (acc - mean) * inv_std + beta
    bn_t2 = bn_t2_full.astype(bfloat16).astype(np.float32)

    bn_t3 = (bn_t1 + bn_t2).astype(bfloat16).astype(np.float32)
    bn_t3 = _silu_np(bn_t3).astype(bfloat16).astype(np.float32)

    bn_t4 = _conv3x3_bn_np(bn_t3, bn2_w, bn2_bw, bn2_bb, bn2_bm, bn2_bv,
                            height, width, neck, with_silu=True)
    bn_t4 = bn_t4.astype(bfloat16).astype(np.float32)

    x1_bottleneck = (bn_input_copy + bn_t4).astype(bfloat16).astype(np.float32)

    # Stage 3: Conv2 bypass
    x2_conv2 = _conv1x1_bn_silu_np(inp, conv2_w, c2_bw, c2_bb, c2_bm, c2_bv,
                                   height, width, in_channels, neck)
    x2_conv2 = x2_conv2.astype(bfloat16).astype(np.float32)

    # Stage 4: Concat along channel
    concat = np.concatenate([x1_bottleneck, x2_conv2], axis=2)

    # Stage 5: Conv3 + BN + SiLU
    out = _conv1x1_bn_silu_np(concat, conv3_w, c3_bw, c3_bb, c3_bm, c3_bv,
                              height, width, total_concat, out_channels)
    return out.astype(bfloat16).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Compile & run
# ────────────────────────────────────────────────────────────────────────


def run_with_xrt(xclbin_path, insts_path, sz, args):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    rng = np.random.default_rng(42)
    input_f32 = rng.standard_normal(sz["input_size"]).astype(np.float32) * 0.5

    # Build weight tensor section-by-section so BN running_var stays positive.
    # Layout (matches the kernel pointer math):
    #   conv1_w | conv1_bn(g, b, mean, var) | bn_conv3x3_w | bn3x3_bn(g, b, mean, var)
    #   bn_conv1x1_w | bn1x1_bn(...) | bn_conv2_w | bn2_bn(...)
    #   conv2_w | conv2_bn(...) | conv3_w | conv3_bn(...)
    H, W, IC, OC = args.height, args.width, args.in_channels, args.out_channels
    neck = sz["neck_channels"]
    total_concat = sz["concat_channels"]

    def w(*shape, scale=0.3):
        return rng.standard_normal(np.prod(shape)).astype(np.float32) * scale

    def bn(c):
        gamma = (rng.standard_normal(c).astype(np.float32) * 0.2 + 1.0)
        beta = rng.standard_normal(c).astype(np.float32) * 0.1
        mean = rng.standard_normal(c).astype(np.float32) * 0.2
        var = (rng.standard_normal(c).astype(np.float32) * 0.1 + 1.0).clip(min=0.1)
        return np.concatenate([gamma, beta, mean, var])

    sections = [
        w(neck, IC),          # conv1 weights
        bn(neck),             # conv1 BN
        w(neck, neck, 3, 3),  # bn_conv3x3 weights
        bn(neck),             # bn_conv3x3 BN
        w(neck, neck),        # bn_conv1x1 weights
        bn(neck),             # bn_conv1x1 BN
        w(neck, neck, 3, 3),  # bn_conv2 weights
        bn(neck),             # bn_conv2 BN
        w(neck, IC),          # conv2 weights
        bn(neck),             # conv2 BN
        w(OC, total_concat),  # conv3 weights
        bn(OC),               # conv3 BN
    ]
    weights_f32 = np.concatenate(sections)
    assert weights_f32.size == sz["total_weight_size"], (
        weights_f32.size, sz["total_weight_size"])

    input_bf16 = input_f32.astype(bfloat16)
    weights_bf16 = weights_f32.astype(bfloat16)

    in_buf = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    w_buf = iron.tensor(weights_bf16.view(np.uint16), dtype=np.uint16)
    out_buf = iron.zeros(sz["output_size"], dtype=np.uint16)

    start = time.time_ns()
    DefaultNPURuntime.run(handle, [in_buf, w_buf, out_buf])
    stop = time.time_ns()
    print(f"      NPU execution time: {(stop - start) / 1000:.1f} us")

    actual_f32 = out_buf.numpy().view(bfloat16).astype(np.float32)
    expected_f32 = numpy_reference(
        input_bf16.view(np.uint16), weights_bf16.view(np.uint16),
        args.height, args.width, args.in_channels, args.out_channels, args.csp_expand,
    ).reshape(-1)
    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    sz = compute_sizes(args.height, args.width, args.in_channels, args.out_channels, args.csp_expand)
    print(
        f"RepNCSP config: H={args.height} W={args.width} "
        f"in_C={args.in_channels} out_C={args.out_channels} "
        f"neck={sz['neck_channels']} csp_expand={args.csp_expand}"
    )
    print(
        f"  input={sz['input_size']}  weights={sz['total_weight_size']}  "
        f"output={sz['output_size']}  neck_buf={sz['neck_size']}  concat={sz['concat_size']}"
    )

    device = NPU2Col1()

    try:
        print("[1/3] Building IRON program + compiling PythoC kernel")
        module, _ = build_mlir_module(
            device, args.height, args.width,
            args.in_channels, args.out_channels, args.csp_expand,
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
        actual, expected = run_with_xrt(xclbin_path, insts_path, sz, args)
        print(f"      Output [0:6]:   {actual[:6]}")
        print(f"      Expected [0:6]: {expected[:6]}")
        max_abs = float(np.max(np.abs(actual - expected)))
        print(f"      max |diff| = {max_abs:.5f}")

        # bf16 + scalar approximations: tolerance similar to original test.py
        if np.allclose(actual, expected, rtol=5e-2, atol=5e-2):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=5e-2, atol=5e-2)
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
