#!/usr/bin/env python3
# elan_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 32 --out-channels 32 --part-channels 32 --process-channels 16 --work-dir ./elan_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 ELAN layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/elan/{aie2.py,elan_bf16.cc} that
replaces the external C++ kernel with an inline PythoC kernel (Pattern B).

ELAN (Efficient Layer Aggregation Network) structure:
    Input -> Conv1 (1x1) + BN + SiLU
          -> split into [x1, x2]
          -> x2 -> Conv2 (3x3) + BN + SiLU -> x3
          -> x3 -> Conv3 (3x3) + BN + SiLU -> x4
          -> 4-way concat [x1, x2, x3, x4]
          -> Conv4 (1x1) + BN + SiLU -> Output

BatchNorm is folded into per-channel (w, b) on the host so the device
kernel just multiplies/adds (no sqrt/division needed). SiLU uses the same
fast_sigmoid approximation as the original C++ kernel:
    sigmoid(x) ~= 0.5 + x / (2 * (1 + |x|))
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

from pythoc import ptr, i32, f32, bf16
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "elan_pythoc_build"


# =====================================================================
# PythoC kernels (scalar bf16 with f32 accumulation, matches C++ semantics)
# =====================================================================
#
# We define `elan_bf16_kernel` as the entrypoint and three helpers
# (conv1x1, conv3x3, concat). The helpers must be decorated with
# @aie_kernel and passed via PythocKernel(..., helpers=[...]) so that
# their source is prepended before the main kernel and they can be
# called from it.


@aie_kernel
def conv1x1_bn_silu(
    input: ptr[bf16, True],
    weights: ptr[bf16, True],
    bn_w: ptr[bf16, True],
    bn_b: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
):
    """1x1 conv + (already-folded) BN + SiLU on HWC bf16 tensors.

    Memory layouts:
        input:   (H, W, IC)   row-major
        weights: (OC, IC)     row-major (no spatial dim for 1x1)
        bn_w/b:  (OC,)        folded per-channel BN affine
        output:  (H, W, OC)
    """
    oc: i32 = 0
    while oc < out_channels:
        gamma: f32 = f32(bn_w[oc])
        beta: f32 = f32(bn_b[oc])
        h: i32 = 0
        while h < height:
            w: i32 = 0
            while w < width:
                acc: f32 = 0.0
                ic: i32 = 0
                while ic < in_channels:
                    in_idx: i32 = (h * width + w) * in_channels + ic
                    wt_idx: i32 = oc * in_channels + ic
                    acc = acc + f32(input[in_idx]) * f32(weights[wt_idx])
                    ic = ic + 1
                bn_out: f32 = gamma * acc + beta
                # SiLU with fast_sigmoid: y * (0.5 + y / (2 * (1 + |y|)))
                abs_y: f32 = bn_out if bn_out > 0.0 else (0.0 - bn_out)
                sig: f32 = 0.5 + bn_out / (2.0 * (1.0 + abs_y))
                out_val: f32 = bn_out * sig
                out_idx: i32 = (h * width + w) * out_channels + oc
                output[out_idx] = bf16(out_val)
                w = w + 1
            h = h + 1
        oc = oc + 1


@aie_kernel
def conv3x3_bn_silu(
    input: ptr[bf16, True],
    weights: ptr[bf16, True],
    bn_w: ptr[bf16, True],
    bn_b: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
):
    """3x3 conv (pad=1) + folded BN + SiLU on HWC bf16 tensors.

    Layouts:
        input:   (H, W, IC)
        weights: (OC, IC, 3, 3) row-major
        output:  (H, W, OC)
    """
    oc: i32 = 0
    while oc < out_channels:
        gamma: f32 = f32(bn_w[oc])
        beta: f32 = f32(bn_b[oc])
        h: i32 = 0
        while h < height:
            w: i32 = 0
            while w < width:
                acc: f32 = 0.0
                ic: i32 = 0
                while ic < in_channels:
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
                                            in_idx: i32 = (ih * width + iw) * in_channels + ic
                                            wt_idx: i32 = ((oc * in_channels + ic) * 3 + kh) * 3 + kw
                                            acc = acc + f32(input[in_idx]) * f32(weights[wt_idx])
                            kw = kw + 1
                        kh = kh + 1
                    ic = ic + 1
                bn_out: f32 = gamma * acc + beta
                abs_y: f32 = bn_out if bn_out > 0.0 else (0.0 - bn_out)
                sig: f32 = 0.5 + bn_out / (2.0 * (1.0 + abs_y))
                out_val: f32 = bn_out * sig
                out_idx: i32 = (h * width + w) * out_channels + oc
                output[out_idx] = bf16(out_val)
                w = w + 1
            h = h + 1
        oc = oc + 1


@aie_kernel
def concat_4way_channels(
    x1: ptr[bf16, True],
    x2: ptr[bf16, True],
    x3: ptr[bf16, True],
    x4: ptr[bf16, True],
    output: ptr[bf16, True],
    height: i32,
    width: i32,
    c1: i32,
    c2: i32,
    c3: i32,
    c4: i32,
):
    """4-way channel concatenation in HWC format."""
    total: i32 = c1 + c2 + c3 + c4
    h: i32 = 0
    while h < height:
        w: i32 = 0
        while w < width:
            spatial: i32 = h * width + w
            out_off: i32 = spatial * total
            c: i32 = 0
            while c < c1:
                output[out_off + c] = x1[spatial * c1 + c]
                c = c + 1
            c = 0
            while c < c2:
                output[out_off + c1 + c] = x2[spatial * c2 + c]
                c = c + 1
            c = 0
            while c < c3:
                output[out_off + c1 + c2 + c] = x3[spatial * c3 + c]
                c = c + 1
            c = 0
            while c < c4:
                output[out_off + c1 + c2 + c3 + c] = x4[spatial * c4 + c]
                c = c + 1
            w = w + 1
        h = h + 1


@aie_kernel
def elan_bf16_kernel(
    input: ptr[bf16, True],
    weights_and_bn: ptr[bf16, True],
    output: ptr[bf16, True],
    conv1_output: ptr[bf16, True],
    x3: ptr[bf16, True],
    x4: ptr[bf16, True],
    concat_buffer: ptr[bf16, True],
    height: i32,
    width: i32,
    in_channels: i32,
    out_channels: i32,
    part_channels: i32,
    process_channels: i32,
):
    """Full ELAN block: conv1 -> split -> conv2 -> conv3 -> concat -> conv4.

    BatchNorm is pre-folded on the host into per-channel (w, b) so the on-device
    BN is just `gamma_eff * acc + beta_eff` (no sqrt). Each conv's params in
    weights_and_bn are laid out as [conv_weights, bn_w (C), bn_b (C)].
    """
    event0()

    half_part: i32 = part_channels // 2
    concat_channels: i32 = part_channels + 2 * process_channels

    # ---- Extract per-conv weight/bn pointers ------------------------------
    # Conv1 (1x1): IC -> PC
    conv1_weight_size: i32 = part_channels * in_channels
    conv1_w: ptr[bf16] = weights_and_bn
    conv1_bn_w: ptr[bf16] = conv1_w + conv1_weight_size
    conv1_bn_b: ptr[bf16] = conv1_bn_w + part_channels

    # Conv2 (3x3): half_part -> PRC
    conv2_weight_size: i32 = process_channels * half_part * 3 * 3
    conv2_w: ptr[bf16] = conv1_bn_b + part_channels
    conv2_bn_w: ptr[bf16] = conv2_w + conv2_weight_size
    conv2_bn_b: ptr[bf16] = conv2_bn_w + process_channels

    # Conv3 (3x3): PRC -> PRC
    conv3_weight_size: i32 = process_channels * process_channels * 3 * 3
    conv3_w: ptr[bf16] = conv2_bn_b + process_channels
    conv3_bn_w: ptr[bf16] = conv3_w + conv3_weight_size
    conv3_bn_b: ptr[bf16] = conv3_bn_w + process_channels

    # Conv4 (1x1): concat_channels -> OC
    conv4_weight_size: i32 = out_channels * concat_channels
    conv4_w: ptr[bf16] = conv3_bn_b + process_channels
    conv4_bn_w: ptr[bf16] = conv4_w + conv4_weight_size
    conv4_bn_b: ptr[bf16] = conv4_bn_w + out_channels

    # ---- Stage 1: Conv1 (1x1) + BN + SiLU ------------------------------
    conv1x1_bn_silu(
        input, conv1_w, conv1_bn_w, conv1_bn_b, conv1_output,
        height, width, in_channels, part_channels,
    )

    # ---- Stage 2: split conv1_output into x1, x2 (pointer arithmetic) --
    # x1 = conv1_output[:, :, :half_part]
    # x2 = conv1_output[:, :, half_part:]
    # Because HWC has channels innermost, "split along C" cannot be a pure
    # pointer offset; we materialise a contiguous x2 buffer by copy.
    # We reuse x3 buffer briefly as scratch is not needed: copy directly
    # while iterating the conv2 input.
    # ---- Stage 3: Conv2 (3x3) + BN + SiLU on x2 -----------------------
    # We do the split by passing a stride-aware view: simplest correct path
    # is to materialise x2 in a scratch buffer. Reuse `concat_buffer` since
    # it's the largest and we don't need its contents until stage 5.
    h_i: i32 = 0
    while h_i < height:
        w_i: i32 = 0
        while w_i < width:
            c_i: i32 = 0
            while c_i < half_part:
                src_idx: i32 = (h_i * width + w_i) * part_channels + half_part + c_i
                dst_idx: i32 = (h_i * width + w_i) * half_part + c_i
                concat_buffer[dst_idx] = conv1_output[src_idx]
                c_i = c_i + 1
            w_i = w_i + 1
        h_i = h_i + 1

    conv3x3_bn_silu(
        concat_buffer, conv2_w, conv2_bn_w, conv2_bn_b, x3,
        height, width, half_part, process_channels,
    )

    # ---- Stage 4: Conv3 (3x3) + BN + SiLU on x3 -----------------------
    conv3x3_bn_silu(
        x3, conv3_w, conv3_bn_w, conv3_bn_b, x4,
        height, width, process_channels, process_channels,
    )

    # ---- Stage 5: 4-way concat [x1, x2, x3, x4] ------------------------
    # x1 and x2 are contiguous halves of conv1_output along C, but in HWC
    # they are interleaved per-pixel. concat_4way_channels expects each
    # input as contiguous (H, W, c_i) HWC tensor. Build x1, x2 explicitly
    # into stack-adjacent buffers. We already have x2 in concat_buffer
    # scratch; build x1 into x4 (reusing it - we no longer need it as
    # conv3 output after we read it for concat below). Actually we still
    # need x4 for concat. So instead, manually concat in-place by writing
    # directly into concat_buffer from the four sources without an
    # intermediate x1/x2 split.

    h_j: i32 = 0
    while h_j < height:
        w_j: i32 = 0
        while w_j < width:
            spatial: i32 = h_j * width + w_j
            out_off: i32 = spatial * concat_channels
            # Copy x1 = conv1_output[:, :, :half_part]
            c_j: i32 = 0
            while c_j < half_part:
                concat_buffer[out_off + c_j] = conv1_output[spatial * part_channels + c_j]
                c_j = c_j + 1
            # Copy x2 = conv1_output[:, :, half_part:half_part*2]
            c_j = 0
            while c_j < half_part:
                concat_buffer[out_off + half_part + c_j] = (
                    conv1_output[spatial * part_channels + half_part + c_j]
                )
                c_j = c_j + 1
            # Copy x3
            c_j = 0
            while c_j < process_channels:
                concat_buffer[out_off + 2 * half_part + c_j] = (
                    x3[spatial * process_channels + c_j]
                )
                c_j = c_j + 1
            # Copy x4
            c_j = 0
            while c_j < process_channels:
                concat_buffer[out_off + 2 * half_part + process_channels + c_j] = (
                    x4[spatial * process_channels + c_j]
                )
                c_j = c_j + 1
            w_j = w_j + 1
        h_j = h_j + 1

    # ---- Stage 6: Conv4 (1x1) + BN + SiLU ------------------------------
    conv1x1_bn_silu(
        concat_buffer, conv4_w, conv4_bn_w, conv4_bn_b, output,
        height, width, concat_channels, out_channels,
    )

    event1()


# =====================================================================
# CLI
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 ELAN layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", "-ic", type=int, default=32)
    parser.add_argument("--out-channels", "-oc", type=int, default=32)
    parser.add_argument("--part-channels", "-pc", type=int, default=32)
    parser.add_argument(
        "--process-channels", "-prc", type=int, default=None,
        help="Defaults to part_channels // 2",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# =====================================================================
# MLIR / IRON construction
# =====================================================================


def build_mlir_module(
    device,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    part_channels: int,
    process_channels: int,
):
    half_part = part_channels // 2
    concat_channels = part_channels + 2 * process_channels

    input_size = height * width * in_channels
    output_size = height * width * out_channels
    conv1_size = height * width * part_channels
    process_size = height * width * process_channels
    concat_size = height * width * concat_channels

    # Pre-folded BN: per conv we have [conv_weights, bn_w (C), bn_b (C)]
    conv1_w_size = part_channels * in_channels + 2 * part_channels
    conv2_w_size = process_channels * half_part * 9 + 2 * process_channels
    conv3_w_size = process_channels * process_channels * 9 + 2 * process_channels
    conv4_w_size = out_channels * concat_channels + 2 * out_channels
    weights_size = conv1_w_size + conv2_w_size + conv3_w_size + conv4_w_size

    # bf16 as uint16
    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]
    conv1_ty = np.ndarray[(conv1_size,), np.dtype[np.uint16]]
    process_ty = np.ndarray[(process_size,), np.dtype[np.uint16]]
    concat_ty = np.ndarray[(concat_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        elan_bf16_kernel,
        [
            input_ty,
            weights_ty,
            output_ty,
            conv1_ty,
            process_ty,
            process_ty,
            concat_ty,
            np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
        ],
        helpers=[conv1x1_bn_silu, conv3x3_bn_silu, concat_4way_channels],
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weights_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")

    conv1_buf = Buffer(conv1_ty, name="conv1_output")
    x3_buf = Buffer(process_ty, name="x3")
    x4_buf = Buffer(process_ty, name="x4")
    concat_buf = Buffer(concat_ty, name="concat_buffer")

    def core_fn(of_in, of_wts, of_out, kernel, conv1_buf, x3_buf, x4_buf, concat_buf):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(
            elem_in, elem_wts, elem_out,
            conv1_buf, x3_buf, x4_buf, concat_buf,
            height, width, in_channels, out_channels, part_channels, process_channels,
        )
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_input.cons(), of_weights.cons(), of_output.prod(), kernel,
         conv1_buf, x3_buf, x4_buf, concat_buf],
        stack_size=4096,
    )

    runtime = Runtime()
    with runtime.sequence(input_ty, weights_ty, output_ty) as (I, W, O):
        runtime.start(worker)
        runtime.fill(of_input.prod(), I)
        runtime.fill(of_weights.prod(), W)
        runtime.drain(of_output.cons(), O, wait=True)

    module = Program(device, runtime).resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# =====================================================================
# Reference (numpy) - matches device kernel exactly
# =====================================================================


def _fast_sigmoid(x):
    """Same approximation as elan_bf16.cc::fast_sigmoid."""
    return 0.5 + x / (2.0 * (1.0 + np.abs(x)))


def _silu(x):
    return x * _fast_sigmoid(x)


def _conv1x1_bn_silu_ref(inp, w, bn_w, bn_b, H, W, IC, OC):
    """Reference matching the device kernel's conv1x1+BN+SiLU exactly."""
    inp = inp.reshape(H, W, IC).astype(np.float32)
    w = w.reshape(OC, IC).astype(np.float32)
    bn_w = bn_w.astype(np.float32)
    bn_b = bn_b.astype(np.float32)
    out = np.zeros((H, W, OC), dtype=np.float32)
    for oc in range(OC):
        for h in range(H):
            for ww in range(W):
                acc = float(np.dot(inp[h, ww], w[oc]))
                bn = bn_w[oc] * acc + bn_b[oc]
                out[h, ww, oc] = _silu(bn)
    return out.astype(bfloat16).astype(np.float32).reshape(-1)


def _conv3x3_bn_silu_ref(inp, w, bn_w, bn_b, H, W, IC, OC):
    """Reference matching the device kernel's conv3x3+BN+SiLU (pad=1)."""
    inp = inp.reshape(H, W, IC).astype(np.float32)
    w = w.reshape(OC, IC, 3, 3).astype(np.float32)
    bn_w = bn_w.astype(np.float32)
    bn_b = bn_b.astype(np.float32)
    out = np.zeros((H, W, OC), dtype=np.float32)
    for oc in range(OC):
        for h in range(H):
            for ww in range(W):
                acc = 0.0
                for ic in range(IC):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = ww + kw - 1
                            if 0 <= ih < H and 0 <= iw < W:
                                acc += inp[ih, iw, ic] * w[oc, ic, kh, kw]
                bn = bn_w[oc] * acc + bn_b[oc]
                out[h, ww, oc] = _silu(bn)
    return out.astype(bfloat16).astype(np.float32).reshape(-1)


def elan_reference(input_bf16, weights_bf16, H, W, IC, OC, PC, PRC):
    """Full ELAN reference mirroring elan_bf16_kernel."""
    half_part = PC // 2
    concat_channels = PC + 2 * PRC

    off = 0
    conv1_w_n = PC * IC
    conv1_w = weights_bf16[off:off + conv1_w_n]; off += conv1_w_n
    conv1_bn_w = weights_bf16[off:off + PC]; off += PC
    conv1_bn_b = weights_bf16[off:off + PC]; off += PC

    conv2_w_n = PRC * half_part * 9
    conv2_w = weights_bf16[off:off + conv2_w_n]; off += conv2_w_n
    conv2_bn_w = weights_bf16[off:off + PRC]; off += PRC
    conv2_bn_b = weights_bf16[off:off + PRC]; off += PRC

    conv3_w_n = PRC * PRC * 9
    conv3_w = weights_bf16[off:off + conv3_w_n]; off += conv3_w_n
    conv3_bn_w = weights_bf16[off:off + PRC]; off += PRC
    conv3_bn_b = weights_bf16[off:off + PRC]; off += PRC

    conv4_w_n = OC * concat_channels
    conv4_w = weights_bf16[off:off + conv4_w_n]; off += conv4_w_n
    conv4_bn_w = weights_bf16[off:off + OC]; off += OC
    conv4_bn_b = weights_bf16[off:off + OC]; off += OC

    # Round-trip every intermediate through bf16 to match the device kernel.
    def to_bf16(x): return x.astype(bfloat16).astype(np.float32)

    conv1_out = _conv1x1_bn_silu_ref(input_bf16, conv1_w, conv1_bn_w, conv1_bn_b, H, W, IC, PC)
    conv1_out_hwc = conv1_out.reshape(H, W, PC)
    x1 = conv1_out_hwc[:, :, :half_part]
    x2 = conv1_out_hwc[:, :, half_part:half_part * 2]

    x3 = _conv3x3_bn_silu_ref(
        to_bf16(x2.reshape(-1)), conv2_w, conv2_bn_w, conv2_bn_b,
        H, W, half_part, PRC,
    )
    x4 = _conv3x3_bn_silu_ref(
        x3, conv3_w, conv3_bn_w, conv3_bn_b,
        H, W, PRC, PRC,
    )

    concat = np.concatenate([
        to_bf16(x1.reshape(H * W, half_part)),
        to_bf16(x2.reshape(H * W, half_part)),
        x3.reshape(H * W, PRC),
        x4.reshape(H * W, PRC),
    ], axis=1).reshape(-1)

    out = _conv1x1_bn_silu_ref(concat, conv4_w, conv4_bn_w, conv4_bn_b, H, W, concat_channels, OC)
    return out


# =====================================================================
# Compile & run
# =====================================================================


def _make_weights(H, W, IC, OC, PC, PRC, rng):
    half_part = PC // 2
    concat_channels = PC + 2 * PRC
    parts = []

    def fused_bn(out_c):
        # Construct realistic-looking gamma, beta, mean, var, fold into (w', b').
        eps = 1e-3
        gamma = rng.standard_normal(out_c).astype(np.float32) * 0.5 + 1.0
        beta = rng.standard_normal(out_c).astype(np.float32) * 0.1
        mean = rng.standard_normal(out_c).astype(np.float32) * 0.1
        var = np.abs(rng.standard_normal(out_c).astype(np.float32)) + 0.5
        inv_std = 1.0 / np.sqrt(var + eps)
        bn_w = (gamma * inv_std).astype(bfloat16)
        bn_b = (beta - mean * gamma * inv_std).astype(bfloat16)
        return bn_w, bn_b

    # Conv1
    parts.append(rng.standard_normal(PC * IC).astype(np.float32) * 0.1)
    bn_w, bn_b = fused_bn(PC)
    parts.append(bn_w.astype(np.float32))
    parts.append(bn_b.astype(np.float32))
    # Conv2
    parts.append(rng.standard_normal(PRC * half_part * 9).astype(np.float32) * 0.1)
    bn_w, bn_b = fused_bn(PRC)
    parts.append(bn_w.astype(np.float32))
    parts.append(bn_b.astype(np.float32))
    # Conv3
    parts.append(rng.standard_normal(PRC * PRC * 9).astype(np.float32) * 0.1)
    bn_w, bn_b = fused_bn(PRC)
    parts.append(bn_w.astype(np.float32))
    parts.append(bn_b.astype(np.float32))
    # Conv4
    parts.append(rng.standard_normal(OC * concat_channels).astype(np.float32) * 0.1)
    bn_w, bn_b = fused_bn(OC)
    parts.append(bn_w.astype(np.float32))
    parts.append(bn_b.astype(np.float32))

    weights_f32 = np.concatenate(parts)
    return weights_f32.astype(bfloat16)


def run_with_xrt(xclbin_path: Path, insts_path: Path,
                 H: int, W: int, IC: int, OC: int, PC: int, PRC: int):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    input_size = H * W * IC
    output_size = H * W * OC

    rng = np.random.default_rng(42)
    input_f32 = rng.standard_normal(input_size).astype(np.float32)
    input_bf16 = input_f32.astype(bfloat16)
    weights_bf16 = _make_weights(H, W, IC, OC, PC, PRC, rng)

    in_input = iron.tensor(input_bf16.view(np.uint16), dtype=np.uint16)
    in_weights = iron.tensor(weights_bf16.view(np.uint16), dtype=np.uint16)
    out_tensor = iron.zeros(output_size, dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_input, in_weights, out_tensor])
    out_u16 = out_tensor.numpy()
    actual_f32 = np.array(out_u16, dtype=np.uint16).view(bfloat16).astype(np.float32)

    expected_f32 = elan_reference(input_bf16, weights_bf16, H, W, IC, OC, PC, PRC)
    return actual_f32, expected_f32


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.process_channels is None:
        args.process_channels = args.part_channels // 2

    if args.part_channels % 2:
        raise ValueError("part_channels must be even")

    device = NPU2Col1()

    try:
        print(
            f"[1/3] Building IRON program "
            f"(H={args.height}, W={args.width}, IC={args.in_channels}, "
            f"OC={args.out_channels}, PC={args.part_channels}, PRC={args.process_channels})"
        )
        module = build_mlir_module(
            device,
            args.height, args.width,
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
        actual, expected = run_with_xrt(
            xclbin_path, insts_path,
            args.height, args.width,
            args.in_channels, args.out_channels,
            args.part_channels, args.process_channels,
        )
        print(f"      Output:   {actual[:8]}")
        print(f"      Expected: {expected[:8]}")

        # Device uses bf16 round-trips on every intermediate; allow a modest
        # tolerance for accumulated rounding.
        rtol = 5e-2
        atol = 5e-2
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
