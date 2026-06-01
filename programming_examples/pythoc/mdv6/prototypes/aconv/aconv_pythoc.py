#!/usr/bin/env python3
# aconv_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --height 8 --width 8 --in-channels 8 --out-channels 8 --work-dir ./aconv_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 AConv layer as a PythoC + IRON example.

Port of programming_examples/ml/mdv6/aconv/{aie2.py,aconv_bf16.cc} that
replaces the external C++ kernel with an inline PythoC kernel.

AConv = AvgPool2d(2×2, stride=1) + Conv3x3(stride=2, padding=1) + BN + SiLU

The implementation mirrors the scalar C++ reference (aconv_bf16.cc) exactly:

  * Element-wise math in f32 (load bf16, cast up; cast back to bf16 on store).
  * BN inverse-std uses the AIE2P `invsqrt` HW intrinsic in place of the
    Quake-style `fast_sqrt` bit hack from the C++ scalar code.
  * SiLU uses the same `fast_sigmoid` rational approximation as the C++
    kernel: sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|)).

The reference computed in NumPy uses the exact same approximations so the
NPU output matches to within bf16 quantization noise (≈1e-2).
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
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module

from pythoc import bf16, f32, i32, ptr
from pythoc.aie import invsqrt
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "aconv_pythoc_build"


# ── PythoC kernel ──────────────────────────────────────────────────────


@aie_kernel
def aconv_bf16_kernel(
    input: ptr[bf16, True],
    weights_and_bn: ptr[bf16, True],
    output: ptr[bf16, True],
    temp_buffer: ptr[bf16, True],
    input_height: i32,
    input_width: i32,
    input_channels: i32,
    output_channels: i32,
):
    """Fused AvgPool2d(2×2,s=1) + Conv3x3(s=2,p=1) + BN + SiLU on bf16.

    Layout (matches the original C++ scalar kernel):
        input        : (H, W, IC)            row-major
        weights_and_bn : [conv_weights(OC,IC,3,3), gamma(OC), beta(OC),
                         mean(OC), var(OC)]
        output       : (H_out, W_out, OC)    row-major
        temp_buffer  : (H-1, W-1, IC)        intermediate pooled output
    """
    event0()

    # Layout offsets within the packed weight buffer
    weight_size: i32 = output_channels * input_channels * 3 * 3
    bn_weight_off: i32 = weight_size
    bn_bias_off: i32 = bn_weight_off + output_channels
    bn_mean_off: i32 = bn_bias_off + output_channels
    bn_var_off: i32 = bn_mean_off + output_channels
    bn_eps: f32 = 0.001  # 1e-3, matches C++ aconv_bf16.cc

    # ── Stage 1: AvgPool2d (2×2, stride=1, padding=0) ─────────────────
    pooled_height: i32 = input_height - 1
    pooled_width: i32 = input_width - 1
    inv4: f32 = 0.25  # 1.0 / 4.0

    oh: i32 = 0
    while oh < pooled_height:
        ow: i32 = 0
        while ow < pooled_width:
            c: i32 = 0
            while c < input_channels:
                sum_v: f32 = 0.0

                # 2×2 average pooling window
                kh: i32 = 0
                while kh < 2:
                    kw: i32 = 0
                    while kw < 2:
                        ih: i32 = oh + kh
                        iw: i32 = ow + kw
                        in_idx: i32 = (ih * input_width + iw) * input_channels + c
                        in_val: bf16 = input[in_idx]
                        sum_v = sum_v + f32(in_val)
                        kw = kw + 1
                    kh = kh + 1

                temp_idx: i32 = (oh * pooled_width + ow) * input_channels + c
                temp_buffer[temp_idx] = bf16(sum_v * inv4)
                c = c + 1
            ow = ow + 1
        oh = oh + 1

    # ── Stage 2: Conv3x3 (s=2,p=1) + BatchNorm + SiLU ─────────────────
    conv_output_height: i32 = (pooled_height + 2 - 3) // 2 + 1
    conv_output_width: i32 = (pooled_width + 2 - 3) // 2 + 1

    half_f: f32 = 0.5
    one_f: f32 = 1.0
    two_f: f32 = 2.0

    oc: i32 = 0
    while oc < output_channels:
        # Hoist BN params for this output channel
        gamma_bf: bf16 = weights_and_bn[bn_weight_off + oc]
        beta_bf: bf16 = weights_and_bn[bn_bias_off + oc]
        mean_bf: bf16 = weights_and_bn[bn_mean_off + oc]
        var_bf: bf16 = weights_and_bn[bn_var_off + oc]
        gamma: f32 = f32(gamma_bf)
        beta: f32 = f32(beta_bf)
        mean: f32 = f32(mean_bf)
        var: f32 = f32(var_bf)
        inv_std: f32 = invsqrt(var + bn_eps)

        ohh: i32 = 0
        while ohh < conv_output_height:
            oww: i32 = 0
            while oww < conv_output_width:
                acc: f32 = 0.0

                # 3×3 convolution over pooled buffer, stride=2, padding=1
                ic: i32 = 0
                while ic < input_channels:
                    kh2: i32 = 0
                    while kh2 < 3:
                        kw2: i32 = 0
                        while kw2 < 3:
                            ih2: i32 = ohh * 2 + kh2 - 1
                            iw2: i32 = oww * 2 + kw2 - 1
                            if ih2 >= 0:
                                if ih2 < pooled_height:
                                    if iw2 >= 0:
                                        if iw2 < pooled_width:
                                            t_idx: i32 = (ih2 * pooled_width + iw2) * input_channels + ic
                                            w_idx: i32 = ((oc * input_channels + ic) * 3 + kh2) * 3 + kw2
                                            t_val: bf16 = temp_buffer[t_idx]
                                            w_val: bf16 = weights_and_bn[w_idx]
                                            acc = acc + f32(t_val) * f32(w_val)
                            kw2 = kw2 + 1
                        kh2 = kh2 + 1
                    ic = ic + 1

                # BatchNorm: y = gamma * (x - mean) * inv_std + beta
                bn_out: f32 = gamma * (acc - mean) * inv_std + beta

                # SiLU using C++ kernel's fast_sigmoid approximation:
                #   sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|))
                #   silu(x) = x * sigmoid(x)
                abs_bn: f32 = bn_out
                if bn_out < 0.0:
                    abs_bn = -bn_out
                denom: f32 = two_f * (one_f + abs_bn)
                sig: f32 = half_f + bn_out / denom
                activated: f32 = bn_out * sig

                out_idx: i32 = (ohh * conv_output_width + oww) * output_channels + oc
                output[out_idx] = bf16(activated)
                oww = oww + 1
            ohh = ohh + 1
        oc = oc + 1

    event1()


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 AConv layer (PythoC + IRON, bf16)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument(
        "--in-channels", "-ic", type=int, default=8, help="Input channels"
    )
    parser.add_argument(
        "--out-channels", "-oc", type=int, default=8, help="Output channels"
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── Shape helpers ─────────────────────────────────────────────────────


def compute_shapes(H, W, IC, OC):
    PH = H - 1
    PW = W - 1
    OH = (PH + 2 - 3) // 2 + 1
    OW = (PW + 2 - 3) // 2 + 1
    sizes = {
        "input": H * W * IC,
        "temp": PH * PW * IC,
        "weights": OC * IC * 3 * 3,
        "bn": 4 * OC,
        "output": OH * OW * OC,
        "pooled_height": PH,
        "pooled_width": PW,
        "output_height": OH,
        "output_width": OW,
    }
    sizes["total_weight"] = sizes["weights"] + sizes["bn"]
    return sizes


# ── MLIR / IRON construction ──────────────────────────────────────────


def build_mlir_module(device, H, W, IC, OC):
    s = compute_shapes(H, W, IC, OC)

    input_ty = np.ndarray[(s["input"],), np.dtype[np.uint16]]
    temp_ty = np.ndarray[(s["temp"],), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(s["total_weight"],), np.dtype[np.uint16]]
    output_ty = np.ndarray[(s["output"],), np.dtype[np.uint16]]

    kernel = PythocKernel(
        aconv_bf16_kernel,
        [input_ty, weight_ty, output_ty, temp_ty, np.int32, np.int32, np.int32, np.int32],
        extra_globals={"invsqrt": invsqrt},
    )

    of_input = ObjectFifo(input_ty, depth=1, name="input_L3L2")
    of_weights = ObjectFifo(weight_ty, depth=1, name="weights_L3L2")
    of_output = ObjectFifo(output_ty, depth=1, name="output_L2L3")
    temp_buffer = Buffer(temp_ty, name="temp_pooled")

    def core_fn(of_in, of_wts, of_out, temp_buf, kernel):
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_wts, elem_out, temp_buf, H, W, IC, OC)
        of_in.release(1)
        of_wts.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [
            of_input.cons(),
            of_weights.cons(),
            of_output.prod(),
            temp_buffer,
            kernel,
        ],
    )

    runtime = Runtime()
    with runtime.sequence(input_ty, weight_ty, output_ty) as (I, W_, O):
        runtime.start(worker)
        runtime.fill(of_input.prod(), I)
        runtime.fill(of_weights.prod(), W_)
        runtime.drain(of_output.cons(), O, wait=True)

    program = Program(device, runtime)
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── NumPy reference (mirrors the kernel's formulas exactly) ───────────


def aconv_reference(input_hwc_bf16, weights_oicc_bf16, gamma_bf16, beta_bf16,
                    mean_bf16, var_bf16, H, W, IC, OC):
    """Reproduce the PythoC kernel's math in NumPy (f32 intermediates)."""
    PH = H - 1
    PW = W - 1
    OH = (PH + 2 - 3) // 2 + 1
    OW = (PW + 2 - 3) // 2 + 1

    inp = input_hwc_bf16.astype(np.float32)  # (H, W, IC)
    wts = weights_oicc_bf16.astype(np.float32)  # (OC, IC, 3, 3)
    gamma = gamma_bf16.astype(np.float32)
    beta = beta_bf16.astype(np.float32)
    mean = mean_bf16.astype(np.float32)
    var = var_bf16.astype(np.float32)
    bn_eps = np.float32(1e-3)

    # Stage 1: AvgPool2d (2x2, s=1)
    pooled = np.zeros((PH, PW, IC), dtype=np.float32)
    for oh in range(PH):
        for ow in range(PW):
            for c in range(IC):
                s = 0.0
                for kh in range(2):
                    for kw in range(2):
                        s += inp[oh + kh, ow + kw, c]
                pooled[oh, ow, c] = s / 4.0
    # Quantize to bf16 between stages (matches kernel's `temp_buffer[temp_idx] = (bfloat16)`)
    pooled = pooled.astype(bfloat16).astype(np.float32)

    out = np.zeros((OH, OW, OC), dtype=np.float32)
    inv_std = np.float32(1.0) / np.sqrt(var + bn_eps)
    for oc in range(OC):
        g = gamma[oc]
        b = beta[oc]
        m = mean[oc]
        ist = inv_std[oc]
        for oh in range(OH):
            for ow in range(OW):
                acc = np.float32(0.0)
                for ic in range(IC):
                    for kh in range(3):
                        for kw in range(3):
                            ih = oh * 2 + kh - 1
                            iw = ow * 2 + kw - 1
                            if 0 <= ih < PH and 0 <= iw < PW:
                                acc += pooled[ih, iw, ic] * wts[oc, ic, kh, kw]
                bn_out = g * (acc - m) * ist + b
                abs_bn = abs(bn_out)
                sig = np.float32(0.5) + bn_out / (np.float32(2.0) * (np.float32(1.0) + abs_bn))
                out[oh, ow, oc] = bn_out * sig
    return out.astype(bfloat16).astype(np.float32)  # (OH, OW, OC)


# ── Compile & run ─────────────────────────────────────────────────────


def run_with_xrt(xclbin_path: Path, insts_path: Path, H, W, IC, OC):
    s = compute_shapes(H, W, IC, OC)

    rng = np.random.default_rng(42)
    # Input: (H, W, IC) in float, cast to bf16
    input_hwc_f32 = rng.standard_normal((H, W, IC)).astype(np.float32)
    input_hwc_bf16 = input_hwc_f32.astype(bfloat16)

    # Conv weights (OC, IC, 3, 3) — small values so accumulation stays sane
    weights_f32 = (rng.standard_normal((OC, IC, 3, 3)) * 0.2).astype(np.float32)
    weights_bf16 = weights_f32.astype(bfloat16)

    # BN params: gamma ≈ 1, beta ≈ 0, mean small, var ≈ 1
    gamma_f32 = (np.ones(OC) + 0.05 * rng.standard_normal(OC)).astype(np.float32)
    beta_f32 = (0.05 * rng.standard_normal(OC)).astype(np.float32)
    mean_f32 = (0.05 * rng.standard_normal(OC)).astype(np.float32)
    var_f32 = (np.ones(OC) + 0.05 * np.abs(rng.standard_normal(OC))).astype(np.float32)

    gamma_bf16 = gamma_f32.astype(bfloat16)
    beta_bf16 = beta_f32.astype(bfloat16)
    mean_bf16 = mean_f32.astype(bfloat16)
    var_bf16 = var_f32.astype(bfloat16)

    # Pack: [weights, gamma, beta, mean, var] as bf16/uint16
    packed_weights = np.concatenate(
        [
            weights_bf16.flatten().view(np.uint16),
            gamma_bf16.view(np.uint16),
            beta_bf16.view(np.uint16),
            mean_bf16.view(np.uint16),
            var_bf16.view(np.uint16),
        ]
    )
    assert packed_weights.size == s["total_weight"], (
        packed_weights.size,
        s["total_weight"],
    )

    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    in_buf = iron.tensor(input_hwc_bf16.flatten().view(np.uint16), dtype=np.uint16)
    wt_buf = iron.tensor(packed_weights, dtype=np.uint16)
    out_buf = iron.zeros(s["output"], dtype=np.uint16)

    DefaultNPURuntime.run(handle, [in_buf, wt_buf, out_buf])

    actual = (
        np.array(out_buf.numpy(), dtype=np.uint16)
        .view(bfloat16)
        .astype(np.float32)
        .reshape(s["output_height"], s["output_width"], OC)
    )
    expected = aconv_reference(
        input_hwc_bf16,
        weights_bf16,
        gamma_bf16,
        beta_bf16,
        mean_bf16,
        var_bf16,
        H,
        W,
        IC,
        OC,
    )
    return actual, expected


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()
    H = args.height
    W = args.width
    IC = args.in_channels
    OC = args.out_channels
    s = compute_shapes(H, W, IC, OC)

    try:
        print(
            f"[1/3] Building IRON program (H={H}, W={W}, IC={IC}, OC={OC}) "
            f"-> output ({s['output_height']}, {s['output_width']}, {OC})"
        )
        module = build_mlir_module(device, H, W, IC, OC)
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
        actual, expected = run_with_xrt(xclbin_path, insts_path, H, W, IC, OC)
        print(f"      Output[0,0,:8]:   {actual[0, 0, :8]}")
        print(f"      Expected[0,0,:8]: {expected[0, 0, :8]}")
        print(
            f"      max |a-e|={np.max(np.abs(actual - expected)):.4f}  "
            f"mean |a-e|={np.mean(np.abs(actual - expected)):.4f}"
        )

        # Tolerance: bf16 round-off plus invsqrt HW approximation accumulate.
        if np.allclose(actual, expected, rtol=5e-2, atol=5e-2):
            print("PASS!")
            return 0
        mism = ~np.isclose(actual, expected, rtol=5e-2, atol=5e-2)
        print(f"FAILED: {int(mism.sum())}/{actual.size} mismatches")
        idx = np.argwhere(mism)
        for (i0, i1, i2) in idx[:10]:
            print(
                f"        [{i0},{i1},{i2}] got {actual[i0, i1, i2]}, "
                f"expected {expected[i0, i1, i2]}"
            )
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
