#!/usr/bin/env python3
# gemm_conv1x1_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./gemm_conv1x1_pythoc_build | FileCheck %s
# CHECK: PASS!

"""MDV6 1x1 convolution (GEMM form) as a single-file PythoC + IRON example.

Port of programming_examples/pythoc/mdv6/gemm_conv1x1/{aie2_gemm_conv1x1.py,
gemm_conv1x1_bf16.cc} that replaces the external C++ kernel with an inline
PythoC kernel.

A 1x1 convolution is a matmul over the spatial dimension:

    Out[M, N] = In[M, K] @ W[K, N]

where M = H*W (spatial pixels), K = input_channels (ic), N = output_channels
(oc). The original layer also fused BatchNorm + SiLU after the matmul; this
port keeps the BFP16 GEMM only (the dominant FLOP path) and validates against
a numpy matmul reference. BN+SiLU is not implemented — see "Deferred" below.

The PythoC kernel reuses the proven BFP16 8x8x8 hardware MAC path from
../../bf16_gemm_single_core.py (bf16 -> accfloat -> bfp16ebs8 -> BFP576 MAC).
The original C++ kernel used aie::mmul<4,8,8,bfloat16,bfloat16> which has no
direct PythoC analogue; we use the 8x8x8 BFP16 MAC instead.

Deferred (vs. original layer):
  - BN + SiLU fusion (the original .cc applied (x * fast_sigmoid(x)) after
    the matmul); validated here as pure GEMM only.
  - Multi-core (n_cores=32) and K-blocking. This single-file example runs
    one tile on one compute core, matching the elementwise port's scope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel

# PythoC types and intrinsics
from pythoc import ptr, i32, f32, bf16, void
from pythoc.aie import (
    load_v,
    store_v,
    aie_vector,
    zeros,
    vector_extract,
    concat,
    vector_cast,
    v32bf16_to_v32accfloat,
    v64accfloat_to_v64bfp16ebs8,
    vshuffle,
    BFP576_BFP576_ACC2048_mac_conf,
    set_ctrl_reg,
    prepare_for_pipelining,
)
from pythoc.aie.profiling import event0, event1

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "gemm_conv1x1_pythoc_build"

# ── Tile dimensions ──────────────────────────────────────────────────────────
#
# Sized to fit on a single compute tile with no mem-tile retiling. The
# single-core bf16 GEMM example proves 32x32x32 fits with depth=2 ObjectFifos.
#
# For Conv1x1: M = pixels (H*W), K = input_channels, N = output_channels.
# Defaults map to a 32-pixel patch with ic=oc=32, exercising the same 2x2
# register blocking and BFP16 MAC pipeline as bf16_gemm_single_core.py.
#
# Layout in L1 (tiled 8x8 blocks):
#   A_tile (In) : [K_MICRO, M_BLOCKS, 8, 8]  bf16  (K_MICRO*M_BLOCKS*64 elems)
#   B_tile (W)  : [N_BLOCKS, K_MICRO, 8, 8]  bf16
#   C_tile (Out): [N_BLOCKS, M_BLOCKS, 8, 8] f32

TILE_M = 32  # pixels per tile
TILE_K = 32  # input channels
TILE_N = 32  # output channels

M_BLOCKS = TILE_M // 8  # 4
N_BLOCKS = TILE_N // 8  # 4
K_MICRO = TILE_K // 8   # 4
BLOCK = 64              # 8*8 elems per block
MAC_CONF = 780          # sgn_x=1, sgn_y=1, amode=2, bmode=1

A_ELEMS = TILE_M * TILE_K  # input  tile
B_ELEMS = TILE_K * TILE_N  # weight tile
C_ELEMS = TILE_M * TILE_N  # output tile (f32)

# Single-core demo: full matrix = one tile (M=K=N=32)
M = TILE_M
K = TILE_K
N = TILE_N


# ── PythoC kernel: BFP16 8x8x8 GEMM tile ────────────────────────────────────
#
# Identical pattern to bf16_gemm_single_core.py's bf16_gemm_kernel; the only
# semantic difference is naming (this kernel computes a 1x1 conv tile as a
# GEMM, so input ~ A and weights ~ B). 2x2 register blocking: each
# inner-most K step issues 4 MACs sharing 2 A and 2 B loads.


@aie_kernel
def gemm_conv1x1_kernel(
    in_buf: ptr[bf16, True],   # input  tile: [K_MICRO, M_BLOCKS, 8, 8] bf16
    wt_buf: ptr[bf16, True],   # weight tile: [N_BLOCKS, K_MICRO, 8, 8] bf16
    out_buf: ptr[f32, True],   # output tile: [N_BLOCKS, M_BLOCKS, 8, 8] f32
) -> void:
    """Out[M,N] += In[M,K] x W[K,N] via 8x8x8 BFP16 MAC chain (2x2 blocked)."""
    # Rounding modes (matches bf16_gemm_single_core.py)
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)

    # Buffer strides (in elements)
    A_K_STRIDE: i32 = 256   # M_BLOCKS * BLOCK = 4*64
    B_K_STRIDE: i32 = 64    # BLOCK
    B_N_STRIDE: i32 = 256   # K_MICRO * BLOCK = 4*64
    C_M_STRIDE: i32 = 64    # BLOCK
    C_N_STRIDE: i32 = 256   # M_BLOCKS * BLOCK = 4*64

    # ── Zero-initialise the C buffer (out tile starts at 0) ──────────
    z: aie_vector[f32, 64] = zeros(f32, 64)
    zi: i32 = 0
    while zi < 16:  # N_BLOCKS * M_BLOCKS = 16
        store_v(out_buf + zi * 64, z)
        zi = zi + 1

    event0()

    # ── 2x2 register-blocked GEMM ────────────────────────────────────
    m: i32 = 0
    while m < 4:  # M_BLOCKS, step 2
        n: i32 = 0
        while n < 4:  # N_BLOCKS, step 2
            c00_off: i32 = n * C_N_STRIDE + m * C_M_STRIDE
            c10_off: i32 = c00_off + C_M_STRIDE
            c01_off: i32 = c00_off + C_N_STRIDE
            c11_off: i32 = c00_off + C_N_STRIDE + C_M_STRIDE

            acc_c00: aie_vector[f32, 64] = load_v(out_buf + c00_off, 64)
            acc_c10: aie_vector[f32, 64] = load_v(out_buf + c10_off, 64)
            acc_c01: aie_vector[f32, 64] = load_v(out_buf + c01_off, 64)
            acc_c11: aie_vector[f32, 64] = load_v(out_buf + c11_off, 64)

            a0_off: i32 = m * 64
            a1_off: i32 = a0_off + 64
            b0_off: i32 = n * B_N_STRIDE
            b1_off: i32 = (n + 1) * B_N_STRIDE

            k: i32 = 0
            while k < 4:  # K_MICRO
                # ── Load A0, B0 ─────────────────────────────────
                va0: aie_vector[bf16, 64] = load_v(in_buf + a0_off, 64)
                a0_off = a0_off + A_K_STRIDE
                vb0: aie_vector[bf16, 64] = load_v(wt_buf + b0_off, 64)
                b0_off = b0_off + B_K_STRIDE

                # ── Convert A0 (no vshuffle) ─────────────────────
                a0_lo: aie_vector[bf16, 32] = vector_extract(va0, 0, 32)
                a0_hi: aie_vector[bf16, 32] = vector_extract(va0, 32, 32)
                a0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_lo)
                a0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_hi)
                a0_acc: aie_vector[f32, 64] = concat(a0_acc_lo, a0_acc_hi)

                # ── Convert B0 (vshuffle path) ───────────────────
                b0_i32: aie_vector[i32, 32] = vector_cast(vb0, i32, 32)
                b0_lo_i: aie_vector[i32, 16] = vector_extract(b0_i32, 0, 16)
                b0_hi_i: aie_vector[i32, 16] = vector_extract(b0_i32, 16, 16)
                b0_even: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 52)
                b0_odd: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 53)
                b0_cat: aie_vector[i32, 32] = concat(b0_even, b0_odd)
                vb0_s: aie_vector[bf16, 64] = vector_cast(b0_cat, bf16, 64)
                b0_s_lo: aie_vector[bf16, 32] = vector_extract(vb0_s, 0, 32)
                b0_s_hi: aie_vector[bf16, 32] = vector_extract(vb0_s, 32, 32)
                b0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0_s_lo)
                b0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0_s_hi)
                b0_acc: aie_vector[f32, 64] = concat(b0_acc_lo, b0_acc_hi)

                a0_mant, a0_exp = v64accfloat_to_v64bfp16ebs8(a0_acc)
                b0_mant, b0_exp = v64accfloat_to_v64bfp16ebs8(b0_acc)

                # ── C00 += A0 x B0 ──────────────────────────────
                acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
                res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, MAC_CONF
                )

                # ── Load & convert B1 (vshuffle) ────────────────
                vb1: aie_vector[bf16, 64] = load_v(wt_buf + b1_off, 64)
                b1_off = b1_off + B_K_STRIDE

                b1_i32: aie_vector[i32, 32] = vector_cast(vb1, i32, 32)
                b1_lo_i: aie_vector[i32, 16] = vector_extract(b1_i32, 0, 16)
                b1_hi_i: aie_vector[i32, 16] = vector_extract(b1_i32, 16, 16)
                b1_even: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 52)
                b1_odd: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 53)
                b1_cat: aie_vector[i32, 32] = concat(b1_even, b1_odd)
                vb1_s: aie_vector[bf16, 64] = vector_cast(b1_cat, bf16, 64)
                b1_s_lo: aie_vector[bf16, 32] = vector_extract(vb1_s, 0, 32)
                b1_s_hi: aie_vector[bf16, 32] = vector_extract(vb1_s, 32, 32)
                b1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1_s_lo)
                b1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1_s_hi)
                b1_acc: aie_vector[f32, 64] = concat(b1_acc_lo, b1_acc_hi)
                b1_mant, b1_exp = v64accfloat_to_v64bfp16ebs8(b1_acc)

                # ── C01 += A0 x B1 ──────────────────────────────
                acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
                res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, MAC_CONF
                )

                # ── Load & convert A1 (no vshuffle) ─────────────
                va1: aie_vector[bf16, 64] = load_v(in_buf + a1_off, 64)
                a1_off = a1_off + A_K_STRIDE

                a1_lo: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
                a1_hi: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
                a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
                a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
                a1_acc: aie_vector[f32, 64] = concat(a1_acc_lo, a1_acc_hi)
                a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

                # ── C10 += A1 x B0,  C11 += A1 x B1 ─────────────
                acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
                res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, MAC_CONF
                )
                acc_i11: aie_vector[i32, 64] = vector_cast(acc_c11, i32, 64)
                res11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b1_mant, b1_exp, acc_i11, MAC_CONF
                )

                acc_c00 = vector_cast(res00, f32, 64)
                acc_c10 = vector_cast(res10, f32, 64)
                acc_c01 = vector_cast(res01, f32, 64)
                acc_c11 = vector_cast(res11, f32, 64)

                k = k + 1

            store_v(out_buf + c00_off, acc_c00)
            store_v(out_buf + c10_off, acc_c10)
            store_v(out_buf + c01_off, acc_c01)
            store_v(out_buf + c11_off, acc_c11)

            n = n + 2
        m = m + 2

    event1()


# Extra globals for the @aie_kernel compiler (these intrinsics are not in
# PythocKernel's default global set; see CONVERSION_PATTERN.md gotcha #2).
KERNEL_EXTRA_GLOBALS = {
    "vector_extract": vector_extract,
    "vector_cast": vector_cast,
    "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
    "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
    "vshuffle": vshuffle,
    "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
    "set_ctrl_reg": set_ctrl_reg,
    "prepare_for_pipelining": prepare_for_pipelining,
    "MAC_CONF": 780,
}


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDV6 1x1 conv as PythoC + IRON GEMM (bf16, single core)",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument(
        "-M", type=int, default=M,
        help=f"Pixels (H*W). Currently fixed at {M} for single-tile demo.",
    )
    parser.add_argument(
        "-K", type=int, default=K,
        help=f"Input channels. Currently fixed at {K}.",
    )
    parser.add_argument(
        "-N", type=int, default=N,
        help=f"Output channels. Currently fixed at {N}.",
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── IRON program ────────────────────────────────────────────────────────────


def build_mlir_module(device):
    """One compute tile, three host-driven ObjectFifos (In, W, Out)."""
    in_ty  = np.ndarray[(A_ELEMS,), np.dtype[np.uint16]]   # bf16 as uint16
    wt_ty  = np.ndarray[(B_ELEMS,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]

    in_host_ty  = np.ndarray[(M * K,), np.dtype[np.uint16]]
    wt_host_ty  = np.ndarray[(K * N,), np.dtype[np.uint16]]
    out_host_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    kernel = PythocKernel(
        gemm_conv1x1_kernel,
        [in_ty, wt_ty, out_ty],
        extra_globals=KERNEL_EXTRA_GLOBALS,
    )

    of_in  = ObjectFifo(in_ty,  depth=2, name="inIn")
    of_wt  = ObjectFifo(wt_ty,  depth=2, name="inWt")
    of_out = ObjectFifo(out_ty, depth=2, name="outOut")

    def core_fn(of_in, of_wt, of_out, kernel):
        for _ in range_(0xFFFFFFFF):
            ein  = of_in.acquire(1)
            ewt  = of_wt.acquire(1)
            eout = of_out.acquire(1)
            kernel(ein, ewt, eout)
            of_in.release(1)
            of_wt.release(1)
            of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_wt.cons(), of_out.prod(), kernel],
        stack_size=0xD00,
    )

    runtime = Runtime()
    with runtime.sequence(in_host_ty, wt_host_ty, out_host_ty) as (a, b, c):
        runtime.start(worker)
        runtime.fill(of_in.prod(),  a)
        runtime.fill(of_wt.prod(),  b)
        runtime.drain(of_out.cons(), c, wait=True)

    program = Program(device, runtime)
    module = program.resolve_program()
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── Host tiling helpers (match bf16_gemm_single_core.py) ────────────────────


def bf16_to_uint16(x):
    flat = x.flatten().astype(np.float32)
    return (flat.view(np.uint32) >> 16).astype(np.uint16)


def uint16_to_float(x):
    flat = x.flatten().astype(np.uint32)
    return (flat << 16).view(np.float32)


def tile_matrix_a(A_bf16, M, K):
    """A[M,K] -> A_tiled[K_MICRO, M_BLOCKS, 8, 8] flat."""
    M_B = M // 8
    K_B = K // 8
    A = A_bf16.reshape(M, K)
    tiled = A.reshape(M_B, 8, K_B, 8).transpose(2, 0, 1, 3)
    return tiled.reshape(-1)


def tile_matrix_b(B_bf16, K, N):
    """B[K,N] -> B_tiled[N_BLOCKS, K_MICRO, 8, 8] flat."""
    K_B = K // 8
    N_B = N // 8
    B = B_bf16.reshape(K, N)
    tiled = B.reshape(K_B, 8, N_B, 8).transpose(2, 0, 1, 3)
    return tiled.reshape(-1)


def untile_matrix_c(C_tiled_flat, M, N):
    """C_tiled[N_BLOCKS, M_BLOCKS, 8, 8] -> C[M,N]."""
    M_B = M // 8
    N_B = N // 8
    tiled = C_tiled_flat.reshape(N_B, M_B, 8, 8)
    return tiled.transpose(1, 2, 0, 3).reshape(M, N)


# ── Run on NPU ──────────────────────────────────────────────────────────────


def run_with_xrt(xclbin_path: Path, insts_path: Path):
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(npu_kernel)

    # Conv1x1 = matmul: Out[M,N] = In[M,K] @ W[K,N]
    np.random.seed(42)
    In_f32 = np.random.randn(M, K).astype(np.float32) * 0.1
    W_f32  = np.random.randn(K, N).astype(np.float32) * 0.1

    In_bf16_flat = bf16_to_uint16(In_f32)
    W_bf16_flat  = bf16_to_uint16(W_f32)

    In_tiled = tile_matrix_a(In_bf16_flat, M, K)
    W_tiled  = tile_matrix_b(W_bf16_flat,  K, N)

    in_buf  = iron.tensor(In_tiled, dtype=np.uint16)
    wt_buf  = iron.tensor(W_tiled,  dtype=np.uint16)
    out_buf = iron.zeros(M * N, dtype=np.float32)

    DefaultNPURuntime.run(handle, [in_buf, wt_buf, out_buf])

    # f32 reference using bf16-rounded inputs
    In_ref = uint16_to_float(In_bf16_flat).reshape(M, K)
    W_ref  = uint16_to_float(W_bf16_flat).reshape(K, N)
    C_ref  = In_ref @ W_ref

    C_npu = untile_matrix_c(np.array(out_buf.numpy()), M, N)
    return C_npu, C_ref


def main():
    args = parse_args()
    if (args.M, args.K, args.N) != (M, K, N):
        print(f"FAILED: only M=K=N={M} supported in this single-tile demo "
              f"(got M={args.M}, K={args.K}, N={args.N})")
        return 1

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    device = NPU2Col1()

    try:
        print(f"[1/3] Building IRON program (M={M}, K={K}, N={N})")
        module = build_mlir_module(device)
        mlir_path = work_dir / "kernel.mlir"
        with open(mlir_path, "w") as fh:
            print(module, file=fh)
        print(f"      -> {mlir_path}")

        print("[2/3] Compiling design with aiecc")
        insts_path  = work_dir / "insts.bin"
        xclbin_path = work_dir / "final.xclbin"
        compile_mlir_module(
            mlir_module=module,
            insts_path=str(insts_path),
            xclbin_path=str(xclbin_path),
            work_dir=str(work_dir),
            verbose=args.verbose,
        )
        print(f"      -> {xclbin_path}")

        print("[3/3] Running on NPU and validating results")
        actual, expected = run_with_xrt(xclbin_path, insts_path)

        max_err = np.max(np.abs(actual - expected))
        rel_err = max_err / (np.max(np.abs(expected)) + 1e-10)
        print(f"      Max absolute error: {max_err:.6f}")
        print(f"      Max relative error: {rel_err:.6f}")

        if rel_err < 0.05:  # 5% bf16->bfp16 quantisation tolerance
            print("PASS!")
            return 0
        print(f"FAILED: relative error {rel_err:.4f} > 5% tolerance")
        return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
