#!/usr/bin/env python3
# bf16_gemm_single_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai_npu2
#
# RUN: %python %s --device npu2 --work-dir ./bf16_gemm_build | FileCheck %s
# CHECK: PASS!

"""BF16 GEMM on a single AIE2p core using PythoC + IRON.

Implements C[M,N] += A[M,K] × B[K,N] with bf16 inputs and f32 accumulation.
The PythoC kernel uses the same BFP16 hardware intrinsic path as the
compiler-generated version in mlir-air/programming_examples/matrix_multiplication/bf16.

Architecture (see aie-kernel-optimization-guide.md for details):
  - BF16 inputs are converted to BFP16 (block floating point) via:
      bf16 → accfloat (v32bf16_to_v32accfloat)
      accfloat → bfp16ebs8 (v64accfloat_to_v64bfp16ebs8)
  - B matrix is rearranged via vshuffle (modes 52/53) for the 8×8T layout
  - Hardware 8×8×8T matmul: BFP576_BFP576_ACC2048_mac_conf (config=780)
  - 2×2 register blocking: 4 accumulators share 2 A + 2 B loads

Tile sizes (single core, matching aie.air.mlir per-core buffers):
  A tile: 16×4 blocks of 8×8 bf16 = 128 M × 32 K = 4096 bf16
  B tile: 4×8 blocks of 8×8 bf16 =  32 K × 64 N = 2048 bf16
  C tile: 16×8 blocks of 8×8 f32 = 128 M × 64 N = 8192 f32
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
from aie.iron.placers import SequentialPlacer
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary

# PythoC types and intrinsics
from pythoc import ptr, i8, i32, f32, bf16, void
from pythoc.aie import (
    load_v,
    store_v,
    aie_vector,
    zeros,
    vector_extract,
    concat,
    vector_cast,
    # Hardware intrinsics (auto-resolved from intrinsic_registry.json)
    v32bf16_to_v32accfloat,
    v64accfloat_to_v64bfp16ebs8,
    vshuffle,
    BFP576_BFP576_ACC2048_mac_conf,
    set_ctrl_reg,
    prepare_for_pipelining,
)
from pythoc.aie.profiling import event0, event1

# Extra globals needed by the @aie_kernel compiler.
# The PythocKernel inline compiler has a fixed set of built-in names (load_v,
# store_v, zeros, concat, etc.) but does NOT include vector_extract,
# vector_cast, or the generic hardware intrinsics. We pass them explicitly.
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

# ── Tile dimensions ──────────────────────────────────────────────────────────
#
# Sized to fit in a single compute tile's 64KB L1 data memory (no mem-tile).
# With depth=2 ping-pong:  A=2KB×2 + B=2KB×2 + C=4KB×2 + stack=3.3KB ≈ 19KB.
#
# The full bf16 GEMM case study uses 128M×32K×64N per core (needs mem-tile).
# This demo uses 32×32×32 which exercises the same 2×2 register blocking,
# bf16→bfp16 conversion, vshuffle, and BFP576 MAC intrinsic chain.
#
# Layout: A[K_MICRO, M_BLOCKS, 8, 8], B[N_BLOCKS, K_MICRO, 8, 8],
#         C[N_BLOCKS, M_BLOCKS, 8, 8]

TILE_M = 32  # M dimension of output tile
TILE_N = 32  # N dimension of output tile
TILE_K = 32  # K (reduction) dimension

M_BLOCKS = 4  # TILE_M / 8
N_BLOCKS = 4  # TILE_N / 8
K_MICRO = 4  # TILE_K / 8
BLOCK = 64  # 8 × 8 elements per block
MAC_CONF = 780  # sgn_x=1, sgn_y=1, amode=2, bmode=1

# Element sizes for buffer types
A_ELEMS = TILE_M * TILE_K  # 1024 bf16
B_ELEMS = TILE_K * TILE_N  # 1024 bf16
C_ELEMS = TILE_M * TILE_N  # 8192 f32

# For single-core demo: full matrix = one tile
M = TILE_M
N = TILE_N
K = TILE_K

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "bf16_gemm_build"


# ── Main compute kernel ─────────────────────────────────────────────────────
#
# C[128×64] += A[128×32] × B[32×64], all in tiled 8×8-block layout.
#
# Buffer layouts (matching aie.air.mlir):
#   A: [K_MICRO=4][M_BLOCKS=16][8][8]  → A[k,m] at offset (k*M_BLOCKS + m)*64
#   B: [N_BLOCKS=8][K_MICRO=4][8][8]   → B[n,k] at offset (n*K_MICRO + k)*64
#   C: [N_BLOCKS=8][M_BLOCKS=16][8][8] → C[n,m] at offset (n*M_BLOCKS + m)*64
#
# The kernel uses 2×2 register blocking (step 2 in both M and N):
#   C[n,m]   += A[m,k]   × B[n,k]       (C00 += A0 × B0)
#   C[n,m+1] += A[m+1,k] × B[n,k]       (C10 += A1 × B0)  ← A1 new, B0 reused
#   C[n+1,m] += A[m,k]   × B[n+1,k]     (C01 += A0 × B1)  ← A0 reused, B1 new
#   C[n+1,m+1] += A[m+1,k] × B[n+1,k]   (C11 += A1 × B1)  ← both reused


@aie_kernel
def bf16_gemm_kernel(
    a_buf: ptr[bf16, True],  # A tile: 1024 bf16 elems [K_MICRO, M_BLOCKS, 8, 8]
    b_buf: ptr[bf16, True],  # B tile: 1024 bf16 elems [N_BLOCKS, K_MICRO, 8, 8]
    c_buf: ptr[f32, True],  # C tile: 1024 f32 elems  [N_BLOCKS, M_BLOCKS, 8, 8]
) -> void:
    """BF16 GEMM tile kernel: C += A × B via BFP16 hardware matmul.

    Generates LLVM IR equivalent to matmul_seg_core_0_2.ll inner loops.
    Uses 2×2 register blocking for 2× operand reuse.

    Tile: 32M × 32K × 32N  (M_BLOCKS=4, N_BLOCKS=4, K_MICRO=4)
    """
    # Set rounding modes (matches LL: set.ctrl.reg(9,1) and (1,12))
    # Reg 9 = conv round mode: 1 = round-to-nearest-even
    # Reg 1 = MAC saturation/rounding config
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)

    # event0()

    # Buffer strides (in elements)
    # A layout: [K_MICRO=4, M_BLOCKS=4, 8, 8] → A[k,m] at (k*M_BLOCKS + m)*64
    # B layout: [N_BLOCKS=4, K_MICRO=4, 8, 8] → B[n,k] at (n*K_MICRO + k)*64
    # C layout: [N_BLOCKS=4, M_BLOCKS=4, 8, 8] → C[n,m] at (n*M_BLOCKS + m)*64
    A_K_STRIDE: i32 = 256  # M_BLOCKS * BLOCK = 4 * 64
    B_K_STRIDE: i32 = 64  # BLOCK
    B_N_STRIDE: i32 = 256  # K_MICRO * BLOCK = 4 * 64
    C_M_STRIDE: i32 = 64  # BLOCK
    C_N_STRIDE: i32 = 256  # M_BLOCKS * BLOCK = 4 * 64

    # ── Zero-initialize entire C buffer ──────────────────────────────
    # Matches the reference which zeroes C in a separate loop nest
    # before any compute, preventing opt from folding store+load.
    z: aie_vector[f32, 64] = zeros(f32, 64)
    zi: i32 = 0
    while zi < 16:  # N_BLOCKS * M_BLOCKS = 4 * 4 = 16
        store_v(c_buf + zi * 64, z)
        zi = zi + 1

    event0()

    # ── 2×2 register-blocked GEMM ─────────────────────────────────
    # Outer loops: M (step 2) × N (step 2) over output blocks
    # Inner loop: K-micro (4 iterations) over reduction dimension

    m: i32 = 0
    while m < 4:  # M_BLOCKS, step 2
        n: i32 = 0
        while n < 4:  # N_BLOCKS, step 2
            # Compute C buffer offsets for 2×2 tile
            c00_off: i32 = n * C_N_STRIDE + m * C_M_STRIDE
            c10_off: i32 = c00_off + C_M_STRIDE
            c01_off: i32 = c00_off + C_N_STRIDE
            c11_off: i32 = c00_off + C_N_STRIDE + C_M_STRIDE

            # Load accumulators from pre-zeroed C buffer
            acc_c00: aie_vector[f32, 64] = load_v(c_buf + c00_off, 64)
            acc_c10: aie_vector[f32, 64] = load_v(c_buf + c10_off, 64)
            acc_c01: aie_vector[f32, 64] = load_v(c_buf + c01_off, 64)
            acc_c11: aie_vector[f32, 64] = load_v(c_buf + c11_off, 64)

            # K-micro reduction loop — use incrementing pointer offsets
            # (matches reference IR pattern: phi-carried offsets with
            # simple add strides, instead of recomputing k*STRIDE)
            a0_off: i32 = m * 64  # A[k=0, m]
            a1_off: i32 = a0_off + 64  # A[k=0, m+1]
            b0_off: i32 = n * B_N_STRIDE  # B[n, k=0]
            b1_off: i32 = (n + 1) * B_N_STRIDE  # B[n+1, k=0]

            k: i32 = 0
            while k < 4:  # K_MICRO
                # ── Interleaved load/convert/MAC (matches reference)
                # At each MAC only 2 BFP pairs live, not 3.

                # ── Load A0, Load B0 ───────────────────────────────
                va0: aie_vector[bf16, 64] = load_v(a_buf + a0_off, 64)
                a0_off = a0_off + A_K_STRIDE
                vb0: aie_vector[bf16, 64] = load_v(b_buf + b0_off, 64)
                b0_off = b0_off + B_K_STRIDE

                # ── Convert A0 (no vshuffle) ───────────────────────
                a0_lo: aie_vector[bf16, 32] = vector_extract(va0, 0, 32)
                a0_hi: aie_vector[bf16, 32] = vector_extract(va0, 32, 32)
                a0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_lo)
                a0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_hi)
                a0_acc: aie_vector[f32, 64] = concat(a0_acc_lo, a0_acc_hi)

                # ── Convert B0 (vshuffle path) ─────────────────────
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

                # ── BFP16 convert A0, B0 ───────────────────────────
                a0_mant, a0_exp = v64accfloat_to_v64bfp16ebs8(a0_acc)
                b0_mant, b0_exp = v64accfloat_to_v64bfp16ebs8(b0_acc)

                # ── MAC: C00 += A0 × B0 ───────────────────────────
                acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
                res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, MAC_CONF
                )

                # ── Load & convert B1 (vshuffle) ──────────────────
                vb1: aie_vector[bf16, 64] = load_v(b_buf + b1_off, 64)
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

                # ── MAC: C01 += A0 × B1  (A0 reused) ─────────────
                acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
                res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, MAC_CONF
                )

                # ── Load & convert A1 (no vshuffle) ───────────────
                va1: aie_vector[bf16, 64] = load_v(a_buf + a1_off, 64)
                a1_off = a1_off + A_K_STRIDE

                a1_lo: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
                a1_hi: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
                a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
                a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
                a1_acc: aie_vector[f32, 64] = concat(a1_acc_lo, a1_acc_hi)
                a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

                # ── MAC: C10 += A1 × B0, C11 += A1 × B1 ──────────
                acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
                res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, MAC_CONF
                )
                acc_i11: aie_vector[i32, 64] = vector_cast(acc_c11, i32, 64)
                res11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b1_mant, b1_exp, acc_i11, MAC_CONF
                )

                # Bitcast results back to f32 for next iteration
                acc_c00 = vector_cast(res00, f32, 64)
                acc_c10 = vector_cast(res10, f32, 64)
                acc_c01 = vector_cast(res01, f32, 64)
                acc_c11 = vector_cast(res11, f32, 64)

                k = k + 1

            # Store 4 accumulated blocks back to C
            store_v(c_buf + c00_off, acc_c00)
            store_v(c_buf + c10_off, acc_c10)
            store_v(c_buf + c01_off, acc_c01)
            store_v(c_buf + c11_off, acc_c11)

            n = n + 2
        m = m + 2

    event1()


# ── IRON integration ─────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="BF16 GEMM on single AIE2p core with PythoC + IRON",
    )
    parser.add_argument("--device", choices=("npu2",), default="npu2")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Generate MLIR and compile but skip NPU execution",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark (10 warmup + 20 measurement)",
    )
    parser.add_argument(
        "--trace-size",
        type=lambda x: int(x, 0),
        default=0,
        metavar="BYTES",
        help="Enable event tracing; size of DDR trace buffer in bytes "
        "(e.g. 0x20000). 0 = disabled (default)",
    )
    return parser.parse_args()


def build_mlir_module(device, trace_size=0):
    """Build IRON program: shim → mem-tile → compute tile → mem-tile → shim."""

    # NumPy types for IRON ObjectFifos
    a_ty = np.ndarray[(A_ELEMS,), np.dtype[np.uint16]]  # bf16 as uint16
    b_ty = np.ndarray[(B_ELEMS,), np.dtype[np.uint16]]  # bf16 as uint16
    c_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]

    # Flat host buffers (one tile each for single-core demo)
    A_host_ty = np.ndarray[(M * K,), np.dtype[np.uint16]]
    B_host_ty = np.ndarray[(K * N,), np.dtype[np.uint16]]
    C_host_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    # PythocKernel wrapping the inline @aie_kernel
    kernel = PythocKernel(
        bf16_gemm_kernel,
        [a_ty, b_ty, c_ty],
        extra_globals=KERNEL_EXTRA_GLOBALS,
    )

    # ObjectFifos: host ↔ compute tile (single core, no mem-tile split)
    of_a = ObjectFifo(a_ty, depth=2, name="inA")
    of_b = ObjectFifo(b_ty, depth=2, name="inB")
    of_c = ObjectFifo(c_ty, depth=2, name="outC")

    # Core function: acquire buffers, call kernel, release
    def core_fn(of_a, of_b, of_c, kernel):
        for _ in range_(0xFFFFFFFF):
            elem_a = of_a.acquire(1)
            elem_b = of_b.acquire(1)
            elem_c = of_c.acquire(1)

            kernel(elem_a, elem_b, elem_c)

            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [of_a.cons(), of_b.cons(), of_c.prod(), kernel],
        stack_size=0xD00,
        trace=1 if trace_size > 0 else None,
    )

    runtime = Runtime()
    with runtime.sequence(A_host_ty, B_host_ty, C_host_ty) as (a_in, b_in, c_out):
        if trace_size > 0:
            runtime.enable_trace(trace_size, workers=[worker])
        runtime.start(worker)
        runtime.fill(of_a.prod(), a_in)
        runtime.fill(of_b.prod(), b_in)
        runtime.drain(of_c.cons(), c_out, wait=True)

    program = Program(device, runtime)
    module = program.resolve_program(SequentialPlacer())
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


def bf16_to_uint16(x):
    """Convert float32 array to bf16 stored as uint16 (vectorized)."""
    # Reinterpret float32 bits, shift right 16 to get upper 16 bits (bf16)
    flat = x.flatten().astype(np.float32)
    return (flat.view(np.uint32) >> 16).astype(np.uint16)


def uint16_to_float(x):
    """Convert bf16 (as uint16) back to float32 (vectorized)."""
    flat = x.flatten().astype(np.uint32)
    return (flat << 16).view(np.float32)


def tile_matrix_a(A_bf16, M, K):
    """Tile A[M,K] → A_tiled[K_MICRO, M_BLOCKS, 8, 8] (flattened).

    Kernel layout: A[k,m] at offset (k*M_BLOCKS + m)*64
    where k = k_block index, m = m_block index.
    A_tiled[k, m, r, c] = A[m*8+r, k*8+c]
    """
    M_B = M // 8
    K_B = K // 8
    A = A_bf16.reshape(M, K)
    # Reshape to [M_B, 8, K_B, 8] then transpose to [K_B, M_B, 8, 8]
    tiled = A.reshape(M_B, 8, K_B, 8).transpose(2, 0, 1, 3)
    return tiled.reshape(-1)


def tile_matrix_b(B_bf16, K, N):
    """Tile B[K,N] → B_tiled[N_BLOCKS, K_MICRO, 8, 8] (flattened).

    B blocks are stored row-major within each 8×8 tile. The kernel
    uses vshuffle(52/53) to rearrange for the transposed MAC operand.

    B_tiled[n, k, r, c] = B[k*8+r, n*8+c]
    """
    K_B = K // 8
    N_B = N // 8
    B = B_bf16.reshape(K, N)
    # [K_B, 8, N_B, 8] → [N_B, K_B, 8, 8] (row-major inner)
    tiled = B.reshape(K_B, 8, N_B, 8).transpose(2, 0, 1, 3)
    return tiled.reshape(-1)


def untile_matrix_c(C_tiled_flat, M, N):
    """Untile C_tiled[N_BLOCKS, M_BLOCKS, 8, 8] → C[M,N].

    C_tiled[n, m, r, c] = C[m*8+r, n*8+c]
    """
    M_B = M // 8
    N_B = N // 8
    tiled = C_tiled_flat.reshape(N_B, M_B, 8, 8)
    # Transpose to [M_B, 8, N_B, 8] then reshape to [M, N]
    return tiled.transpose(1, 2, 0, 3).reshape(M, N)


# ── Performance benchmark ────────────────────────────────────────────────────
#
# Replicates the methodology from mlir-air/programming_examples/
# matrix_multiplication/bf16/test.cpp:
#   - 10 warmup iterations (discarded)
#   - 20 measurement iterations
#   - Wall-clock timing per iteration (kernel dispatch + wait + sync)
#   - Metric: GFLOPS = 2*M*K*N / (time_us * 1000)

WARMUP_ITERS = 10
MEASURE_ITERS = 20


def run_benchmark(kernel_handle, A_tiled, B_tiled):
    """Run performance benchmark and report GFLOPS."""
    import time

    macs = 2.0 * M * K * N
    total_iters = WARMUP_ITERS + MEASURE_ITERS
    times_us = []

    print(f"[3/3] Benchmarking: {M}×{K}×{N} bf16 GEMM, single core")
    print(f"      {WARMUP_ITERS} warmup + {MEASURE_ITERS} measurement iterations")

    for i in range(total_iters):
        in_a = iron.tensor(A_tiled, dtype=np.uint16)
        in_b = iron.tensor(B_tiled, dtype=np.uint16)
        out_c = iron.zeros(M * N, dtype=np.float32)

        t0 = time.perf_counter()
        DefaultNPURuntime.run(kernel_handle, [in_a, in_b, out_c])
        t1 = time.perf_counter()

        elapsed_us = (t1 - t0) * 1e6
        if i >= WARMUP_ITERS:
            times_us.append(elapsed_us)

    avg_us = sum(times_us) / len(times_us)
    min_us = min(times_us)
    max_us = max(times_us)

    avg_gflops = macs / (avg_us * 1000)
    max_gflops = macs / (min_us * 1000)
    min_gflops = macs / (max_us * 1000)

    print()
    print(f"      Avg latency: {avg_us:10.1f} us  ->  {avg_gflops:.4f} GFLOPS")
    print(f"      Min latency: {min_us:10.1f} us  ->  {max_gflops:.4f} GFLOPS (peak)")
    print(f"      Max latency: {max_us:10.1f} us  ->  {min_gflops:.4f} GFLOPS")
    return 0


def main():
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    device = NPU2Col1()

    try:
        trace_size = args.trace_size
        print(
            f"[1/3] Building IRON program with PythoC bf16 GEMM kernel"
            + (f" [trace={trace_size:#x}]" if trace_size else "")
        )
        module = build_mlir_module(device, trace_size=trace_size)
        mlir_path = work_dir / "kernel.mlir"
        with open(mlir_path, "w") as f:
            print(module, file=f)
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
        print(f"      -> {xclbin_path}")

        if args.compile_only:
            print("PASS! (compile-only)")
            return 0

        # Generate test data
        np.random.seed(42)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.1
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.1

        A_bf16_flat = bf16_to_uint16(A_f32)
        B_bf16_flat = bf16_to_uint16(B_f32)

        # Pre-tile inputs to match kernel's expected block layout
        A_tiled = tile_matrix_a(A_bf16_flat, M, K)
        B_tiled = tile_matrix_b(B_bf16_flat, K, N)

        # Load kernel
        trace_config = (
            TraceConfig(trace_size, trace_file=str(work_dir / "trace.txt"))
            if trace_size > 0
            else None
        )
        npu_kernel = NPUKernel(
            str(xclbin_path),
            str(insts_path),
            kernel_name="MLIR_AIE",
            trace_config=trace_config,
        )
        kernel_handle = DefaultNPURuntime.load(npu_kernel)

        if args.benchmark:
            return run_benchmark(kernel_handle, A_tiled, B_tiled)

        # ── Validation run ───────────────────────────────────────────────
        print("[3/3] Running on NPU and validating results")

        # Reference: compute in f32 using bf16-rounded inputs
        A_ref = uint16_to_float(A_bf16_flat).reshape(M, K)
        B_ref = uint16_to_float(B_bf16_flat).reshape(K, N)
        C_ref = A_ref @ B_ref

        in_a = iron.tensor(A_tiled, dtype=np.uint16)
        in_b = iron.tensor(B_tiled, dtype=np.uint16)
        out_c = iron.zeros(M * N, dtype=np.float32)

        if trace_config:
            run_args = [in_a, in_b, out_c]
            DefaultNPURuntime.prepare_args_for_trace(run_args, trace_config)
            DefaultNPURuntime.run(kernel_handle, run_args)
            trace_buf, _ = DefaultNPURuntime.extract_trace_from_args(
                run_args, trace_config
            )
            DefaultNPURuntime.process_trace(trace_buf, None, trace_config)
        else:
            DefaultNPURuntime.run(kernel_handle, [in_a, in_b, out_c])

        # Untile output C from [N_BLOCKS, M_BLOCKS, 8, 8] → [M, N]
        C_npu = untile_matrix_c(np.array(out_c.numpy()), M, N)

        # Validate with tolerance (bf16 precision ~1e-2 relative)
        max_err = np.max(np.abs(C_npu - C_ref))
        rel_err = max_err / (np.max(np.abs(C_ref)) + 1e-10)
        print(f"      Max absolute error: {max_err:.6f}")
        print(f"      Max relative error: {rel_err:.6f}")

        if rel_err < 0.05:  # 5% tolerance for bf16→bfp16 path
            print("PASS!")
            if trace_config:
                trace_json = work_dir / "trace_mlir.json"
                print(f"\n[trace] Parsing trace → {trace_json}")
                trace_config.trace_to_json(str(mlir_path), str(trace_json))
                print("[trace] Cycle summary (event0→event1):")
                cycles = get_cycles_summary(str(trace_json))
                for entry in cycles:
                    name, vals = entry[0], entry[1:]
                    if vals:
                        print(
                            f"  {name}: {len(vals)} invocations  "
                            f"first={vals[0]}  min={min(vals)}  "
                            f"avg={sum(vals)//len(vals)}  max={max(vals)} cycles"
                        )
                    else:
                        print(f"  {name}: no complete invocations captured")
            return 0
        else:
            print(f"FAILED: relative error {rel_err:.4f} > 5% tolerance")
            return 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
