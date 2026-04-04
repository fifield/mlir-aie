#!/usr/bin/env python3
# bf16_gemm_multi_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

"""Multi-core BF16 GEMM using PythoC + IRON.

Implements C[M,N] = A[M,K] × B[K,N] with bf16 inputs and f32 accumulation,
distributed across a 4×N AIE core array (default N=4, configurable).

This adapts whole_array_iron.py's multi-core ObjectFifo topology to use
PythoC kernels (from bf16_gemm_single_core.py) instead of pre-compiled .o files.

Architecture:
  - 4 rows × n_aie_cols columns of compute cores
  - A: split/distribute across rows via ObjectFifo
  - B: forward/broadcast to columns via ObjectFifo
  - C: join from rows via ObjectFifo
  - Each core computes (m, n) output tiles with K//k k-reduction steps
  - BFP16 emulation: bf16 → accfloat → bfp16ebs8, with 8×8×8 MAC

Buffer layouts in L1 (set by dims_to_stream):
  A: [M_BLOCKS, K_MICRO, 8, 8]  (M_BLOCKS = m/8, K_MICRO = k/8)
  B: [K_MICRO, N_BLOCKS, 8, 8]  (N_BLOCKS = n/8)
  C: [M_BLOCKS, N_BLOCKS, 8, 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils.compile import compile_mlir_module
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.helpers.taplib import TensorTiler2D

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

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "bf16_gemm_multi_core_build"

# Fixed microkernel dimensions for bf16 with bfp16 emulation on NPU2
R, S, T = 8, 8, 8

N_AIE_ROWS = 4  # always 4 rows in NPU2

WARMUP_ITERS = 10
MEASURE_ITERS = 20


# ── PythoC Kernel: bf16 GEMM tile ───────────────────────────────────────────
#
# Accumulates C += A × B for one (m, k) × (k, n) tile.
# All *_CONST names are injected via extra_globals at PythocKernel compile time.
#
# Buffer layouts (produced by dims_to_stream DMA retiling):
#   A: [M_BLOCKS, K_MICRO, 8, 8]  →  A[m_blk, k_blk] at (m*K_MICRO + k)*64
#   B: [K_MICRO, N_BLOCKS, 8, 8]  →  B[k_blk, n_blk] at (k*N_BLOCKS + n)*64
#   C: [M_BLOCKS, N_BLOCKS, 8, 8] →  C[m_blk, n_blk] at (m*N_BLOCKS + n)*64


@aie_kernel
def bf16_gemm_tile_kernel(
    a_buf: ptr[bf16, True],
    b_buf: ptr[bf16, True],
    c_buf: ptr[f32, True],
) -> void:
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)
    event0()

    m: i32 = 0
    while m < M_BLOCKS_CONST:
        n: i32 = 0
        while n < N_BLOCKS_CONST:
            # C block offsets for 2×2 register tile
            c00_off: i32 = m * C_M_STRIDE_CONST + n * C_N_STRIDE_CONST
            c10_off: i32 = c00_off + C_M_STRIDE_CONST
            c01_off: i32 = c00_off + C_N_STRIDE_CONST
            c11_off: i32 = c00_off + C_M_STRIDE_CONST + C_N_STRIDE_CONST

            # Load existing C accumulators (zeroed by zero kernel initially)
            acc_c00: aie_vector[f32, 64] = load_v(c_buf + c00_off, 64)
            acc_c10: aie_vector[f32, 64] = load_v(c_buf + c10_off, 64)
            acc_c01: aie_vector[f32, 64] = load_v(c_buf + c01_off, 64)
            acc_c11: aie_vector[f32, 64] = load_v(c_buf + c11_off, 64)

            # Incrementing pointer offsets (phi-carried pattern)
            # Matches reference IR: simple add strides each iteration
            a0_off: i32 = m * A_M_STRIDE_CONST  # A[m, k=0]
            a1_off: i32 = a0_off + A_M_STRIDE_CONST  # A[m+1, k=0]
            b0_off: i32 = n * B_N_STRIDE_CONST  # B[k=0, n]
            b1_off: i32 = (n + 1) * B_N_STRIDE_CONST  # B[k=0, n+1]

            k: i32 = 0
            while k < K_MICRO_CONST:
                # ── Load A0, Load B0 ───────────────────────────────
                va0: aie_vector[bf16, 64] = load_v(a_buf + a0_off, 64)
                a0_off = a0_off + A_K_STRIDE_CONST
                vb0: aie_vector[bf16, 64] = load_v(b_buf + b0_off, 64)
                b0_off = b0_off + B_K_STRIDE_CONST

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

                # ── C00 += A0 × B0 ────────────────────────────────
                acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
                res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, MAC_CONF
                )

                # ── Load & convert B1 (vshuffle) ──────────────────
                vb1: aie_vector[bf16, 64] = load_v(b_buf + b1_off, 64)
                b1_off = b1_off + B_K_STRIDE_CONST

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

                # ── C01 += A0 × B1   (A0 reused) ─────────────────
                acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
                res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, MAC_CONF
                )

                # ── Load & convert A1 (no vshuffle) ───────────────
                va1: aie_vector[bf16, 64] = load_v(a_buf + a1_off, 64)
                a1_off = a1_off + A_K_STRIDE_CONST

                a1_lo: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
                a1_hi: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
                a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
                a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
                a1_acc: aie_vector[f32, 64] = concat(a1_acc_lo, a1_acc_hi)
                a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

                # ── C10 += A1 × B0   (B0 reused, then dies) ───────
                acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
                res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, MAC_CONF
                )

                # ── C11 += A1 × B1   (B1 reused, then dies) ───────
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

            # Store accumulated blocks back to C
            store_v(c_buf + c00_off, acc_c00)
            store_v(c_buf + c10_off, acc_c10)
            store_v(c_buf + c01_off, acc_c01)
            store_v(c_buf + c11_off, acc_c11)

            n = n + 2
        m = m + 2

    event1()


# ── PythoC Kernel: zero C buffer ────────────────────────────────────────────


@aie_kernel
def bf16_zero_kernel(c_buf: ptr[f32, True]) -> void:
    i: i32 = 0
    zero_vec: aie_vector[f32, 64] = zeros(f32, 64)
    while i < C_ELEMS_CONST:
        store_v(c_buf + i, zero_vec)
        i = i + 64


# ── IRON program builder ────────────────────────────────────────────────────


def ceildiv(a, b):
    return (a + b - 1) // b


def build_mlir_module(M, K, N, m, k, n, n_aie_cols):
    """Build IRON multi-core GEMM program following whole_array_iron.py topology."""

    n_aie_rows = N_AIE_ROWS
    n_aie_cores = n_aie_rows * n_aie_cols
    r, s, t = R, S, T

    # ── Validate dimensions ──────────────────────────────────────────────
    assert (
        M % (m * n_aie_rows) == 0
    ), f"M={M} must be divisible by m*n_aie_rows={m * n_aie_rows}"
    assert K % k == 0, f"K={K} must be divisible by k={k}"
    assert (
        N % (n * n_aie_cols) == 0
    ), f"N={N} must be divisible by n*n_aie_cols={n * n_aie_cols}"
    assert m % r == 0, f"m={m} must be divisible by r={r}"
    assert k % s == 0, f"k={k} must be divisible by s={s}"
    assert n % t == 0, f"n={n} must be divisible by t={t}"

    # ── Derived constants ────────────────────────────────────────────────
    M_BLOCKS = m // r
    N_BLOCKS = n // t
    K_MICRO = k // s

    fifo_depth = 2
    c_fifo_depth = 1  # C stays resident as accumulator, not double-buffered
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    # Shim/mem tile allocation for A (same logic as whole_array_iron.py)
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    dev_ty = NPU2()

    # ── Tensor types ─────────────────────────────────────────────────────
    dtype_in = bfloat16
    dtype_out = np.float32

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # ── PythoC kernels ───────────────────────────────────────────────────
    # Stride constants for the L1 tiled layout produced by dims_to_stream:
    #   A: [M_BLOCKS, K_MICRO, 8, 8] → A_M_STRIDE = K_MICRO*64, A_K_STRIDE = 64
    #   B: [K_MICRO, N_BLOCKS, 8, 8] → B_K_STRIDE = N_BLOCKS*64, B_N_STRIDE = 64
    #   C: [M_BLOCKS, N_BLOCKS, 8, 8] → C_M_STRIDE = N_BLOCKS*64, C_N_STRIDE = 64
    gemm_extra_globals = {
        "vector_extract": vector_extract,
        "vector_cast": vector_cast,
        "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
        "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
        "vshuffle": vshuffle,
        "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
        "set_ctrl_reg": set_ctrl_reg,
        "prepare_for_pipelining": prepare_for_pipelining,
        "MAC_CONF": 780,
        "M_BLOCKS_CONST": M_BLOCKS,
        "N_BLOCKS_CONST": N_BLOCKS,
        "K_MICRO_CONST": K_MICRO,
        "A_M_STRIDE_CONST": K_MICRO * 64,
        "A_K_STRIDE_CONST": 64,
        "B_K_STRIDE_CONST": N_BLOCKS * 64,
        "B_N_STRIDE_CONST": 64,
        "C_M_STRIDE_CONST": N_BLOCKS * 64,
        "C_N_STRIDE_CONST": 64,
    }
    matmul_kernel = PythocKernel(
        bf16_gemm_tile_kernel,
        [A_l1_ty, B_l1_ty, C_l1_ty],
        extra_globals=gemm_extra_globals,
    )

    zero_extra_globals = {
        "C_ELEMS_CONST": m * n,
    }
    zero_kernel = PythocKernel(
        bf16_zero_kernel,
        [C_l1_ty],
        extra_globals=zero_extra_globals,
    )

    # ── Tile declarations ────────────────────────────────────────────────
    tiles = [[(col, row) for col in range(n_aie_cols)] for row in range(6)]
    core_tiles = tiles[2:]

    # ── ObjectFifo topology ──────────────────────────────────────────────
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A: split/distribute across rows
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(2 * i if n_aie_cols == 8 else i, 1),
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B: forward/broadcast to columns
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        b_dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=b_dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        # Output C: join from rows
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[c_fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # ── Worker function ──────────────────────────────────────────────────
    def core_fn(in_a, in_b, out_c, zero_k, matmul_k):
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero_k(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul_k(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    # ── Set up compute tiles ─────────────────────────────────────────────
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        zero_kernel,
                        matmul_kernel,
                    ],
                    placement=Tile(tile_col, tile_row),
                    stack_size=0xD00,
                )
            )

    # ── Runtime sequence ─────────────────────────────────────────────────
    tb_max_n_rows = 4
    tb_n_rows = min(tb_max_n_rows // 2, M // (m * n_aie_rows))

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N),
        (k, n),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True,
        prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)

        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                        placement=Tile(col, 0),
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)

                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                                placement=Tile(2 * col if n_aie_cols == 8 else col, 0),
                            )

                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                            placement=Tile(col, 0),
                        )

                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    # ── Build and verify ─────────────────────────────────────────────────
    my_program = Program(dev_ty, rt)
    module = my_program.resolve_program(SequentialPlacer())
    assert module.operation.verify(), "Generated MLIR failed verification"
    return module


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-core BF16 GEMM with PythoC + IRON",
    )
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-k", type=int, default=32)
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
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
    return parser.parse_args()


# ── Host data helpers ────────────────────────────────────────────────────────


def bf16_to_uint16(x):
    """Convert float32 array to bf16 stored as uint16 (vectorized)."""
    flat = x.flatten().astype(np.float32)
    return (flat.view(np.uint32) >> 16).astype(np.uint16)


def uint16_to_float(x):
    """Convert bf16 (as uint16) back to float32 (vectorized)."""
    flat = x.flatten().astype(np.uint32)
    return (flat << 16).view(np.float32)


# ── Benchmark ────────────────────────────────────────────────────────────────


def run_benchmark(kernel_handle, A_bf16, B_bf16, M, K, N):
    """Run performance benchmark and report GFLOPS."""
    import time

    macs = 2.0 * M * K * N
    total_iters = WARMUP_ITERS + MEASURE_ITERS
    times_us = []
    n_aie_cores = N_AIE_ROWS * 4  # default; informational only

    print(f"[3/3] Benchmarking: {M}×{K}×{N} bf16 GEMM, multi-core")
    print(f"      {WARMUP_ITERS} warmup + {MEASURE_ITERS} measurement iterations")

    for i in range(total_iters):
        in_a = iron.tensor(A_bf16, dtype=bfloat16)
        in_b = iron.tensor(B_bf16, dtype=bfloat16)
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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    M, K, N = args.M, args.K, args.N
    m, k, n = args.m, args.k, args.n
    n_aie_cols = args.n_aie_cols
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(
            f"[1/3] Building IRON program: {M}×{K}×{N} bf16 GEMM, "
            f"{N_AIE_ROWS}×{n_aie_cols} cores, tile {m}×{k}×{n}"
        )
        module = build_mlir_module(M, K, N, m, k, n, n_aie_cols)
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

        # ── Generate test data ───────────────────────────────────────────
        np.random.seed(42)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.1
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.1

        # Convert to bfloat16 (host data stays flat row-major;
        # ObjectFifo dims_to_stream handles retiling into kernel layout)
        A_bf16 = A_f32.astype(bfloat16).flatten()
        B_bf16 = B_f32.astype(bfloat16).flatten()

        trace_config = None
        npu_kernel = NPUKernel(
            str(xclbin_path),
            str(insts_path),
            kernel_name="MLIR_AIE",
        )
        kernel_handle = DefaultNPURuntime.load(npu_kernel)

        if args.benchmark:
            return run_benchmark(kernel_handle, A_bf16, B_bf16, M, K, N)

        # ── Validation run ───────────────────────────────────────────────
        print("[3/3] Running on NPU and validating results")

        # Reference: f32 matmul of bf16-rounded inputs
        A_ref = A_bf16.astype(np.float32).reshape(M, K)
        B_ref = B_bf16.astype(np.float32).reshape(K, N)
        C_ref = A_ref @ B_ref

        in_a = iron.tensor(A_bf16, dtype=bfloat16)
        in_b = iron.tensor(B_bf16, dtype=bfloat16)
        out_c = iron.zeros(M * N, dtype=np.float32)

        DefaultNPURuntime.run(kernel_handle, [in_a, in_b, out_c])

        # Output is row-major (dims_to_stream on C handles retiling)
        C_npu = np.array(out_c.numpy()).reshape(M, N)

        max_err = np.max(np.abs(C_npu - C_ref))
        rel_err = max_err / (np.max(np.abs(C_ref)) + 1e-10)
        print(f"      Max absolute error: {max_err:.6f}")
        print(f"      Max relative error: {rel_err:.6f}")

        if rel_err < 0.05:  # 5% tolerance for bf16→bfp16 path
            print("PASS!")
        else:
            print(f"FAILED: relative error {rel_err:.4f} > 5% tolerance")
            return 1

        return 0

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
