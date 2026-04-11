#!/usr/bin/env python3
# flash_attention_kernel_fusion.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

"""AIE2P flash attention using direct AIE dialect Python bindings.

This emitter mirrors the lowered kernel-fusion AIR design in
`mlir-air/programming_examples/flash_attention/kernel_fusion_based`.

Unlike the existing dataflow-based direct emitter, this variant models the
fully fused 32-core layout from the generated `aie.air.mlir` / `npu.air.mlir`:

  - 2 head segments laid out side-by-side across columns 0-3 and 4-7
  - 4 query tiles per head
  - 4 cascade stages per query tile
  - fused per-tile QK + softmax + GV compute using `attn.cc`
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from shutil import copy2, which

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import memref, vector
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    buffer,
    cascade_flow,
    core,
    device,
    dma_bd,
    dma_start,
    external_buffer,
    external_func,
    flow,
    get_cascade,
    lock,
    mem,
    memtile_dma,
    next_bd,
    put_cascade,
    shim_dma_allocation,
    tile,
    use_lock,
)
from aie.dialects.aiex import (
    EndOp,
    bds,
    dma_await_task,
    dma_configure_task_for,
    dma_free_task,
    dma_start_task,
    runtime_sequence,
)
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.iron.pythoc import PythocKernel
from aie.ir import AffineDimExpr, AffineMap, MemRefType
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from pythoc.aie import ACC2048_accfloat_add_conf, BFP576_BFP576_ACC2048_mac_conf, I1024_I1024_ACC2048_bf_mul_conf, I512_I512_ACC1024_bf_mac_conf, I512_I512_ACC1024_bf_mul_conf, I512_I512_ACC1024_bf_negmul_conf, acc_extract, acc_grow, concat, getExpBf16, set_ctrl_reg, v32accfloat_to_v32bf16, v32bf16_to_v32accfloat, v64accfloat_to_v64bfp16ebs8, vector_add, vector_blend, vector_cast, vector_extract, vector_insert, vector_mul, vector_sub, vmax_ltbf16, vshuffle

from attn import (
    add_gp_g_pythoc,
    accum_sp_r_s_pythoc,
    copy_tile_pythoc,
    div_gp_sp_pythoc,
    exp_up_minus_u_pythoc,
    matmul_g_b_bf16_pythoc,
    maximum_up_u_bf16_pythoc,
    mul_r_gp_pythoc,
    neg_inf_fill_up_bf16_pythoc,
    vector_copy_32elems_pythoc,
    zero_fill_g_bf16_pythoc,
    zero_fill_gp_bf16_pythoc,
    zero_fill_sp_bf16_pythoc,
)


FLASH_ATTN_KERNEL_GLOBALS = {
    "ACC2048_accfloat_add_conf": ACC2048_accfloat_add_conf,
    "BFP576_BFP576_ACC2048_mac_conf": BFP576_BFP576_ACC2048_mac_conf,
    "I1024_I1024_ACC2048_bf_mul_conf": I1024_I1024_ACC2048_bf_mul_conf,
    "I512_I512_ACC1024_bf_mac_conf": I512_I512_ACC1024_bf_mac_conf,
    "I512_I512_ACC1024_bf_mul_conf": I512_I512_ACC1024_bf_mul_conf,
    "I512_I512_ACC1024_bf_negmul_conf": I512_I512_ACC1024_bf_negmul_conf,
    "acc_extract": acc_extract,
    "acc_grow": acc_grow,
    "concat": concat,
    "getExpBf16": getExpBf16,
    "set_ctrl_reg": set_ctrl_reg,
    "v32accfloat_to_v32bf16": v32accfloat_to_v32bf16,
    "v32bf16_to_v32accfloat": v32bf16_to_v32accfloat,
    "v64accfloat_to_v64bfp16ebs8": v64accfloat_to_v64bfp16ebs8,
    "vector_add": vector_add,
    "vector_blend": vector_blend,
    "vector_cast": vector_cast,
    "vector_extract": vector_extract,
    "vector_insert": vector_insert,
    "vector_mul": vector_mul,
    "vector_sub": vector_sub,
    "vmax_ltbf16": vmax_ltbf16,
    "vshuffle": vshuffle,
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "flash_attention_kernel_fusion_build"
REFERENCE_AIE_MLIR = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "kernel_fusion_based"
    / "build_peano"
    / "air_project"
    / "aie.air.mlir"
)
REFERENCE_NPU_MLIR = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "kernel_fusion_based"
    / "build_peano"
    / "air_project"
    / "npu.air.mlir"
)
KERNEL_SOURCE = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "kernel_fusion_based"
    / "attn.cc"
)
REFERENCE_KERNEL_OBJECT = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "kernel_fusion_based"
    / "build_peano"
    / "attn.o"
)
KERNEL_OBJECT = "attn.o"


# ---------------------------------------------------------------------------
# Fixed lowered configuration
# ---------------------------------------------------------------------------
NUM_HEADS = 2
LK = 512
LKP = 64
LQ = 512
DK = 64
DV = 64

NUM_Q_TILES = 4
NUM_CASCADE_STAGES = 4
NUM_SEGMENTS = 2
Q_GROUPS = 2

Q_TILE_ROWS = 64
Q_ROWS_PER_GROUP = NUM_Q_TILES * Q_TILE_ROWS
CHUNKS_PER_STAGE = LK // (LKP * NUM_CASCADE_STAGES)

QK_TILE_SIZE = Q_TILE_ROWS * DK
KV_TILE_SIZE = LKP * DK
V_TILE_SIZE = LKP * DV
G_TILE_SIZE = Q_TILE_ROWS * LKP
OUTPUT_TILE_SIZE = Q_TILE_ROWS * DV
HEAD_Q_SIZE = LQ * DK
HEAD_KV_SIZE = LK * DK
HEAD_OUT_SIZE = LQ * DV
ROW_SIZE = Q_TILE_ROWS


# ---------------------------------------------------------------------------
# NumPy golden reference
# ---------------------------------------------------------------------------
def flash_attention_golden(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    lkp: int = LKP,
) -> np.ndarray:
    dt = bfloat16
    out = np.zeros((NUM_HEADS, LQ, DV), dtype=dt)

    for head in range(NUM_HEADS):
        q_head = Q[head].astype(np.float32)
        k_head = K[head].astype(np.float32)
        v_head = V[head].astype(np.float32)

        gp = np.zeros((LQ, DV), dtype=np.float32)
        up = np.full((LQ, 1), -np.inf, dtype=np.float32)
        sp = np.zeros((LQ, 1), dtype=np.float32)

        for chunk in range(LK // lkp):
            k_chunk = k_head[chunk * lkp : (chunk + 1) * lkp, :]
            v_chunk = v_head[chunk * lkp : (chunk + 1) * lkp, :]

            g = (q_head @ k_chunk.T) / sqrt(DK)
            u = np.max(g, axis=-1, keepdims=True)
            u = np.maximum(u, up)
            g = np.exp(g - u).astype(np.float32)
            r = np.exp(up - u).astype(np.float32)

            gp = gp * r
            gp = (g @ v_chunk + gp).astype(np.float32)

            s = np.sum(g, axis=-1, keepdims=True).astype(np.float32)
            s = (sp * r + s).astype(np.float32)
            sp = s
            up = u

        out[head] = (gp / sp).astype(dt)

    return out


def generate_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = bfloat16
    q = (
        np.arange(NUM_HEADS * LQ * DK, dtype=np.float32).reshape(NUM_HEADS, LQ, DK)
        / float(NUM_HEADS * LQ * DK)
        * 2.0
    ).astype(dt)
    k = (
        np.arange(NUM_HEADS * LK * DK, dtype=np.float32).reshape(NUM_HEADS, LK, DK)
        / float(NUM_HEADS * LK * DK)
        * 2.0
    ).astype(dt)
    v = (
        np.arange(NUM_HEADS * LK * DV, dtype=np.float32).reshape(NUM_HEADS, LK, DV)
        / float(NUM_HEADS * LK * DV)
        * 2.0
    ).astype(dt)
    return q, k, v


@dataclass(frozen=True)
class KernelSet:
    zero_fill_g: object
    zero_fill_gp: object
    zero_fill_sp: object
    neg_inf_fill_up: object
    copy_tile: object
    matmul_a_b: object
    fused_softmax: object
    mul_r_gp: object
    matmul_g_b: object
    accum_sp_r_s: object
    vector_copy_32: object
    maximum_up_u: object
    exp_up_minus_u: object
    add_gp_g: object
    div_gp_sp: object


@dataclass(frozen=True)
class MemTileSpec:
    segment: int
    index: int
    tile: object
    qk: object
    v: object
    out: object
    out_wait: object
    out_ready: object
    qk_wait: object
    qk_ready: object
    v_wait: object
    v_ready: object


@dataclass(frozen=True)
class ComputeTileSpec:
    segment: int
    stage: int
    q_col: int
    tile: object
    qk: object
    q: object
    v: object
    g: object
    gp: object
    up: object
    sp: object
    s: object
    r: object
    merged_gp: object | None
    merged_up: object | None
    merged_sp: object | None
    prev_up: object | None
    r_from_cascade: object | None
    r_from_local: object | None
    tmp_sp: object | None
    out_dma_acquire: object | None
    out_ready: object | None
    qk_dma_acquire: object
    qk_ready: object
    v_dma_acquire: object
    v_ready: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flash_attention_kernel_fusion.py",
        description="AIE2P flash attention kernel-fusion direct AIE dialect emitter",
    )
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--lk", type=int, default=LK)
    parser.add_argument("--lkp", type=int, default=LKP)
    parser.add_argument("--lq", type=int, default=LQ)
    parser.add_argument("--dk", type=int, default=DK)
    parser.add_argument("--dv", type=int, default=DV)
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print generated MLIR module and exit",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Generate MLIR and compile but skip NPU execution",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark (warmup + timed iterations)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Directory for generated MLIR artifacts",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def ensure_supported_configuration(args: argparse.Namespace) -> None:
    fixed = (NUM_HEADS, LK, LKP, LQ, DK, DV)
    given = (args.num_heads, args.lk, args.lkp, args.lq, args.dk, args.dv)
    if given != fixed:
        raise ValueError(
            "The direct kernel-fusion emitter currently mirrors the fixed lowered "
            f"configuration only: num_heads={NUM_HEADS}, lk={LK}, lkp={LKP}, "
            f"lq={LQ}, dk={DK}, dv={DV}."
        )
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")


def declare_kernels(
    qk_ty: type[np.ndarray],
    q_ty: type[np.ndarray],
    v_ty: type[np.ndarray],
    g_flat_ty: type[np.ndarray],
    gp_ty: type[np.ndarray],
    row_ty: type[np.ndarray],
) -> KernelSet:
    copy_tile_kernel = PythocKernel(
        copy_tile_pythoc,
        [qk_ty, q_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    neg_inf_fill_up_kernel = PythocKernel(
        neg_inf_fill_up_bf16_pythoc,
        [row_ty],
        target_arch="aie2p",
    )
    vector_copy_32_kernel = PythocKernel(
        vector_copy_32elems_pythoc,
        [np.int32, row_ty, row_ty],
        target_arch="aie2p",
    )
    mul_r_gp_kernel = PythocKernel(
        mul_r_gp_pythoc,
        [row_ty, gp_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    exp_up_minus_u_kernel = PythocKernel(
        exp_up_minus_u_pythoc,
        [row_ty, row_ty, row_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    maximum_up_u_kernel = PythocKernel(
        maximum_up_u_bf16_pythoc,
        [row_ty, row_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    add_gp_g_kernel = PythocKernel(
        add_gp_g_pythoc,
        [gp_ty, gp_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    accum_sp_r_s_kernel = PythocKernel(
        accum_sp_r_s_pythoc,
        [row_ty, row_ty, row_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    div_gp_sp_kernel = PythocKernel(
        div_gp_sp_pythoc,
        [row_ty, gp_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    matmul_g_b_kernel = PythocKernel(
        matmul_g_b_bf16_pythoc,
        [g_flat_ty, v_ty, gp_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    zero_fill_g_kernel = PythocKernel(
        zero_fill_g_bf16_pythoc,
        [g_flat_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    zero_fill_gp_kernel = PythocKernel(
        zero_fill_gp_bf16_pythoc,
        [gp_ty],
        target_arch="aie2p",
        extra_globals=FLASH_ATTN_KERNEL_GLOBALS,
    )
    zero_fill_sp_kernel = PythocKernel(
        zero_fill_sp_bf16_pythoc,
        [row_ty],
        target_arch="aie2p",
    )

    return KernelSet(
        zero_fill_g=external_func(
            "zero_fill_g_bf16_pythoc",
            inputs=[g_flat_ty],
            link_with=zero_fill_g_kernel.object_file_name,
        ),
        zero_fill_gp=external_func(
            "zero_fill_gp_bf16_pythoc",
            inputs=[gp_ty],
            link_with=zero_fill_gp_kernel.object_file_name,
        ),
        zero_fill_sp=external_func(
            "zero_fill_sp_bf16_pythoc",
            inputs=[row_ty],
            link_with=zero_fill_sp_kernel.object_file_name,
        ),
        neg_inf_fill_up=external_func(
            "neg_inf_fill_up_bf16_pythoc",
            inputs=[row_ty],
            link_with=neg_inf_fill_up_kernel.object_file_name,
        ),
        copy_tile=external_func(
            "copy_tile_pythoc",
            inputs=[qk_ty, q_ty],
            link_with=copy_tile_kernel.object_file_name,
        ),
        matmul_a_b=external_func(
            "matmul_a_b_bf16", inputs=[q_ty, qk_ty, g_flat_ty], link_with=KERNEL_OBJECT
        ),
        fused_softmax=external_func(
            "fused_softmax", inputs=[g_flat_ty, row_ty, row_ty, row_ty], link_with=KERNEL_OBJECT
        ),
        mul_r_gp=external_func(
            "mul_r_gp_pythoc",
            inputs=[row_ty, gp_ty],
            link_with=mul_r_gp_kernel.object_file_name,
        ),
        matmul_g_b=external_func(
            "matmul_g_b_bf16_pythoc",
            inputs=[g_flat_ty, v_ty, gp_ty],
            link_with=matmul_g_b_kernel.object_file_name,
        ),
        accum_sp_r_s=external_func(
            "accum_sp_r_s_pythoc",
            inputs=[row_ty, row_ty, row_ty],
            link_with=accum_sp_r_s_kernel.object_file_name,
        ),
        vector_copy_32=external_func(
            "vector_copy_32elems_pythoc",
            inputs=[np.int32, row_ty, row_ty],
            link_with=vector_copy_32_kernel.object_file_name,
        ),
        maximum_up_u=external_func(
            "maximum_up_u_bf16_pythoc",
            inputs=[row_ty, row_ty],
            link_with=maximum_up_u_kernel.object_file_name,
        ),
        exp_up_minus_u=external_func(
            "exp_up_minus_u_pythoc",
            inputs=[row_ty, row_ty, row_ty],
            link_with=exp_up_minus_u_kernel.object_file_name,
        ),
        add_gp_g=external_func(
            "add_gp_g_pythoc",
            inputs=[gp_ty, gp_ty],
            link_with=add_gp_g_kernel.object_file_name,
        ),
        div_gp_sp=external_func(
            "div_gp_sp_pythoc",
            inputs=[row_ty, gp_ty],
            link_with=div_gp_sp_kernel.object_file_name,
        ),
    )


def collapsed_memref_type(buffer_ref: object, total_elems: int) -> MemRefType:
    buffer_ty = MemRefType(buffer_ref.type)
    return MemRefType.get(
        (total_elems,),
        buffer_ty.element_type,
        None,
        buffer_ty.memory_space,
    )


def emit_cascade_send(buffer_ref: object, total_elems: int, zero_bf16: object, c0: object) -> None:
    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
    flat = memref.collapse_shape(
        collapsed_memref_type(buffer_ref, total_elems),
        buffer_ref,
        [[0, 1]],
    )
    for offset in range_(0, total_elems, 32):
        chunk = memref.subview(flat, [offset], [32], [1])
        value = vector.transfer_read(
            T.vector(32, T.bf16()),
            chunk,
            [c0],
            permutation_map=perm,
            padding=zero_bf16,
            in_bounds=[True],
        )
        put_cascade(value)


def emit_cascade_receive(buffer_ref: object, total_elems: int, c0: object) -> None:
    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
    flat = memref.collapse_shape(
        collapsed_memref_type(buffer_ref, total_elems),
        buffer_ref,
        [[0, 1]],
    )
    for offset in range_(0, total_elems, 32):
        chunk = memref.subview(flat, [offset], [32], [1])
        value = get_cascade(T.vector(32, T.bf16()))
        vector.transfer_write(
            None,
            value,
            chunk,
            [c0],
            permutation_map=perm,
            in_bounds=[True],
        )


def emit_reference_flows(
    shim_tiles: dict[int, object],
    mem_tiles: dict[int, object],
    compute_tiles: dict[tuple[int, int], object],
) -> None:
    for segment in range(NUM_SEGMENTS):
        base = segment * 4

        for idx in range(4):
            flow(shim_tiles[base + idx], WireBundle.DMA, 0, mem_tiles[base + idx], WireBundle.DMA, 0)
            flow(shim_tiles[base + idx], WireBundle.DMA, 1, mem_tiles[base + idx], WireBundle.DMA, 1)
            flow(mem_tiles[base + idx], WireBundle.DMA, 0, shim_tiles[base + idx], WireBundle.DMA, 0)

        for stage in range(NUM_CASCADE_STAGES):
            mem_tile_ref = mem_tiles[base + stage]
            row = 2 + stage
            for q_col in range(NUM_Q_TILES):
                compute_tile = compute_tiles[(base + q_col, row)]
                flow(mem_tile_ref, WireBundle.DMA, 1, compute_tile, WireBundle.DMA, 0)
                flow(mem_tile_ref, WireBundle.DMA, 2, compute_tile, WireBundle.DMA, 1)

        for q_col in range(NUM_Q_TILES):
            flow(
                compute_tiles[(base + q_col, 2)],
                WireBundle.DMA,
                0,
                mem_tiles[base + q_col],
                WireBundle.DMA,
                2,
            )

        for q_col in range(NUM_Q_TILES):
            for stage in range(NUM_CASCADE_STAGES - 1, 0, -1):
                cascade_flow(
                    compute_tiles[(base + q_col, 2 + stage)],
                    compute_tiles[(base + q_col, 1 + stage)],
                )


def emit_reference_shim_allocations(shim_tiles: dict[int, object]) -> dict[str, list[list[object]]]:
    qk_allocs: list[list[object]] = []
    v_allocs: list[list[object]] = []
    out_allocs: list[list[object]] = []

    for segment in range(NUM_SEGMENTS):
        base = segment * 4
        qk_allocs.append(
            [
                shim_dma_allocation(
                    f"air_QKIn_{stage}_{segment}_0_0",
                    shim_tiles[base + stage],
                    DMAChannelDir.MM2S,
                    0,
                )
                for stage in range(NUM_CASCADE_STAGES)
            ]
        )
        v_allocs.append(
            [
                shim_dma_allocation(
                    f"air_VIn_{stage}_{segment}_0_0",
                    shim_tiles[base + stage],
                    DMAChannelDir.MM2S,
                    1,
                )
                for stage in range(NUM_CASCADE_STAGES)
            ]
        )
        out_allocs.append(
            [
                shim_dma_allocation(
                    f"air_channel_0_{segment}_0_{q_col}",
                    shim_tiles[base + q_col],
                    DMAChannelDir.S2MM,
                    0,
                )
                for q_col in range(NUM_Q_TILES)
            ]
        )

    return {"qk": qk_allocs, "v": v_allocs, "out": out_allocs}


def emit_memtile_dma(spec: MemTileSpec) -> None:
    @memtile_dma(spec.tile)
    def memtile_body(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(spec.out_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.out, offset=0, len=OUTPUT_TILE_SIZE)
            use_lock(spec.out_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(spec.qk_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.qk, offset=0, len=QK_TILE_SIZE, dimensions=[(8, 8), (64, 64), (8, 1)])
            use_lock(spec.qk_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.MM2S, 2, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(spec.v_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.v, offset=0, len=V_TILE_SIZE, dimensions=[(8, 8), (64, 64), (8, 1)])
            use_lock(spec.v_ready, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[8])
        with block[7]:
            use_lock(spec.qk_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.qk, offset=0, len=QK_TILE_SIZE)
            use_lock(spec.qk_wait, LockAction.Release, value=1)
            next_bd(block[7])
        with block[8]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[9], chain=block[10])
        with block[9]:
            use_lock(spec.v_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.v, offset=0, len=V_TILE_SIZE)
            use_lock(spec.v_wait, LockAction.Release, value=1)
            next_bd(block[9])
        with block[10]:
            dma_start(DMAChannelDir.S2MM, 2, dest=block[11], chain=block[12])
        with block[11]:
            use_lock(spec.out_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.out, offset=0, len=OUTPUT_TILE_SIZE)
            use_lock(spec.out_wait, LockAction.Release, value=1)
            next_bd(block[11])
        with block[12]:
            EndOp()


def emit_compute_mem(spec: ComputeTileSpec) -> None:
    @mem(spec.tile)
    def tile_dma(block):
        if spec.stage == 0:
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(spec.out_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(
                    spec.merged_gp,
                    offset=0,
                    len=OUTPUT_TILE_SIZE,
                    dimensions=[(64, 8), (8, 512), (8, 1)],
                )
                use_lock(spec.out_ready, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(spec.qk_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(spec.qk, offset=0, len=QK_TILE_SIZE)
                use_lock(spec.qk_ready, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[5], chain=block[6])
            with block[5]:
                use_lock(spec.v_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(spec.v, offset=0, len=V_TILE_SIZE)
                use_lock(spec.v_ready, LockAction.Release, value=1)
                next_bd(block[5])
            with block[6]:
                EndOp()
            return

        dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(spec.qk_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.qk, offset=0, len=QK_TILE_SIZE)
            use_lock(spec.qk_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(spec.v_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.v, offset=0, len=V_TILE_SIZE)
            use_lock(spec.v_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            EndOp()


def emit_selective_q_capture(spec: ComputeTileSpec, kernels: KernelSet) -> None:
    for q_index in range(NUM_Q_TILES):
        use_lock(spec.qk_ready, LockAction.AcquireGreaterEqual, value=1)
        if q_index == spec.q_col:
            kernels.copy_tile(spec.qk, spec.q)
        use_lock(spec.qk_dma_acquire, LockAction.Release, value=1)


def emit_compute_core(spec: ComputeTileSpec, kernels: KernelSet) -> None:
    @core(spec.tile)
    def compute_core_body():
        c0 = arith.constant(0, index=True)
        c0_i32 = arith.constant(0, T.i32())
        zero_bf16 = arith.constant(0.0, T.bf16())
        g_flat_ty = collapsed_memref_type(spec.g, G_TILE_SIZE)

        for _ in range_(sys.maxsize):
            if spec.stage == 0:
                use_lock(spec.out_ready, LockAction.AcquireGreaterEqual, value=1)

            kernels.zero_fill_gp(spec.gp)
            kernels.zero_fill_sp(spec.sp)
            kernels.neg_inf_fill_up(spec.up)

            emit_selective_q_capture(spec, kernels)

            for _ in range_(CHUNKS_PER_STAGE):
                g_flat = memref.collapse_shape(g_flat_ty, spec.g, [[0, 1]])
                kernels.zero_fill_g(g_flat)

                use_lock(spec.qk_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.matmul_a_b(spec.q, spec.qk, g_flat)
                use_lock(spec.qk_dma_acquire, LockAction.Release, value=1)

                use_lock(spec.v_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.fused_softmax(g_flat, spec.up, spec.s, spec.r)
                kernels.mul_r_gp(spec.r, spec.gp)
                kernels.matmul_g_b(g_flat, spec.v, spec.gp)
                kernels.accum_sp_r_s(spec.sp, spec.r, spec.s)
                kernels.vector_copy_32(c0_i32, spec.s, spec.sp)
                use_lock(spec.v_dma_acquire, LockAction.Release, value=1)

            if spec.stage == NUM_CASCADE_STAGES - 1:
                emit_cascade_send(spec.gp, OUTPUT_TILE_SIZE, zero_bf16, c0)
                emit_cascade_send(spec.up, ROW_SIZE, zero_bf16, c0)
                emit_cascade_send(spec.sp, ROW_SIZE, zero_bf16, c0)
                continue

            emit_cascade_receive(spec.merged_gp, OUTPUT_TILE_SIZE, c0)
            emit_cascade_receive(spec.merged_up, ROW_SIZE, c0)
            emit_cascade_receive(spec.merged_sp, ROW_SIZE, c0)

            kernels.vector_copy_32(c0_i32, spec.up, spec.prev_up)
            kernels.maximum_up_u(spec.merged_up, spec.up)
            kernels.exp_up_minus_u(spec.merged_up, spec.up, spec.r_from_cascade)
            kernels.exp_up_minus_u(spec.prev_up, spec.up, spec.r_from_local)
            kernels.mul_r_gp(spec.r_from_cascade, spec.merged_gp)
            kernels.mul_r_gp(spec.r_from_local, spec.gp)
            kernels.add_gp_g(spec.gp, spec.merged_gp)
            kernels.zero_fill_sp(spec.tmp_sp)
            kernels.accum_sp_r_s(spec.merged_sp, spec.r_from_cascade, spec.tmp_sp)
            kernels.accum_sp_r_s(spec.sp, spec.r_from_local, spec.tmp_sp)
            kernels.vector_copy_32(c0_i32, spec.tmp_sp, spec.merged_sp)

            if spec.stage == 0:
                kernels.div_gp_sp(spec.merged_sp, spec.merged_gp)
                use_lock(spec.out_dma_acquire, LockAction.Release, value=1)
                continue

            emit_cascade_send(spec.merged_gp, OUTPUT_TILE_SIZE, zero_bf16, c0)
            emit_cascade_send(spec.up, ROW_SIZE, zero_bf16, c0)
            emit_cascade_send(spec.merged_sp, ROW_SIZE, zero_bf16, c0)


def emit_runtime_sequence(
    allocations: dict[str, list[list[object]]],
    q_host_ty: type[np.ndarray],
    k_host_ty: type[np.ndarray],
    v_host_ty: type[np.ndarray],
    out_host_ty: type[np.ndarray],
) -> None:
    @runtime_sequence(q_host_ty, k_host_ty, v_host_ty, out_host_ty)
    def attention_bf16(q, k, v, out):
        head_q_offset = HEAD_Q_SIZE
        head_kv_offset = HEAD_KV_SIZE
        head_out_offset = HEAD_OUT_SIZE
        q_group_offset = Q_ROWS_PER_GROUP * DK
        out_group_offset = Q_ROWS_PER_GROUP * DV
        stage_kv_offset = CHUNKS_PER_STAGE * LKP * DK
        out_tile_offset = Q_TILE_ROWS * DV

        for q_group in range(Q_GROUPS):
            q_tasks: dict[tuple[int, int], object] = {}
            k_tasks: dict[tuple[int, int], object] = {}
            v_tasks: dict[tuple[int, int], object] = {}
            out_tasks: dict[tuple[int, int], object] = {}

            for segment in range(NUM_SEGMENTS):
                q_offset = segment * head_q_offset + q_group * q_group_offset
                kv_base_offset = segment * head_kv_offset
                out_base_offset = segment * head_out_offset + q_group * out_group_offset

                for stage in range(NUM_CASCADE_STAGES):
                    qk_task = dma_configure_task_for(allocations["qk"][segment][stage])
                    with bds(qk_task) as bd:
                        with bd[0]:
                            dma_bd(
                                q,
                                offset=q_offset,
                                len=Q_ROWS_PER_GROUP * DK,
                                dimensions=[(32, 512), (512, 1)],
                            )
                            EndOp()
                    dma_start_task(qk_task)
                    q_tasks[(segment, stage)] = qk_task

                    k_task = dma_configure_task_for(allocations["qk"][segment][stage])
                    with bds(k_task) as bd:
                        with bd[0]:
                            dma_bd(
                                k,
                                offset=kv_base_offset + stage * stage_kv_offset,
                                len=CHUNKS_PER_STAGE * LKP * DK,
                                dimensions=[(16, 512), (512, 1)],
                            )
                            EndOp()
                    dma_start_task(k_task)
                    k_tasks[(segment, stage)] = k_task

                for stage in range(NUM_CASCADE_STAGES):
                    v_task = dma_configure_task_for(allocations["v"][segment][stage])
                    with bds(v_task) as bd:
                        with bd[0]:
                            dma_bd(
                                v,
                                offset=kv_base_offset + stage * stage_kv_offset,
                                len=CHUNKS_PER_STAGE * LKP * DV,
                                dimensions=[(16, 512), (512, 1)],
                            )
                            EndOp()
                    dma_start_task(v_task)
                    v_tasks[(segment, stage)] = v_task

                for q_col in range(NUM_Q_TILES):
                    out_task = dma_configure_task_for(
                        allocations["out"][segment][q_col],
                        issue_token=True,
                    )
                    with bds(out_task) as bd:
                        with bd[0]:
                            dma_bd(
                                out,
                                offset=out_base_offset + q_col * out_tile_offset,
                                len=OUTPUT_TILE_SIZE,
                                dimensions=[(8, 512), (512, 1)],
                            )
                            EndOp()
                    dma_start_task(out_task)
                    out_tasks[(segment, q_col)] = out_task

            # Match the AIR-lowered task lifetime ordering. The shared QK/V shim
            # channels are reused within a group, so teardown order matters.
            dma_free_task(v_tasks[(0, 1)])
            dma_free_task(v_tasks[(0, 3)])
            dma_await_task(out_tasks[(0, 1)])
            dma_await_task(out_tasks[(0, 3)])

            dma_free_task(v_tasks[(1, 1)])
            dma_free_task(v_tasks[(1, 3)])
            dma_await_task(out_tasks[(1, 1)])
            dma_await_task(out_tasks[(1, 3)])

            for stage in range(NUM_CASCADE_STAGES):
                dma_free_task(q_tasks[(0, stage)])
                dma_free_task(k_tasks[(0, stage)])
            for stage in range(NUM_CASCADE_STAGES):
                dma_free_task(q_tasks[(1, stage)])
                dma_free_task(k_tasks[(1, stage)])

            dma_await_task(out_tasks[(1, 2)])
            dma_await_task(out_tasks[(1, 0)])
            dma_free_task(v_tasks[(1, 2)])
            dma_free_task(v_tasks[(1, 0)])

            dma_await_task(out_tasks[(0, 2)])
            dma_await_task(out_tasks[(0, 0)])
            dma_free_task(v_tasks[(0, 2)])
            dma_free_task(v_tasks[(0, 0)])


def build_mlir_module() -> object:
    q_host_ty = np.ndarray[(NUM_HEADS, LQ, DK), np.dtype[bfloat16]]
    k_host_ty = np.ndarray[(NUM_HEADS, LK, DK), np.dtype[bfloat16]]
    v_host_ty = np.ndarray[(NUM_HEADS, LK, DV), np.dtype[bfloat16]]
    out_host_ty = np.ndarray[(NUM_HEADS, LQ, DV), np.dtype[bfloat16]]

    tile_ty = np.ndarray[(Q_TILE_ROWS, DK), np.dtype[bfloat16]]
    v_tile_ty = np.ndarray[(LKP, DV), np.dtype[bfloat16]]
    gp_ty = np.ndarray[(Q_TILE_ROWS, DV), np.dtype[bfloat16]]
    row_ty = np.ndarray[(Q_TILE_ROWS, 1), np.dtype[bfloat16]]
    g_tile_ty = np.ndarray[(Q_TILE_ROWS, LKP), np.dtype[bfloat16]]
    g_flat_ty = np.ndarray[(G_TILE_SIZE,), np.dtype[bfloat16]]
    l2_tile_ty = np.ndarray[(Q_TILE_ROWS, DK), np.dtype[bfloat16]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            external_buffer(q_host_ty, name="Q")
            external_buffer(k_host_ty, name="K")
            external_buffer(v_host_ty, name="V")
            external_buffer(out_host_ty, name="Out")

            kernels = declare_kernels(l2_tile_ty, tile_ty, v_tile_ty, g_flat_ty, gp_ty, row_ty)

            shim_tiles = {col: tile(col, 0) for col in range(8)}
            mem_tiles = {col: tile(col, 1) for col in range(8)}
            compute_tiles = {(col, row): tile(col, row) for col in range(8) for row in range(2, 6)}

            emit_reference_flows(shim_tiles, mem_tiles, compute_tiles)
            allocations = emit_reference_shim_allocations(shim_tiles)

            mem_specs: dict[int, MemTileSpec] = {}
            for col in range(8):
                mem_specs[col] = MemTileSpec(
                    segment=col // 4,
                    index=col % 4,
                    tile=mem_tiles[col],
                    qk=buffer(mem_tiles[col], datatype=l2_tile_ty, name=f"qk_l2_col{col}"),
                    v=buffer(mem_tiles[col], datatype=l2_tile_ty, name=f"v_l2_col{col}"),
                    out=buffer(mem_tiles[col], datatype=gp_ty, name=f"out_l2_col{col}"),
                    out_wait=lock(mem_tiles[col], lock_id=0, init=0),
                    out_ready=lock(mem_tiles[col], lock_id=1, init=1),
                    qk_wait=lock(mem_tiles[col], lock_id=2, init=0),
                    qk_ready=lock(mem_tiles[col], lock_id=3, init=1),
                    v_wait=lock(mem_tiles[col], lock_id=4, init=0),
                    v_ready=lock(mem_tiles[col], lock_id=5, init=1),
                )

            compute_specs: list[ComputeTileSpec] = []
            for segment in range(NUM_SEGMENTS):
                base = segment * 4
                for q_col in range(NUM_Q_TILES):
                    for stage in range(NUM_CASCADE_STAGES):
                        row = 2 + stage
                        tile_ref = compute_tiles[(base + q_col, row)]
                        if stage == 0:
                            compute_specs.append(
                                ComputeTileSpec(
                                    segment=segment,
                                    stage=stage,
                                    q_col=q_col,
                                    tile=tile_ref,
                                    qk=buffer(tile_ref, datatype=tile_ty, name=f"qk_seg{segment}_s{stage}_q{q_col}"),
                                    q=buffer(tile_ref, datatype=tile_ty, name=f"q_seg{segment}_s{stage}_q{q_col}"),
                                    v=buffer(tile_ref, datatype=v_tile_ty, name=f"v_seg{segment}_s{stage}_q{q_col}"),
                                    g=buffer(tile_ref, datatype=g_tile_ty, name=f"g_seg{segment}_s{stage}_q{q_col}"),
                                    gp=buffer(tile_ref, datatype=gp_ty, name=f"gp_seg{segment}_s{stage}_q{q_col}"),
                                    up=buffer(tile_ref, datatype=row_ty, name=f"up_seg{segment}_s{stage}_q{q_col}"),
                                    sp=buffer(tile_ref, datatype=row_ty, name=f"sp_seg{segment}_s{stage}_q{q_col}"),
                                    s=buffer(tile_ref, datatype=row_ty, name=f"s_seg{segment}_s{stage}_q{q_col}"),
                                    r=buffer(tile_ref, datatype=row_ty, name=f"r_seg{segment}_s{stage}_q{q_col}"),
                                    merged_gp=buffer(tile_ref, datatype=gp_ty, name=f"merged_gp_seg{segment}_q{q_col}"),
                                    merged_up=buffer(tile_ref, datatype=row_ty, name=f"merged_up_seg{segment}_q{q_col}"),
                                    merged_sp=buffer(tile_ref, datatype=row_ty, name=f"merged_sp_seg{segment}_q{q_col}"),
                                    prev_up=buffer(tile_ref, datatype=row_ty, name=f"prev_up_seg{segment}_q{q_col}"),
                                    r_from_cascade=buffer(tile_ref, datatype=row_ty, name=f"r_cascade_seg{segment}_q{q_col}"),
                                    r_from_local=buffer(tile_ref, datatype=row_ty, name=f"r_local_seg{segment}_q{q_col}"),
                                    tmp_sp=buffer(tile_ref, datatype=row_ty, name=f"tmp_sp_seg{segment}_q{q_col}"),
                                    out_dma_acquire=lock(tile_ref, lock_id=0, init=0),
                                    out_ready=lock(tile_ref, lock_id=1, init=1),
                                    qk_dma_acquire=lock(tile_ref, lock_id=3, init=1),
                                    qk_ready=lock(tile_ref, lock_id=2, init=0),
                                    v_dma_acquire=lock(tile_ref, lock_id=5, init=1),
                                    v_ready=lock(tile_ref, lock_id=4, init=0),
                                )
                            )
                            continue

                        if stage == NUM_CASCADE_STAGES - 1:
                            compute_specs.append(
                                ComputeTileSpec(
                                    segment=segment,
                                    stage=stage,
                                    q_col=q_col,
                                    tile=tile_ref,
                                    qk=buffer(tile_ref, datatype=tile_ty, name=f"qk_seg{segment}_s{stage}_q{q_col}"),
                                    q=buffer(tile_ref, datatype=tile_ty, name=f"q_seg{segment}_s{stage}_q{q_col}"),
                                    v=buffer(tile_ref, datatype=v_tile_ty, name=f"v_seg{segment}_s{stage}_q{q_col}"),
                                    g=buffer(tile_ref, datatype=g_tile_ty, name=f"g_seg{segment}_s{stage}_q{q_col}"),
                                    gp=buffer(tile_ref, datatype=gp_ty, name=f"gp_seg{segment}_s{stage}_q{q_col}"),
                                    up=buffer(tile_ref, datatype=row_ty, name=f"up_seg{segment}_s{stage}_q{q_col}"),
                                    sp=buffer(tile_ref, datatype=row_ty, name=f"sp_seg{segment}_s{stage}_q{q_col}"),
                                    s=buffer(tile_ref, datatype=row_ty, name=f"s_seg{segment}_s{stage}_q{q_col}"),
                                    r=buffer(tile_ref, datatype=row_ty, name=f"r_seg{segment}_s{stage}_q{q_col}"),
                                    merged_gp=None,
                                    merged_up=None,
                                    merged_sp=None,
                                    prev_up=None,
                                    r_from_cascade=None,
                                    r_from_local=None,
                                    tmp_sp=None,
                                    out_dma_acquire=None,
                                    out_ready=None,
                                    qk_dma_acquire=lock(tile_ref, lock_id=1, init=1),
                                    qk_ready=lock(tile_ref, lock_id=0, init=0),
                                    v_dma_acquire=lock(tile_ref, lock_id=3, init=1),
                                    v_ready=lock(tile_ref, lock_id=2, init=0),
                                )
                            )
                            continue

                        compute_specs.append(
                            ComputeTileSpec(
                                segment=segment,
                                stage=stage,
                                q_col=q_col,
                                tile=tile_ref,
                                qk=buffer(tile_ref, datatype=tile_ty, name=f"qk_seg{segment}_s{stage}_q{q_col}"),
                                q=buffer(tile_ref, datatype=tile_ty, name=f"q_seg{segment}_s{stage}_q{q_col}"),
                                v=buffer(tile_ref, datatype=v_tile_ty, name=f"v_seg{segment}_s{stage}_q{q_col}"),
                                g=buffer(tile_ref, datatype=g_tile_ty, name=f"g_seg{segment}_s{stage}_q{q_col}"),
                                gp=buffer(tile_ref, datatype=gp_ty, name=f"gp_seg{segment}_s{stage}_q{q_col}"),
                                up=buffer(tile_ref, datatype=row_ty, name=f"up_seg{segment}_s{stage}_q{q_col}"),
                                sp=buffer(tile_ref, datatype=row_ty, name=f"sp_seg{segment}_s{stage}_q{q_col}"),
                                s=buffer(tile_ref, datatype=row_ty, name=f"s_seg{segment}_s{stage}_q{q_col}"),
                                r=buffer(tile_ref, datatype=row_ty, name=f"r_seg{segment}_s{stage}_q{q_col}"),
                                merged_gp=buffer(tile_ref, datatype=gp_ty, name=f"merged_gp_seg{segment}_s{stage}_q{q_col}"),
                                merged_up=buffer(tile_ref, datatype=row_ty, name=f"merged_up_seg{segment}_s{stage}_q{q_col}"),
                                merged_sp=buffer(tile_ref, datatype=row_ty, name=f"merged_sp_seg{segment}_s{stage}_q{q_col}"),
                                prev_up=buffer(tile_ref, datatype=row_ty, name=f"prev_up_seg{segment}_s{stage}_q{q_col}"),
                                r_from_cascade=buffer(tile_ref, datatype=row_ty, name=f"r_cascade_seg{segment}_s{stage}_q{q_col}"),
                                r_from_local=buffer(tile_ref, datatype=row_ty, name=f"r_local_seg{segment}_s{stage}_q{q_col}"),
                                tmp_sp=buffer(tile_ref, datatype=row_ty, name=f"tmp_sp_seg{segment}_s{stage}_q{q_col}"),
                                out_dma_acquire=None,
                                out_ready=None,
                                qk_dma_acquire=lock(tile_ref, lock_id=1, init=1),
                                qk_ready=lock(tile_ref, lock_id=0, init=0),
                                v_dma_acquire=lock(tile_ref, lock_id=3, init=1),
                                v_ready=lock(tile_ref, lock_id=2, init=0),
                            )
                        )

            for spec in mem_specs.values():
                emit_memtile_dma(spec)
            for spec in compute_specs:
                emit_compute_mem(spec)
                emit_compute_core(spec, kernels)

            emit_runtime_sequence(allocations, q_host_ty, k_host_ty, v_host_ty, out_host_ty)

        if not ctx.module.operation.verify():
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


def prepare_kernel_object(work_dir: Path) -> Path:
    staged_kernel_object = work_dir / KERNEL_OBJECT

    peano_install_dir = Path(
        os.environ.get("PEANO_INSTALL_DIR", str(WORKSPACE_ROOT / "install" / "llvm-aie"))
    )
    compiler = peano_install_dir / "bin" / "clang++"
    aie_opt = which("aie-opt")

    if compiler.exists() and aie_opt is not None:
        aieopt_dir = Path(aie_opt).resolve().parent.parent
        command = [
            str(compiler),
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-DNDEBUG",
            "-I",
            str(aieopt_dir / "include"),
            "-DBIT_WIDTH=8",
            "-c",
            str(KERNEL_SOURCE),
            "-o",
            str(staged_kernel_object),
            f"-Dlqp={Q_TILE_ROWS}",
            f"-Dlkp={LKP}",
            f"-Ddk={LKP}",
            f"-Ddk_full={DK}",
            f"-Ddv={LKP}",
            f"-Ddv_full={DV}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DROUND_CONV_EVEN",
        ]
        subprocess.run(command, check=True)
        return staged_kernel_object

    if REFERENCE_KERNEL_OBJECT.exists():
        copy2(REFERENCE_KERNEL_OBJECT, staged_kernel_object)
        return staged_kernel_object

    raise FileNotFoundError(
        "Unable to prepare the kernel-fusion flash-attention kernel object. "
        f"Neither a usable compiler at {compiler} nor a fallback object at "
        f"{REFERENCE_KERNEL_OBJECT} was found."
    )


def load_kernel(work_dir: Path) -> tuple[NPUKernel, object]:
    insts_path = work_dir / "insts.bin"
    xclbin_path = work_dir / "final.xclbin"
    npu_kernel = NPUKernel(str(xclbin_path), str(insts_path), kernel_name="MLIR_AIE")
    DefaultNPURuntime.cleanup()
    return npu_kernel, DefaultNPURuntime.load(npu_kernel)


def run_once(work_dir: Path, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, float]:
    _, kernel_handle = load_kernel(work_dir)
    in_q = iron.tensor(Q.reshape(-1), dtype=Q.dtype)
    in_k = iron.tensor(K.reshape(-1), dtype=K.dtype)
    in_v = iron.tensor(V.reshape(-1), dtype=V.dtype)
    out = iron.zeros(NUM_HEADS * LQ * DV, dtype=Q.dtype)
    try:
        result = DefaultNPURuntime.run(kernel_handle, [in_q, in_k, in_v, out])
        observed = np.array(out.numpy()).reshape(NUM_HEADS, LQ, DV)
        return observed, float(result.npu_time)
    finally:
        DefaultNPURuntime.cleanup()


def validate_output(work_dir: Path, Q: np.ndarray, K: np.ndarray, V: np.ndarray, expected: np.ndarray) -> bool:
    print("[3/3] Running on NPU and validating results")
    observed, npu_time = run_once(work_dir, Q, K, V)
    max_abs = float(np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32))))
    max_rel = max_abs / (float(np.max(np.abs(expected.astype(np.float32)))) + 1e-10)
    is_close = bool(
        np.allclose(
            observed.astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-1,
            atol=1e-2,
        )
    )
    print(f"Kernel runtime: {npu_time:.0f} ns")
    print(f"Max absolute error: {max_abs:.6f}")
    print(f"Max relative error: {max_rel:.6f}")
    print(f"Output[0,0,:8]: {observed[0, 0, :8]}")
    print(f"Expect[0,0,:8]: {expected[0, 0, :8]}")
    return is_close


def run_benchmark(work_dir: Path, Q: np.ndarray, K: np.ndarray, V: np.ndarray, warmup: int, iterations: int) -> int:
    import time

    macs = float(NUM_HEADS * (LQ * LK * DK * 2 + LK * LQ * DV * 2))
    total_iters = warmup + iterations
    times_us: list[float] = []

    print(
        f"[3/3] Benchmarking kernel-fusion flash attention: heads={NUM_HEADS}, "
        f"lq={LQ}, lk={LK}, dk={DK}, dv={DV}"
    )
    print(f"      {warmup} warmup + {iterations} measurement iterations")

    _, kernel_handle = load_kernel(work_dir)
    try:
        for i in range(total_iters):
            in_q = iron.tensor(Q.reshape(-1), dtype=Q.dtype)
            in_k = iron.tensor(K.reshape(-1), dtype=K.dtype)
            in_v = iron.tensor(V.reshape(-1), dtype=V.dtype)
            out = iron.zeros(NUM_HEADS * LQ * DV, dtype=Q.dtype)
            try:
                t0 = time.perf_counter()
                DefaultNPURuntime.run(kernel_handle, [in_q, in_k, in_v, out])
                t1 = time.perf_counter()
            finally:
                del in_q, in_k, in_v, out

            if i >= warmup:
                times_us.append((t1 - t0) * 1e6)
    finally:
        DefaultNPURuntime.cleanup()

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


def main() -> int:
    args = parse_args()
    ensure_supported_configuration(args)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    Q, K, V = generate_test_data()
    expected = None if args.benchmark else flash_attention_golden(Q, K, V)

    if args.verbose:
        print("Flash Attention kernel-fusion direct AIE emitter")
        print(f"  reference aie.mlir: {REFERENCE_AIE_MLIR}")
        print(f"  reference npu.mlir: {REFERENCE_NPU_MLIR}")
        print(f"  kernel source:      {KERNEL_SOURCE}")
        print(f"  kernel object:      {KERNEL_OBJECT}")
        print(f"  shape Q={Q.shape} K={K.shape} V={V.shape}")
        print(f"  q_groups={Q_GROUPS} chunks_per_stage={CHUNKS_PER_STAGE}")

    module = build_mlir_module()
    mlir_path = work_dir / "flash_attention_kernel_fusion_direct_aie.mlir"
    with open(mlir_path, "w", encoding="utf-8") as handle:
        print(module, file=handle)

    if args.print_module_only:
        print(module)
        return 0

    print(f"Wrote direct AIE module to {mlir_path}")

    kernel_object_path = prepare_kernel_object(work_dir)
    print(f"Prepared kernel object at {kernel_object_path}")

    insts_path = work_dir / "insts.bin"
    xclbin_path = work_dir / "final.xclbin"
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(insts_path),
        xclbin_path=str(xclbin_path),
        work_dir=str(work_dir),
        verbose=args.verbose,
    )
    print(f"Compiled design to {xclbin_path}")

    if args.compile_only:
        print("PASS! (compile-only)")
        return 0

    if args.benchmark:
        return run_benchmark(work_dir, Q, K, V, args.warmup, args.iterations)

    assert expected is not None
    if validate_output(work_dir, Q, K, V, expected):
        print("PASS!")
        return 0

    print("FAILED: output does not match the NumPy golden within tolerance")
    return 1


if __name__ == "__main__":
    sys.exit(main())