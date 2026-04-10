#!/usr/bin/env python3
# flash_attention.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

"""AIE2P Flash Attention using direct AIE dialect Python bindings.

This emitter now mirrors both lowered references closely:
  - `aie.air.mlir` for the device-side tile, lock, DMA, and cascade structure
  - `npu.air.mlir` for the `aie.runtime_sequence` shim DMA task schedule

The Python generation stays in the direct `aie` / `aiex` dialect layer rather
than going back through higher-level IRON placement helpers.
"""

from __future__ import annotations

import argparse
import gc
from shutil import copy2
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron

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
from aie.ir import AffineDimExpr, AffineMap, MemRefType
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "flash_attention_build"
REFERENCE_AIE_MLIR = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "dataflow_based"
    / "build_peano"
    / "air_project"
    / "aie.air.mlir"
)
REFERENCE_NPU_MLIR = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "dataflow_based"
    / "build_peano"
    / "air_project"
    / "npu.air.mlir"
)
KERNEL_SOURCE = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "dataflow_based"
    / "attn_aie2p.cc"
)
REFERENCE_KERNEL_OBJECT = (
    WORKSPACE_ROOT
    / "mlir-air"
    / "programming_examples"
    / "flash_attention"
    / "dataflow_based"
    / "build_peano"
    / "attn.o"
)
KERNEL_OBJECT = "attn.o"


# ---------------------------------------------------------------------------
# Problem-size constants (Makefile defaults for aie2p)
# ---------------------------------------------------------------------------
LK = 12288
LKP = 96
LQ = 64
DK = 64
DV = 64

NUM_CASCADE_STAGES = 4
NUM_CHUNKS = LK // LKP
CHUNKS_PER_STAGE = NUM_CHUNKS // NUM_CASCADE_STAGES

Q_SIZE = LQ * DK
K_CHUNK_SIZE = DK * LKP
V_CHUNK_SIZE = LKP * DV
G_SIZE = LQ * LKP
GP_SIZE = LQ * DV
ROW_SIZE = LQ
OUTPUT_SIZE = LQ * DV


# ---------------------------------------------------------------------------
# NumPy golden reference
# ---------------------------------------------------------------------------
def flash_attention_golden(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    M: np.ndarray,
    lk: int = LK,
    lkp: int = LKP,
) -> np.ndarray:
    """Online flash-attention in bf16 (matches AIE kernel numerics)."""
    dt = bfloat16
    lq, dv = Q.shape[0], V.shape[1]

    Gp = np.zeros((lq, dv), dtype=dt)
    up = np.full((lq, 1), -np.inf, dtype=dt)
    sp = np.zeros((lq, 1), dtype=dt)

    for j in range(lk // lkp):
        G = M[:, j * lkp : (j + 1) * lkp]
        B = K[:, j * lkp : (j + 1) * lkp]
        G = Q @ B + G
        G = G.astype(dt)

        u = np.max(G, axis=-1, keepdims=True).astype(dt)
        u = np.maximum(u, up)
        G = np.exp(G - u).astype(dt)

        B = V[j * lkp : (j + 1) * lkp, :]
        r = np.exp(up - u).astype(dt)
        Gp = Gp * r
        Gp = (G @ B + Gp).astype(dt)

        s = np.sum(G, axis=-1, keepdims=True).astype(dt)
        s = (s + sp * r).astype(dt)
        sp, up = s, u

    return (Gp / sp).astype(dt)


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------
def generate_test_data(
    lq: int = LQ, dk: int = DK, lk: int = LK, dv: int = DV
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic test inputs matching attn.py's pattern."""
    dt = bfloat16
    Q = (np.arange(0, lq * dk, dtype=dt).reshape(lq, dk) / (lq * dk) * 2).astype(dt)
    K = (np.arange(0, dk * lk, dtype=dt).reshape(dk, lk) / (dk * lk) * 2).astype(dt)
    V = (np.arange(0, lk * dv, dtype=dt).reshape(lk, dv) / (lk * dv) * 2).astype(dt)
    M = np.zeros((lq, lk), dtype=dt)

    Q_scaled = (Q / sqrt(dk)).astype(dt)
    return Q_scaled, K, V, M


@dataclass(frozen=True)
class KernelSet:
    matmul_a_b: object
    matmul_g_b: object
    zero_fill_gp: object
    zero_fill_sp: object
    zero_fill_g: object
    neg_inf_fill_up: object
    max_g: object
    maximum_up_u: object
    exp_g_minus_u: object
    exp_up_minus_u: object
    mul_r_gp: object
    sum_g: object
    accum_sp_r_s: object
    vector_copy_32: object
    vector_copy_32x96: object
    vector_accum_32x64: object
    div_gp_sp: object
    add_gp_g: object


@dataclass(frozen=True)
class QKStage:
    stage: int
    tile: object
    q: object
    k: object
    g: object
    q_dma_acquire: object
    q_ready: object
    k_dma_acquire: object
    k_ready: object
    g_dma_acquire: object
    g_ready: object


@dataclass(frozen=True)
class GVStage:
    stage: int
    tile: object
    gp: object
    g: object
    v: object
    gp_dma_acquire: object
    gp_ready: object
    v_dma_acquire: object
    v_ready: object
    g_dma_acquire: object
    g_ready: object


@dataclass(frozen=True)
class SoftmaxStage:
    stage: int
    tile: object
    gp: object
    sp: object
    up: object
    u: object
    r: object
    s: object
    g: object
    g_copy: object
    gv: object
    merged_gp: object | None
    merged_up: object | None
    merged_sp: object | None
    prev_up: object | None
    r_from_cascade: object | None
    r_from_local: object | None
    tmp_sp: object | None
    out_dma_acquire: object | None
    out_ready: object | None
    g_copy_dma_acquire: object
    g_copy_ready: object
    g_dma_acquire: object
    g_ready: object
    gv_dma_acquire: object
    gv_ready: object


@dataclass(frozen=True)
class QKMemtileStage:
    tile: object
    q: object
    k: object
    q_wait: object
    q_ready: object
    k_wait: object
    k_ready: object


@dataclass(frozen=True)
class VMemtileStage:
    tile: object
    v: object
    v_wait: object
    v_ready: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flash_attention.py",
        description="AIE2P Flash Attention direct AIE dialect emitter",
    )
    parser.add_argument("--lk", type=int, default=LK, help="K/V sequence length")
    parser.add_argument("--lkp", type=int, default=LKP, help="K/V chunk size")
    parser.add_argument("--lq", type=int, default=LQ, help="Q sequence length")
    parser.add_argument("--dk", type=int, default=DK, help="Key dimension")
    parser.add_argument("--dv", type=int, default=DV, help="Value dimension")
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
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup kernel launches before timing (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of timed benchmark launches to average (default: 20)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def ensure_supported_configuration(args: argparse.Namespace) -> None:
    if (args.lk, args.lkp, args.lq, args.dk, args.dv) != (LK, LKP, LQ, DK, DV):
        raise ValueError(
            "The direct AIE emitter currently mirrors the fixed lowered flash-attention "
            f"configuration only: lk={LK}, lkp={LKP}, lq={LQ}, dk={DK}, dv={DV}."
        )
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")


def declare_kernels(
    q_ty: type[np.ndarray],
    k_ty: type[np.ndarray],
    v_ty: type[np.ndarray],
    g_ty: type[np.ndarray],
    gp_ty: type[np.ndarray],
    row_ty: type[np.ndarray],
) -> KernelSet:
    return KernelSet(
        matmul_a_b=external_func(
            "matmul_a_b_bf16",
            inputs=[q_ty, k_ty, g_ty],
            link_with=KERNEL_OBJECT,
        ),
        matmul_g_b=external_func(
            "matmul_g_b_bf16",
            inputs=[g_ty, v_ty, gp_ty],
            link_with=KERNEL_OBJECT,
        ),
        zero_fill_gp=external_func(
            "zero_fill_gp_bf16",
            inputs=[gp_ty],
            link_with=KERNEL_OBJECT,
        ),
        zero_fill_sp=external_func(
            "zero_fill_sp_bf16",
            inputs=[row_ty],
            link_with=KERNEL_OBJECT,
        ),
        zero_fill_g=external_func(
            "zero_fill_g_bf16",
            inputs=[g_ty],
            link_with=KERNEL_OBJECT,
        ),
        neg_inf_fill_up=external_func(
            "neg_inf_fill_up_bf16",
            inputs=[row_ty],
            link_with=KERNEL_OBJECT,
        ),
        max_g=external_func(
            "max_g_bf16",
            inputs=[g_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        maximum_up_u=external_func(
            "maximum_up_u_bf16",
            inputs=[row_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        exp_g_minus_u=external_func(
            "exp_g_minus_u",
            inputs=[row_ty, g_ty],
            link_with=KERNEL_OBJECT,
        ),
        exp_up_minus_u=external_func(
            "exp_up_minus_u",
            inputs=[row_ty, row_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        mul_r_gp=external_func(
            "mul_r_gp",
            inputs=[row_ty, gp_ty],
            link_with=KERNEL_OBJECT,
        ),
        sum_g=external_func(
            "sum_g",
            inputs=[g_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        accum_sp_r_s=external_func(
            "accum_sp_r_s",
            inputs=[row_ty, row_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        vector_copy_32=external_func(
            "vector_copy_32elems",
            inputs=[np.int32, row_ty, row_ty],
            link_with=KERNEL_OBJECT,
        ),
        vector_copy_32x96=external_func(
            "vector_copy_32x96elems",
            inputs=[np.int32, g_ty, g_ty],
            link_with=KERNEL_OBJECT,
        ),
        vector_accum_32x64=external_func(
            "vector_accum_32x64elems",
            inputs=[gp_ty, gp_ty],
            link_with=KERNEL_OBJECT,
        ),
        div_gp_sp=external_func(
            "div_gp_sp",
            inputs=[row_ty, gp_ty],
            link_with=KERNEL_OBJECT,
        ),
        add_gp_g=external_func(
            "add_gp_g",
            inputs=[gp_ty, gp_ty],
            link_with=KERNEL_OBJECT,
        ),
    )


def emit_reference_flows(
    shim_tiles: dict[int, object],
    mem_tiles: dict[int, object],
    qk_tiles: list[object],
    gv_tiles: list[object],
    sf_tiles: list[object],
) -> None:
    for col in (0, 1, 2, 3, 5, 6, 7):
        flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)

    flow(shim_tiles[0], WireBundle.DMA, 1, mem_tiles[0], WireBundle.DMA, 1)

    for stage in range(NUM_CASCADE_STAGES):
        flow(mem_tiles[stage], WireBundle.DMA, 0, qk_tiles[stage], WireBundle.DMA, 0)
        flow(mem_tiles[stage], WireBundle.DMA, 1, qk_tiles[stage], WireBundle.DMA, 1)

    v_mem_cols = [5, 6, 7, 0]
    v_mem_channels = [0, 0, 0, 2]
    for stage in range(NUM_CASCADE_STAGES):
        flow(
            mem_tiles[v_mem_cols[stage]],
            WireBundle.DMA,
            v_mem_channels[stage],
            gv_tiles[stage],
            WireBundle.DMA,
            0,
        )

    flow(sf_tiles[0], WireBundle.DMA, 0, mem_tiles[4], WireBundle.DMA, 0)
    flow(mem_tiles[4], WireBundle.DMA, 0, shim_tiles[4], WireBundle.DMA, 0)

    for stage in range(NUM_CASCADE_STAGES - 1, -1, -1):
        flow(qk_tiles[stage], WireBundle.DMA, 0, sf_tiles[stage], WireBundle.DMA, 0)
        gv_input_channel = 1 if stage == 0 else 0
        flow(sf_tiles[stage], WireBundle.DMA, gv_input_channel, gv_tiles[stage], WireBundle.DMA, 1)
        flow(gv_tiles[stage], WireBundle.DMA, 0, sf_tiles[stage], WireBundle.DMA, 1)

    cascade_flow(sf_tiles[3], sf_tiles[2])
    cascade_flow(sf_tiles[2], sf_tiles[1])
    cascade_flow(sf_tiles[1], sf_tiles[0])


def emit_reference_shim_allocations(shim_tiles: dict[int, object]) -> dict[str, object]:
    allocations = {
        "out": shim_dma_allocation("air_L2ToL3Chan1", shim_tiles[4], DMAChannelDir.S2MM, 0),
        "qk_0": shim_dma_allocation("air_L3ToL2Chan1_0", shim_tiles[0], DMAChannelDir.MM2S, 0),
        "qk_1": shim_dma_allocation("air_L3ToL2Chan1_1", shim_tiles[1], DMAChannelDir.MM2S, 0),
        "qk_2": shim_dma_allocation("air_L3ToL2Chan1_2", shim_tiles[2], DMAChannelDir.MM2S, 0),
        "qk_3": shim_dma_allocation("air_L3ToL2Chan1_3", shim_tiles[3], DMAChannelDir.MM2S, 0),
        "v_0": shim_dma_allocation("air_L3ToL2Chan3_0", shim_tiles[5], DMAChannelDir.MM2S, 0),
        "v_1": shim_dma_allocation("air_L3ToL2Chan3_1", shim_tiles[6], DMAChannelDir.MM2S, 0),
        "v_2": shim_dma_allocation("air_L3ToL2Chan3_2", shim_tiles[7], DMAChannelDir.MM2S, 0),
        "v_3": shim_dma_allocation("air_L3ToL2Chan3_3", shim_tiles[0], DMAChannelDir.MM2S, 1),
    }
    return allocations


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
    flat = memref.collapse_shape(collapsed_memref_type(buffer_ref, total_elems), buffer_ref, [[0, 1]])
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
    flat = memref.collapse_shape(collapsed_memref_type(buffer_ref, total_elems), buffer_ref, [[0, 1]])
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


def emit_qk_mem(stage: QKStage) -> None:
    @mem(stage.tile)
    def qk_dma(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(stage.g_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.g, offset=0, len=G_SIZE, dimensions=[(64, 8), (12, 512), (8, 1)])
            use_lock(stage.g_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(stage.q_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.q, offset=0, len=Q_SIZE)
            use_lock(stage.q_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(stage.k_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.k, offset=0, len=K_CHUNK_SIZE)
            use_lock(stage.k_ready, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            EndOp()


def emit_qk_core(stage: QKStage, kernels: KernelSet) -> None:
    @core(stage.tile)
    def qk_core_body():
        for _ in range_(sys.maxsize):
            use_lock(stage.q_ready, LockAction.AcquireGreaterEqual, value=1)
            for _ in range_(CHUNKS_PER_STAGE):
                use_lock(stage.g_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.zero_fill_g(stage.g)
                use_lock(stage.k_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.matmul_a_b(stage.q, stage.k, stage.g)
                use_lock(stage.k_dma_acquire, LockAction.Release, value=1)
                use_lock(stage.g_dma_acquire, LockAction.Release, value=1)
            use_lock(stage.q_dma_acquire, LockAction.Release, value=1)


def emit_gv_mem(stage: GVStage) -> None:
    @mem(stage.tile)
    def gv_dma(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(stage.gp_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.gp, offset=0, len=GP_SIZE, dimensions=[(64, 8), (8, 512), (8, 1)])
            use_lock(stage.gp_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(stage.v_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.v, offset=0, len=V_CHUNK_SIZE)
            use_lock(stage.v_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(stage.g_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.g, offset=0, len=G_SIZE)
            use_lock(stage.g_ready, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            EndOp()


def emit_gv_core(stage: GVStage, kernels: KernelSet) -> None:
    @core(stage.tile)
    def gv_core_body():
        for _ in range_(sys.maxsize):
            for _ in range_(CHUNKS_PER_STAGE):
                use_lock(stage.gp_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.zero_fill_gp(stage.gp)
                use_lock(stage.g_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(stage.v_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.matmul_g_b(stage.g, stage.v, stage.gp)
                use_lock(stage.v_dma_acquire, LockAction.Release, value=1)
                use_lock(stage.g_dma_acquire, LockAction.Release, value=1)
                use_lock(stage.gp_dma_acquire, LockAction.Release, value=1)


def emit_softmax_mem(stage: SoftmaxStage) -> None:
    @mem(stage.tile)
    def softmax_dma(block):
        if stage.stage == 0:
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(stage.out_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(stage.merged_gp, offset=0, len=OUTPUT_SIZE)
                use_lock(stage.out_ready, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.MM2S, 1, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(stage.g_copy_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(stage.g_copy, offset=0, len=G_SIZE, dimensions=[(12, 8), (64, 96), (8, 1)])
                use_lock(stage.g_copy_ready, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[6])
            with block[5]:
                use_lock(stage.g_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(stage.g, offset=0, len=G_SIZE)
                use_lock(stage.g_ready, LockAction.Release, value=1)
                next_bd(block[5])
            with block[6]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[8])
            with block[7]:
                use_lock(stage.gv_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(stage.gv, offset=0, len=GP_SIZE)
                use_lock(stage.gv_ready, LockAction.Release, value=1)
                next_bd(block[7])
            with block[8]:
                EndOp()
            return

        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(stage.g_copy_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.g_copy, offset=0, len=G_SIZE, dimensions=[(12, 8), (64, 96), (8, 1)])
            use_lock(stage.g_copy_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(stage.g_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.g, offset=0, len=G_SIZE)
            use_lock(stage.g_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(stage.gv_dma_acquire, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(stage.gv, offset=0, len=GP_SIZE)
            use_lock(stage.gv_ready, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            EndOp()


def emit_softmax_core(stage: SoftmaxStage, kernels: KernelSet) -> None:
    @core(stage.tile)
    def softmax_core_body():
        c0 = arith.constant(0, index=True)
        zero_bf16 = arith.constant(0.0, T.bf16())

        for _ in range_(sys.maxsize):
            if stage.stage == 0:
                use_lock(stage.out_ready, LockAction.AcquireGreaterEqual, value=1)

            kernels.zero_fill_gp(stage.gp)
            kernels.zero_fill_sp(stage.sp)
            kernels.neg_inf_fill_up(stage.up)

            for _ in range_(CHUNKS_PER_STAGE):
                use_lock(stage.g_copy_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(stage.g_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.max_g(stage.g, stage.u)
                kernels.maximum_up_u(stage.up, stage.u)
                kernels.exp_g_minus_u(stage.u, stage.g)
                kernels.exp_up_minus_u(stage.up, stage.u, stage.r)
                kernels.mul_r_gp(stage.r, stage.gp)
                kernels.vector_copy_32x96(0, stage.g, stage.g_copy)
                use_lock(stage.g_copy_dma_acquire, LockAction.Release, value=1)
                use_lock(stage.gv_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.vector_accum_32x64(stage.gv, stage.gp)
                use_lock(stage.gv_dma_acquire, LockAction.Release, value=1)
                kernels.sum_g(stage.g, stage.s)
                kernels.accum_sp_r_s(stage.sp, stage.r, stage.s)
                kernels.vector_copy_32(0, stage.s, stage.sp)
                kernels.vector_copy_32(0, stage.u, stage.up)
                use_lock(stage.g_dma_acquire, LockAction.Release, value=1)

            if stage.stage == NUM_CASCADE_STAGES - 1:
                emit_cascade_send(stage.gp, OUTPUT_SIZE, zero_bf16, c0)
                emit_cascade_send(stage.up, ROW_SIZE, zero_bf16, c0)
                emit_cascade_send(stage.sp, ROW_SIZE, zero_bf16, c0)
                continue

            emit_cascade_receive(stage.merged_gp, OUTPUT_SIZE, c0)
            emit_cascade_receive(stage.merged_up, ROW_SIZE, c0)
            emit_cascade_receive(stage.merged_sp, ROW_SIZE, c0)

            kernels.vector_copy_32(0, stage.up, stage.prev_up)
            kernels.maximum_up_u(stage.merged_up, stage.up)
            kernels.exp_up_minus_u(stage.merged_up, stage.up, stage.r_from_cascade)
            kernels.exp_up_minus_u(stage.prev_up, stage.up, stage.r_from_local)
            kernels.mul_r_gp(stage.r_from_cascade, stage.merged_gp)
            kernels.mul_r_gp(stage.r_from_local, stage.gp)
            kernels.add_gp_g(stage.gp, stage.merged_gp)
            kernels.zero_fill_sp(stage.tmp_sp)
            kernels.accum_sp_r_s(stage.merged_sp, stage.r_from_cascade, stage.tmp_sp)
            kernels.accum_sp_r_s(stage.sp, stage.r_from_local, stage.tmp_sp)
            kernels.vector_copy_32(0, stage.tmp_sp, stage.merged_sp)

            if stage.stage == 0:
                kernels.div_gp_sp(stage.merged_sp, stage.merged_gp)
                use_lock(stage.out_dma_acquire, LockAction.Release, value=1)
                continue

            emit_cascade_send(stage.merged_gp, OUTPUT_SIZE, zero_bf16, c0)
            emit_cascade_send(stage.up, ROW_SIZE, zero_bf16, c0)
            emit_cascade_send(stage.merged_sp, ROW_SIZE, zero_bf16, c0)


def emit_qk_memtile_dma(spec: QKMemtileStage) -> None:
    @memtile_dma(spec.tile)
    def qk_memtile(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(spec.q_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.q, offset=0, len=Q_SIZE, dimensions=[(8, 8), (64, 64), (8, 1)])
            use_lock(spec.q_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(spec.k_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.k, offset=0, len=K_CHUNK_SIZE, dimensions=[(12, 8), (64, 96), (8, 1)])
            use_lock(spec.k_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(spec.q_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.q, offset=0, len=Q_SIZE)
            use_lock(spec.q_wait, LockAction.Release, value=1)
            next_bd(block[8])
        with block[6]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[8], repeat_count=31)
        with block[7]:
            use_lock(spec.k_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.k, offset=0, len=K_CHUNK_SIZE)
            use_lock(spec.k_wait, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            EndOp()


def emit_qk_v_memtile0_dma(
    qk_spec: QKMemtileStage,
    v_spec: VMemtileStage,
) -> None:
    @memtile_dma(qk_spec.tile)
    def memtile0_dma(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(qk_spec.q_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(qk_spec.q, offset=0, len=Q_SIZE, dimensions=[(8, 8), (64, 64), (8, 1)])
            use_lock(qk_spec.q_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(qk_spec.k_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(qk_spec.k, offset=0, len=K_CHUNK_SIZE, dimensions=[(12, 8), (64, 96), (8, 1)])
            use_lock(qk_spec.k_ready, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            dma_start(DMAChannelDir.MM2S, 2, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(v_spec.v_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(v_spec.v, offset=0, len=V_CHUNK_SIZE, dimensions=[(8, 8), (96, 64), (8, 1)])
            use_lock(v_spec.v_ready, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[8])
        with block[7]:
            use_lock(qk_spec.q_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(qk_spec.q, offset=0, len=Q_SIZE)
            use_lock(qk_spec.q_wait, LockAction.Release, value=1)
            next_bd(block[12])
        with block[8]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[9], chain=block[10], repeat_count=31)
        with block[9]:
            use_lock(qk_spec.k_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(qk_spec.k, offset=0, len=K_CHUNK_SIZE)
            use_lock(qk_spec.k_wait, LockAction.Release, value=1)
            next_bd(block[12])
        with block[10]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[11], chain=block[12])
        with block[11]:
            use_lock(v_spec.v_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(v_spec.v, offset=0, len=V_CHUNK_SIZE)
            use_lock(v_spec.v_wait, LockAction.Release, value=1)
            next_bd(block[11])
        with block[12]:
            EndOp()


def emit_v_memtile_dma(spec: VMemtileStage) -> None:
    @memtile_dma(spec.tile)
    def v_memtile(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(spec.v_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.v, offset=0, len=V_CHUNK_SIZE, dimensions=[(8, 8), (96, 64), (8, 1)])
            use_lock(spec.v_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(spec.v_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(spec.v, offset=0, len=V_CHUNK_SIZE)
            use_lock(spec.v_wait, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            EndOp()


def emit_output_memtile_dma(out_tile: object, out_buffer: object, out_wait: object, out_ready: object) -> None:
    @memtile_dma(out_tile)
    def out_memtile(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(out_wait, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(out_buffer, offset=0, len=OUTPUT_SIZE)
            use_lock(out_ready, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(out_ready, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(out_buffer, offset=0, len=OUTPUT_SIZE)
            use_lock(out_wait, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            EndOp()


def emit_runtime_sequence(
    allocations: dict[str, object],
    q_host_ty: type[np.ndarray],
    k_host_ty: type[np.ndarray],
    v_host_ty: type[np.ndarray],
    mask_host_ty: type[np.ndarray],
    out_host_ty: type[np.ndarray],
) -> None:
    qk_allocs = [
        allocations["qk_0"],
        allocations["qk_1"],
        allocations["qk_2"],
        allocations["qk_3"],
    ]
    v_allocs = [
        allocations["v_0"],
        allocations["v_1"],
        allocations["v_2"],
        allocations["v_3"],
    ]

    @runtime_sequence(q_host_ty, k_host_ty, v_host_ty, mask_host_ty, out_host_ty)
    def attention_bf16(q, k, v, _mask, out):
        q_tasks = []
        for alloc in qk_allocs:
            task = dma_configure_task_for(alloc)
            with bds(task) as bd:
                with bd[0]:
                    dma_bd(q, offset=0, len=Q_SIZE, dimensions=[(8, 512), (512, 1)])
                    EndOp()
            dma_start_task(task)
            q_tasks.append(task)

        k_tasks = []
        for stage, alloc in enumerate(qk_allocs):
            task = dma_configure_task_for(alloc)
            with bds(task) as bd:
                with bd[0]:
                    dma_bd(
                        k,
                        offset=stage * LKP,
                        len=32 * DK * LKP,
                        dimensions=[(32, 384), (64, LK), (LKP, 1)],
                    )
                    EndOp()
            dma_start_task(task)
            k_tasks.append(task)

        v_tasks = []
        for stage, alloc in enumerate(v_allocs):
            task = dma_configure_task_for(alloc)
            with bds(task) as bd:
                with bd[0]:
                    dma_bd(
                        v,
                        offset=stage * V_CHUNK_SIZE,
                        len=32 * LKP * DV,
                        dimensions=[(32, 24576), (LKP, DV), (DV, 1)],
                    )
                    EndOp()
            dma_start_task(task)
            v_tasks.append(task)

        out_task = dma_configure_task_for(allocations["out"], issue_token=True)
        with bds(out_task) as bd:
            with bd[0]:
                dma_bd(out, offset=0, len=OUTPUT_SIZE, dimensions=[(8, 512), (512, 1)])
                EndOp()
        dma_start_task(out_task)

        dma_free_task(q_tasks[3])
        dma_free_task(q_tasks[2])
        dma_free_task(q_tasks[1])
        dma_free_task(q_tasks[0])
        dma_free_task(v_tasks[0])
        dma_free_task(v_tasks[2])
        dma_free_task(k_tasks[3])
        dma_free_task(k_tasks[2])
        dma_free_task(k_tasks[1])
        dma_free_task(k_tasks[0])
        dma_await_task(out_task)
        dma_free_task(v_tasks[3])
        dma_free_task(v_tasks[1])


def build_mlir_module() -> object:
    q_host_ty = np.ndarray[(LQ, DK), np.dtype[bfloat16]]
    k_host_ty = np.ndarray[(DK, LK), np.dtype[bfloat16]]
    v_host_ty = np.ndarray[(LK, DV), np.dtype[bfloat16]]
    mask_host_ty = np.ndarray[(LQ, LK), np.dtype[bfloat16]]
    out_host_ty = np.ndarray[(LQ, DV), np.dtype[bfloat16]]

    q_l2_ty = np.ndarray[(LQ, DK), np.dtype[bfloat16]]
    k_l2_ty = np.ndarray[(DK, LKP), np.dtype[bfloat16]]
    v_l2_ty = np.ndarray[(LKP, DV), np.dtype[bfloat16]]
    out_l2_ty = np.ndarray[(LQ, DV), np.dtype[bfloat16]]

    q_l1_ty = np.ndarray[(LQ, DK), np.dtype[bfloat16]]
    k_l1_ty = np.ndarray[(DK, LKP), np.dtype[bfloat16]]
    v_l1_ty = np.ndarray[(DK, LKP), np.dtype[bfloat16]]
    g_ty = np.ndarray[(G_SIZE,), np.dtype[bfloat16]]
    gp_ty = np.ndarray[(LQ, DV), np.dtype[bfloat16]]
    row_ty = np.ndarray[(LQ, 1), np.dtype[bfloat16]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            external_buffer(q_host_ty, name="Q")
            external_buffer(k_host_ty, name="K")
            external_buffer(v_host_ty, name="V")
            external_buffer(out_host_ty, name="Out")

            kernels = declare_kernels(q_l1_ty, k_l1_ty, v_l1_ty, g_ty, gp_ty, row_ty)

            shim_tiles = {col: tile(col, 0) for col in range(8)}
            mem_tiles = {col: tile(col, 1) for col in range(8)}
            qk_tiles = [tile(0, row) for row in range(2, 6)]
            gv_tiles = [tile(1, row) for row in range(2, 6)]
            sf_tiles = [tile(2, row) for row in range(2, 6)]

            emit_reference_flows(shim_tiles, mem_tiles, qk_tiles, gv_tiles, sf_tiles)
            allocations = emit_reference_shim_allocations(shim_tiles)

            mem0_q_ready = lock(mem_tiles[0], lock_id=1, init=1)
            mem0_q_wait = lock(mem_tiles[0], lock_id=0, init=0)
            mem0_k_ready = lock(mem_tiles[0], lock_id=3, init=1)
            mem0_k_wait = lock(mem_tiles[0], lock_id=2, init=0)
            mem0_v_ready = lock(mem_tiles[0], lock_id=5, init=1)
            mem0_v_wait = lock(mem_tiles[0], lock_id=4, init=0)

            qk_memtiles: list[QKMemtileStage] = [
                QKMemtileStage(
                    tile=mem_tiles[0],
                    q=buffer(mem_tiles[0], datatype=q_l2_ty, name="q_l2_stage0"),
                    k=buffer(mem_tiles[0], datatype=k_l2_ty, name="k_l2_stage0"),
                    q_wait=mem0_q_wait,
                    q_ready=mem0_q_ready,
                    k_wait=mem0_k_wait,
                    k_ready=mem0_k_ready,
                ),
                QKMemtileStage(
                    tile=mem_tiles[1],
                    q=buffer(mem_tiles[1], datatype=q_l2_ty, name="q_l2_stage1"),
                    k=buffer(mem_tiles[1], datatype=k_l2_ty, name="k_l2_stage1"),
                    q_wait=lock(mem_tiles[1], lock_id=0, init=0),
                    q_ready=lock(mem_tiles[1], lock_id=1, init=1),
                    k_wait=lock(mem_tiles[1], lock_id=2, init=0),
                    k_ready=lock(mem_tiles[1], lock_id=3, init=1),
                ),
                QKMemtileStage(
                    tile=mem_tiles[2],
                    q=buffer(mem_tiles[2], datatype=q_l2_ty, name="q_l2_stage2"),
                    k=buffer(mem_tiles[2], datatype=k_l2_ty, name="k_l2_stage2"),
                    q_wait=lock(mem_tiles[2], lock_id=0, init=0),
                    q_ready=lock(mem_tiles[2], lock_id=1, init=1),
                    k_wait=lock(mem_tiles[2], lock_id=2, init=0),
                    k_ready=lock(mem_tiles[2], lock_id=3, init=1),
                ),
                QKMemtileStage(
                    tile=mem_tiles[3],
                    q=buffer(mem_tiles[3], datatype=q_l2_ty, name="q_l2_stage3"),
                    k=buffer(mem_tiles[3], datatype=k_l2_ty, name="k_l2_stage3"),
                    q_wait=lock(mem_tiles[3], lock_id=0, init=0),
                    q_ready=lock(mem_tiles[3], lock_id=1, init=1),
                    k_wait=lock(mem_tiles[3], lock_id=2, init=0),
                    k_ready=lock(mem_tiles[3], lock_id=3, init=1),
                ),
            ]

            v_memtiles = [
                VMemtileStage(
                    tile=mem_tiles[5],
                    v=buffer(mem_tiles[5], datatype=v_l2_ty, name="v_l2_stage0"),
                    v_wait=lock(mem_tiles[5], lock_id=0, init=0),
                    v_ready=lock(mem_tiles[5], lock_id=1, init=1),
                ),
                VMemtileStage(
                    tile=mem_tiles[6],
                    v=buffer(mem_tiles[6], datatype=v_l2_ty, name="v_l2_stage1"),
                    v_wait=lock(mem_tiles[6], lock_id=0, init=0),
                    v_ready=lock(mem_tiles[6], lock_id=1, init=1),
                ),
                VMemtileStage(
                    tile=mem_tiles[7],
                    v=buffer(mem_tiles[7], datatype=v_l2_ty, name="v_l2_stage2"),
                    v_wait=lock(mem_tiles[7], lock_id=0, init=0),
                    v_ready=lock(mem_tiles[7], lock_id=1, init=1),
                ),
                VMemtileStage(
                    tile=mem_tiles[0],
                    v=buffer(mem_tiles[0], datatype=v_l2_ty, name="v_l2_stage3"),
                    v_wait=mem0_v_wait,
                    v_ready=mem0_v_ready,
                ),
            ]

            out_tile = mem_tiles[4]
            out_wait = lock(out_tile, lock_id=0, init=0)
            out_ready = lock(out_tile, lock_id=1, init=1)
            out_l2 = buffer(out_tile, datatype=out_l2_ty, name="out_l2")

            qk_stages: list[QKStage] = []
            gv_stages: list[GVStage] = []
            softmax_stages: list[SoftmaxStage] = []

            for stage in range(NUM_CASCADE_STAGES):
                qk_tile = qk_tiles[stage]
                qk_stages.append(
                    QKStage(
                        stage=stage,
                        tile=qk_tile,
                        q=buffer(qk_tile, datatype=q_l1_ty, name=f"q_stage{stage}"),
                        k=buffer(qk_tile, datatype=k_l1_ty, name=f"k_stage{stage}"),
                        g=buffer(qk_tile, datatype=g_ty, name=f"g_stage{stage}"),
                        q_dma_acquire=lock(qk_tile, lock_id=3, init=1),
                        q_ready=lock(qk_tile, lock_id=2, init=0),
                        k_dma_acquire=lock(qk_tile, lock_id=5, init=1),
                        k_ready=lock(qk_tile, lock_id=4, init=0),
                        g_dma_acquire=lock(qk_tile, lock_id=0, init=0),
                        g_ready=lock(qk_tile, lock_id=1, init=1),
                    )
                )

                gv_tile = gv_tiles[stage]
                gv_stages.append(
                    GVStage(
                        stage=stage,
                        tile=gv_tile,
                        gp=buffer(gv_tile, datatype=gp_ty, name=f"gp_in_stage{stage}"),
                        g=buffer(gv_tile, datatype=g_ty, name=f"g_in_stage{stage}"),
                        v=buffer(gv_tile, datatype=v_l1_ty, name=f"v_stage{stage}"),
                        gp_dma_acquire=lock(gv_tile, lock_id=0, init=0),
                        gp_ready=lock(gv_tile, lock_id=1, init=1),
                        v_dma_acquire=lock(gv_tile, lock_id=3, init=1),
                        v_ready=lock(gv_tile, lock_id=2, init=0),
                        g_dma_acquire=lock(gv_tile, lock_id=5, init=1),
                        g_ready=lock(gv_tile, lock_id=4, init=0),
                    )
                )

                sf_tile = sf_tiles[stage]
                if stage == 0:
                    softmax_stages.append(
                        SoftmaxStage(
                            stage=stage,
                            tile=sf_tile,
                            gp=buffer(sf_tile, datatype=gp_ty, name="softmax_gp_stage0"),
                            sp=buffer(sf_tile, datatype=row_ty, name="softmax_sp_stage0"),
                            up=buffer(sf_tile, datatype=row_ty, name="softmax_up_stage0"),
                            u=buffer(sf_tile, datatype=row_ty, name="softmax_u_stage0"),
                            r=buffer(sf_tile, datatype=row_ty, name="softmax_r_stage0"),
                            s=buffer(sf_tile, datatype=row_ty, name="softmax_s_stage0"),
                            g=buffer(sf_tile, datatype=g_ty, name="softmax_g_stage0"),
                            g_copy=buffer(sf_tile, datatype=g_ty, name="softmax_g_copy_stage0"),
                            gv=buffer(sf_tile, datatype=gp_ty, name="softmax_gv_stage0"),
                            merged_gp=buffer(sf_tile, datatype=gp_ty, name="softmax_out_stage0"),
                            merged_up=buffer(sf_tile, datatype=row_ty, name="cascade_up_stage0"),
                            merged_sp=buffer(sf_tile, datatype=row_ty, name="cascade_sp_stage0"),
                            prev_up=buffer(sf_tile, datatype=row_ty, name="prev_up_stage0"),
                            r_from_cascade=buffer(sf_tile, datatype=row_ty, name="r_from_cascade_stage0"),
                            r_from_local=buffer(sf_tile, datatype=row_ty, name="r_from_local_stage0"),
                            tmp_sp=buffer(sf_tile, datatype=row_ty, name="tmp_sp_stage0"),
                            out_dma_acquire=lock(sf_tile, lock_id=0, init=0),
                            out_ready=lock(sf_tile, lock_id=1, init=1),
                            g_copy_dma_acquire=lock(sf_tile, lock_id=2, init=0),
                            g_copy_ready=lock(sf_tile, lock_id=3, init=1),
                            g_dma_acquire=lock(sf_tile, lock_id=5, init=1),
                            g_ready=lock(sf_tile, lock_id=4, init=0),
                            gv_dma_acquire=lock(sf_tile, lock_id=7, init=1),
                            gv_ready=lock(sf_tile, lock_id=6, init=0),
                        )
                    )
                    continue

                if stage == NUM_CASCADE_STAGES - 1:
                    softmax_stages.append(
                        SoftmaxStage(
                            stage=stage,
                            tile=sf_tile,
                            gp=buffer(sf_tile, datatype=gp_ty, name=f"softmax_gp_stage{stage}"),
                            sp=buffer(sf_tile, datatype=row_ty, name=f"softmax_sp_stage{stage}"),
                            up=buffer(sf_tile, datatype=row_ty, name=f"softmax_up_stage{stage}"),
                            u=buffer(sf_tile, datatype=row_ty, name=f"softmax_u_stage{stage}"),
                            r=buffer(sf_tile, datatype=row_ty, name=f"softmax_r_stage{stage}"),
                            s=buffer(sf_tile, datatype=row_ty, name=f"softmax_s_stage{stage}"),
                            g=buffer(sf_tile, datatype=g_ty, name=f"softmax_g_stage{stage}"),
                            g_copy=buffer(sf_tile, datatype=g_ty, name=f"softmax_g_copy_stage{stage}"),
                            gv=buffer(sf_tile, datatype=gp_ty, name=f"softmax_gv_stage{stage}"),
                            merged_gp=None,
                            merged_up=None,
                            merged_sp=None,
                            prev_up=None,
                            r_from_cascade=None,
                            r_from_local=None,
                            tmp_sp=None,
                            out_dma_acquire=None,
                            out_ready=None,
                            g_copy_dma_acquire=lock(sf_tile, lock_id=0, init=0),
                            g_copy_ready=lock(sf_tile, lock_id=1, init=1),
                            g_dma_acquire=lock(sf_tile, lock_id=3, init=1),
                            g_ready=lock(sf_tile, lock_id=2, init=0),
                            gv_dma_acquire=lock(sf_tile, lock_id=5, init=1),
                            gv_ready=lock(sf_tile, lock_id=4, init=0),
                        )
                    )
                    continue

                softmax_stages.append(
                    SoftmaxStage(
                        stage=stage,
                        tile=sf_tile,
                        gp=buffer(sf_tile, datatype=gp_ty, name=f"softmax_gp_stage{stage}"),
                        sp=buffer(sf_tile, datatype=row_ty, name=f"softmax_sp_stage{stage}"),
                        up=buffer(sf_tile, datatype=row_ty, name=f"softmax_up_stage{stage}"),
                        u=buffer(sf_tile, datatype=row_ty, name=f"softmax_u_stage{stage}"),
                        r=buffer(sf_tile, datatype=row_ty, name=f"softmax_r_stage{stage}"),
                        s=buffer(sf_tile, datatype=row_ty, name=f"softmax_s_stage{stage}"),
                        g=buffer(sf_tile, datatype=g_ty, name=f"softmax_g_stage{stage}"),
                        g_copy=buffer(sf_tile, datatype=g_ty, name=f"softmax_g_copy_stage{stage}"),
                        gv=buffer(sf_tile, datatype=gp_ty, name=f"softmax_gv_stage{stage}"),
                        merged_gp=buffer(sf_tile, datatype=gp_ty, name=f"cascade_gp_stage{stage}"),
                        merged_up=buffer(sf_tile, datatype=row_ty, name=f"cascade_up_stage{stage}"),
                        merged_sp=buffer(sf_tile, datatype=row_ty, name=f"cascade_sp_stage{stage}"),
                        prev_up=buffer(sf_tile, datatype=row_ty, name=f"prev_up_stage{stage}"),
                        r_from_cascade=buffer(sf_tile, datatype=row_ty, name=f"r_from_cascade_stage{stage}"),
                        r_from_local=buffer(sf_tile, datatype=row_ty, name=f"r_from_local_stage{stage}"),
                        tmp_sp=buffer(sf_tile, datatype=row_ty, name=f"tmp_sp_stage{stage}"),
                        out_dma_acquire=None,
                        out_ready=None,
                        g_copy_dma_acquire=lock(sf_tile, lock_id=0, init=0),
                        g_copy_ready=lock(sf_tile, lock_id=1, init=1),
                        g_dma_acquire=lock(sf_tile, lock_id=3, init=1),
                        g_ready=lock(sf_tile, lock_id=2, init=0),
                        gv_dma_acquire=lock(sf_tile, lock_id=5, init=1),
                        gv_ready=lock(sf_tile, lock_id=4, init=0),
                    )
                )

            for stage in qk_stages:
                emit_qk_mem(stage)
                emit_qk_core(stage, kernels)

            for stage in gv_stages:
                emit_gv_mem(stage)
                emit_gv_core(stage, kernels)

            for stage in softmax_stages:
                emit_softmax_mem(stage)
                emit_softmax_core(stage, kernels)

            emit_qk_v_memtile0_dma(qk_memtiles[0], v_memtiles[3])
            emit_qk_memtile_dma(qk_memtiles[1])
            emit_qk_memtile_dma(qk_memtiles[2])
            emit_qk_memtile_dma(qk_memtiles[3])
            emit_v_memtile_dma(v_memtiles[0])
            emit_v_memtile_dma(v_memtiles[1])
            emit_v_memtile_dma(v_memtiles[2])
            emit_output_memtile_dma(out_tile, out_l2, out_wait, out_ready)

            emit_runtime_sequence(
                allocations,
                q_host_ty,
                k_host_ty,
                v_host_ty,
                mask_host_ty,
                out_host_ty,
            )

        if not ctx.module.operation.verify():
            raise RuntimeError("Generated MLIR failed verification")
        return ctx.module


def stage_kernel_object(work_dir: Path) -> Path:
    staged_kernel_object = work_dir / KERNEL_OBJECT
    if staged_kernel_object.exists():
        return staged_kernel_object
    if not REFERENCE_KERNEL_OBJECT.exists():
        raise FileNotFoundError(
            "Missing flash-attention kernel object. Expected to find "
            f"{REFERENCE_KERNEL_OBJECT} so it can be staged into {work_dir}."
        )
    copy2(REFERENCE_KERNEL_OBJECT, staged_kernel_object)
    return staged_kernel_object


def load_flash_attention_kernel(work_dir: Path) -> tuple[NPUKernel, object]:
    insts_path = work_dir / "insts.bin"
    xclbin_path = work_dir / "final.xclbin"

    npu_kernel = NPUKernel(
        str(xclbin_path),
        str(insts_path),
        kernel_name="MLIR_AIE",
    )
    DefaultNPURuntime.cleanup()
    return npu_kernel, DefaultNPURuntime.load(npu_kernel)


def run_flash_attention_once(
    work_dir: Path,
    Q: np.ndarray,
    K_mat: np.ndarray,
    V: np.ndarray,
    M_mask: np.ndarray,
    *,
    retry_on_timeout: bool,
) -> tuple[np.ndarray, float, int]:
    npu_kernel, kernel_handle = load_flash_attention_kernel(work_dir)
    retries = 0
    in_q = iron.tensor(Q.reshape(-1), dtype=Q.dtype)
    in_k = iron.tensor(K_mat.reshape(-1), dtype=K_mat.dtype)
    in_v = iron.tensor(V.reshape(-1), dtype=V.dtype)
    in_m = iron.tensor(M_mask.reshape(-1), dtype=M_mask.dtype)
    out = iron.zeros(OUTPUT_SIZE, dtype=Q.dtype)

    try:
        try:
            result = DefaultNPURuntime.run(kernel_handle, [in_q, in_k, in_v, in_m, out])
        except HostRuntimeError as err:
            if not retry_on_timeout or "ERT_CMD_STATE_TIMEOUT" not in str(err):
                raise
            retries = 1
            DefaultNPURuntime.cleanup()
            gc.collect()
            kernel_handle = DefaultNPURuntime.load(npu_kernel)
            result = DefaultNPURuntime.run(kernel_handle, [in_q, in_k, in_v, in_m, out])

        observed = np.array(out.numpy()).reshape(LQ, DV)
        return observed, float(result.npu_time), retries
    finally:
        DefaultNPURuntime.cleanup()
        del in_q, in_k, in_v, in_m, out
        gc.collect()


def validate_flash_attention(
    work_dir: Path,
    Q: np.ndarray,
    K_mat: np.ndarray,
    V: np.ndarray,
    M_mask: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float, bool]:
    print("[3/3] Running on NPU and validating results")
    observed, npu_time, retries = run_flash_attention_once(
        work_dir,
        Q,
        K_mat,
        V,
        M_mask,
        retry_on_timeout=True,
    )

    max_abs = float(
        np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32)))
    )
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
    if retries:
        print(f"Kernel runtime retries after timeout recovery: {retries}")
    print(f"Max absolute error: {max_abs:.6f}")
    print(f"Max relative error: {max_rel:.6f}")
    print(f"Output[0,:8]: {observed[0, :8]}")
    print(f"Expect[0,:8]: {expected[0, :8]}")

    return max_abs, max_rel, is_close


def run_benchmark(
    work_dir: Path,
    Q: np.ndarray,
    K_mat: np.ndarray,
    V: np.ndarray,
    M_mask: np.ndarray,
    warmup: int,
    iterations: int,
) -> int:
    import time

    macs = float(LQ * LK * DK * 2 + LK * LQ * DV * 2)
    total_iters = warmup + iterations
    times_us = []
    retries = 0

    print(
        f"[3/3] Benchmarking flash attention: lq={LQ}, lk={LK}, dk={DK}, dv={DV}"
    )
    print(f"      {warmup} warmup + {iterations} measurement iterations")

    npu_kernel, kernel_handle = load_flash_attention_kernel(work_dir)
    try:
        for i in range(total_iters):
            while True:
                in_q = iron.tensor(Q.reshape(-1), dtype=Q.dtype)
                in_k = iron.tensor(K_mat.reshape(-1), dtype=K_mat.dtype)
                in_v = iron.tensor(V.reshape(-1), dtype=V.dtype)
                in_m = iron.tensor(M_mask.reshape(-1), dtype=M_mask.dtype)
                out = iron.zeros(OUTPUT_SIZE, dtype=Q.dtype)

                try:
                    t0 = time.perf_counter()
                    DefaultNPURuntime.run(kernel_handle, [in_q, in_k, in_v, in_m, out])
                    t1 = time.perf_counter()
                    break
                except HostRuntimeError as err:
                    if "ERT_CMD_STATE_TIMEOUT" not in str(err):
                        raise
                    retries += 1
                    DefaultNPURuntime.cleanup()
                    gc.collect()
                    kernel_handle = DefaultNPURuntime.load(npu_kernel)
                finally:
                    del in_q, in_k, in_v, in_m, out
                    gc.collect()

            elapsed_us = (t1 - t0) * 1e6
            if i >= warmup:
                times_us.append(elapsed_us)
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
    if retries:
        print(f"      Runtime reload retries: {retries}")
    return 0


def main() -> int:
    args = parse_args()
    ensure_supported_configuration(args)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    Q, K_mat, V, M_mask = generate_test_data(LQ, DK, LK, DV)
    expected = None if args.benchmark else flash_attention_golden(
        Q, K_mat, V, M_mask, lk=LK, lkp=LKP
    )

    if args.verbose:
        print("Flash Attention direct AIE emitter")
        print(f"  reference aie.mlir: {REFERENCE_AIE_MLIR}")
        print(f"  reference npu.mlir: {REFERENCE_NPU_MLIR}")
        print(f"  kernel source:      {KERNEL_SOURCE}")
        print(f"  kernel object:      {KERNEL_OBJECT}")
        print(f"  shape Q={Q.shape} K={K_mat.shape} V={V.shape}")
        print(f"  chunks={NUM_CHUNKS} chunks_per_stage={CHUNKS_PER_STAGE}")
        if expected is not None:
            print(f"  expected[0,:4]={expected[0, :4]}")

    module = build_mlir_module()
    mlir_path = work_dir / "flash_attention_direct_aie.mlir"
    with open(mlir_path, "w", encoding="utf-8") as handle:
        print(module, file=handle)

    if args.print_module_only:
        print(module)
        return 0

    print(f"Wrote direct AIE module to {mlir_path}")

    kernel_object_path = stage_kernel_object(work_dir)
    print(f"Staged kernel object to {kernel_object_path}")

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
        return run_benchmark(
            work_dir,
            Q,
            K_mat,
            V,
            M_mask,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    assert expected is not None
    _, _, is_close = validate_flash_attention(
        work_dir,
        Q,
        K_mat,
        V,
        M_mask,
        expected,
    )
    if is_close:
        print("PASS!")
        return 0

    print("FAILED: output does not match the NumPy golden within tolerance")
    return 1


if __name__ == "__main__":
    sys.exit(main())
