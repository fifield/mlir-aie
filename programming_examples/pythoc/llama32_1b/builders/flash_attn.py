# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b prefill flash-attention kernel.

Replaces the cached AIR-stitched ``flash_attn.npu.air.mlir`` with an
mlir-aie Python program that emits structurally equivalent
``aie/aiex``-dialect text using the dialect Python bindings directly.

Compared to the standalone ``flash_attention/flash_attention_kernel_fusion.py``
scaffold this builder:
  * replaces the per-chunk softmax chain
    (max_g / maximum_up_u / exp_g_minus_u / exp_up_minus_u / vector_copy_32 /
     sum_g / mul_r_gp / ...) with the two fused kernel calls
    ``apply_causal_mask(g, q_block, kv_block)`` and
    ``fused_softmax(g_flat, up, s, r)`` — both already ship in
    ``attn_pythoc.o`` (via ``compile_attn``).
  * adds one ``memref<3xi32, 2 : i32>`` "causal counter" scratch buffer
    per compute tile (32 extra L1 buffers) that tracks the rolling
    ``(q_block, boot_flag, head_local)`` triple used to drive
    ``apply_causal_mask`` and the per-iteration q_block advancement
    inside the herd's ``while-true`` loop.
  * scales the dispatcher to ``seq_len=2048`` — 128 q_groups instead
    of the 2 q_groups the standalone scaffold ships with — and switches
    the runtime sequence to use the cached MLIR's exact offset formula
    (``Q[rg*524288 + head*64]``, ``K/V[stage*262144 + kv_head*64]``,
    ``O[rg*524288 + q_col*131072 + head*64]``).
  * wraps the 32-core compute device as ``aie.device(npu2) @attn_seg``
    and produces an outer dispatcher
    ``aie.device(npu2) { aie.runtime_sequence @attention_bf16 ... }``
    with a single ``aiex.configure @attn_seg / aiex.run @attn_seg_sequence``
    call — matching the cached MLIR exactly.

References:
  * ``reference_mlir/flash_attn.npu.air.mlir`` -- ground truth (29,734 lines,
    produced by AIR's aircc).
  * ``../flash_attention/flash_attention_kernel_fusion.py`` -- the topology
    scaffold (non-causal variant, 2 q_groups).
  * ``../../mlir-air/programming_examples/flash_attention/kernel_fusion_based/
    attn_npu2_seqfirst.py`` -- the AIR-level builder this kernel was lowered
    from (causal=True, lq=lk=2048, num_heads=32, num_kv_heads=8).
  * ``kernels/attn.py`` -- ``apply_causal_mask`` (line 935) and
    ``fused_softmax`` (line 1010).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

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
from ._emit import attach_loop_annotation_to_all_scf_for
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.ir import AffineDimExpr, AffineMap, InsertionPoint, MemRefType


# ---------------------------------------------------------------------------
# Fixed lowered configuration (Llama-3.2-1B prefill flash-attention).
# Must match the cached AIR-stitched IR.
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64

LK = SEQ_LEN
LQ = SEQ_LEN
LKP = 64
LQP = 256
DK = HEAD_DIM
DV = HEAD_DIM

NUM_Q_TILES = 4
NUM_CASCADE_STAGES = 4
NUM_SEGMENTS = 2
NUM_HEADS_PER_UNROLL = 2

GQA_GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS  # 32 // 8 = 4

NUM_LQ_ITERS = LQ // LQP                                          # 2048/256 = 8
NUM_HEAD_GROUPS = NUM_HEADS // NUM_HEADS_PER_UNROLL                # 32/2 = 16
Q_GROUPS = NUM_LQ_ITERS * NUM_HEAD_GROUPS                          # 8*16 = 128
CHUNKS_PER_STAGE = LK // (LKP * NUM_CASCADE_STAGES)                # 2048/(64*4) = 8
LK_PER_STAGE = LKP * CHUNKS_PER_STAGE                              # 512

Q_TILE_ROWS = LQP // NUM_Q_TILES                                   # 256/4 = 64

QK_TILE_SIZE = Q_TILE_ROWS * DK                                    # 64*64 = 4096
V_TILE_SIZE = LKP * DV                                             # 64*64 = 4096
G_TILE_SIZE = Q_TILE_ROWS * LKP                                    # 64*64 = 4096
OUTPUT_TILE_SIZE = Q_TILE_ROWS * DV                                # 64*64 = 4096
ROW_SIZE = Q_TILE_ROWS                                             # 64

EMB_DIM_Q = NUM_HEADS * DK                                         # 2048
EMB_DIM_KV = NUM_KV_HEADS * DK                                     # 512
EMB_DIM_OUT = NUM_HEADS * DV                                       # 2048

# Per-q_group L3 strides
Q_HEAD_STRIDE = DK                                                 # head_off = head * 64
LQ_ITER_STRIDE = LQP * EMB_DIM_Q                                   # 256*2048 = 524288
KV_HEAD_STRIDE = DK                                                # kv_head_off = kv_head * 64
KV_STAGE_STRIDE = LK_PER_STAGE * EMB_DIM_KV                        # 512*512 = 262144
OUT_QCOL_STRIDE = Q_TILE_ROWS * EMB_DIM_OUT                        # 64*2048 = 131072

KERNEL_OBJECT = "attn_pythoc.o"


# ---------------------------------------------------------------------------
# Specs (mirror the standalone scaffold, with an added causal counter).
# ---------------------------------------------------------------------------
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
    counter: object  # memref<3xi32, 2 : i32>
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


@dataclass(frozen=True)
class KernelSet:
    zero_fill_g: object
    zero_fill_gp: object
    zero_fill_sp: object
    neg_inf_fill_up: object
    copy_tile: object
    matmul_a_b: object
    matmul_g_b: object
    apply_causal_mask: object
    fused_softmax: object
    mul_r_gp: object
    accum_sp_r_s: object
    vector_copy_32: object
    maximum_up_u: object
    exp_up_minus_u: object
    add_gp_g: object
    div_gp_sp: object


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
def _bf16_memref(*shape, memory_space=None):
    from aie.ir import IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


def _i32_memref(*shape, memory_space=None):
    from aie.ir import IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.i32(), None, ms)


def _collapsed_memref_type(buffer_ref: object, total_elems: int) -> MemRefType:
    buffer_ty = MemRefType(buffer_ref.type)
    return MemRefType.get(
        (total_elems,),
        buffer_ty.element_type,
        None,
        buffer_ty.memory_space,
    )


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


# ---------------------------------------------------------------------------
# Kernel declarations
# ---------------------------------------------------------------------------
def _declare_kernels() -> KernelSet:
    """Declare the 16 external_funcs needed by the prefill flash-attention.

    All kernels live in ``attn_pythoc.o`` (built by ``compile_attn`` from
    ``kernels/attn.py``).  Plain symbol names (no ``_pythoc`` suffix) match
    the cached IR's func.func declarations.
    """
    from aie.ir import UnitAttr

    qk_ty = _bf16_memref(Q_TILE_ROWS, DK, memory_space=2)
    q_ty = _bf16_memref(Q_TILE_ROWS, DK, memory_space=2)
    v_ty = _bf16_memref(LKP, DV, memory_space=2)
    g_flat_ty = _bf16_memref(G_TILE_SIZE, memory_space=2)
    g_2d_ty = _bf16_memref(Q_TILE_ROWS, LKP, memory_space=2)
    gp_ty = _bf16_memref(Q_TILE_ROWS, DV, memory_space=2)
    row_ty = _bf16_memref(Q_TILE_ROWS, 1, memory_space=2)

    def _ef(name, inputs):
        fn = external_func(name, inputs=inputs, link_with=KERNEL_OBJECT)
        fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return fn

    return KernelSet(
        zero_fill_g=_ef("zero_fill_g_bf16", [g_flat_ty]),
        zero_fill_gp=_ef("zero_fill_gp_bf16", [gp_ty]),
        zero_fill_sp=_ef("zero_fill_sp_bf16", [row_ty]),
        neg_inf_fill_up=_ef("neg_inf_fill_up_bf16", [row_ty]),
        copy_tile=_ef("copy_tile", [qk_ty, q_ty]),
        matmul_a_b=_ef("matmul_a_b_bf16", [q_ty, qk_ty, g_flat_ty]),
        matmul_g_b=_ef("matmul_g_b_bf16", [g_flat_ty, v_ty, gp_ty]),
        # apply_causal_mask takes the 2D G memref (64x64) plus q_block, kv_block.
        apply_causal_mask=_ef("apply_causal_mask", [g_2d_ty, T.i32(), T.i32()]),
        fused_softmax=_ef("fused_softmax", [g_flat_ty, row_ty, row_ty, row_ty]),
        mul_r_gp=_ef("mul_r_gp", [row_ty, gp_ty]),
        accum_sp_r_s=_ef("accum_sp_r_s", [row_ty, row_ty, row_ty]),
        vector_copy_32=_ef("vector_copy_32elems", [T.i32(), row_ty, row_ty]),
        maximum_up_u=_ef("maximum_up_u_bf16", [row_ty, row_ty]),
        exp_up_minus_u=_ef("exp_up_minus_u", [row_ty, row_ty, row_ty]),
        add_gp_g=_ef("add_gp_g", [gp_ty, gp_ty]),
        div_gp_sp=_ef("div_gp_sp", [row_ty, gp_ty]),
    )


# ---------------------------------------------------------------------------
# Topology emitters
# ---------------------------------------------------------------------------
def _emit_flows(
    shim_tiles: dict[int, object],
    mem_tiles: dict[int, object],
    compute_tiles: dict[tuple[int, int], object],
) -> None:
    """Emit shim <-> mem <-> compute flows and cascade flows.

    Topology per segment (4 columns each):
      - shim/mem-DMA0 (weights/Q/K route)
      - shim/mem-DMA1 (V route)
      - mem/shim-DMA0 (output route back)
      - mem(stage_col)/compute(row=2+stage) DMA1+DMA2 broadcast for QK+V
      - compute(row=2)/mem(q_col) DMA0 for output
      - cascade(row=5) -> cascade(row=4) -> cascade(row=3) -> cascade(row=2)
    """
    for segment in range(NUM_SEGMENTS):
        base = segment * 4

        for idx in range(4):
            flow(shim_tiles[base + idx], WireBundle.DMA, 0,
                 mem_tiles[base + idx], WireBundle.DMA, 0)
            flow(shim_tiles[base + idx], WireBundle.DMA, 1,
                 mem_tiles[base + idx], WireBundle.DMA, 1)
            flow(mem_tiles[base + idx], WireBundle.DMA, 0,
                 shim_tiles[base + idx], WireBundle.DMA, 0)

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


def _emit_shim_allocations(
    shim_tiles: dict[int, object]
) -> dict[str, list[list[object]]]:
    """Emit the 24 shim_dma_allocations matching the cached IR's names."""
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


# ---------------------------------------------------------------------------
# Mem tile DMA
# ---------------------------------------------------------------------------
def _emit_memtile_dma(spec: MemTileSpec) -> None:
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


# ---------------------------------------------------------------------------
# Compute tile mem DMA
# ---------------------------------------------------------------------------
def _emit_compute_mem(spec: ComputeTileSpec) -> None:
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


# ---------------------------------------------------------------------------
# Cascade helpers
# ---------------------------------------------------------------------------
def _emit_cascade_send(buffer_ref: object, total_elems: int,
                       zero_bf16: object, c0: object) -> None:
    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
    flat = memref.collapse_shape(
        _collapsed_memref_type(buffer_ref, total_elems),
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


def _emit_cascade_receive(buffer_ref: object, total_elems: int, c0: object) -> None:
    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
    flat = memref.collapse_shape(
        _collapsed_memref_type(buffer_ref, total_elems),
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


# ---------------------------------------------------------------------------
# Selective Q capture — per-tile (stage, q_col): receive all NQ Q tiles
# but only call copy_tile when q_index == q_col.
# ---------------------------------------------------------------------------
def _emit_selective_q_capture(spec: ComputeTileSpec, kernels: KernelSet) -> None:
    for q_index in range(NUM_Q_TILES):
        use_lock(spec.qk_ready, LockAction.AcquireGreaterEqual, value=1)
        if q_index == spec.q_col:
            kernels.copy_tile(spec.qk, spec.q)
        use_lock(spec.qk_dma_acquire, LockAction.Release, value=1)


# ---------------------------------------------------------------------------
# Compute core body
# ---------------------------------------------------------------------------
def _emit_compute_core(spec: ComputeTileSpec, kernels: KernelSet) -> None:
    """Emit the AIE core compute body for one (segment, stage, q_col) tile.

    The body mirrors the cached MLIR exactly:

        loop forever {
          zero_fill_gp / zero_fill_sp / neg_inf_fill_up
          if counter[1] == 0: counter[:] = [0, 1, 0]    # boot once
          for q_index in 0..NQ: { acquire qk_ready; if q==q_col copy_tile; release }
          [if stage==0: acquire out_ready BEFORE init? -- happens at top of loop]
          for chunk in 0..CHUNKS_PER_STAGE: {
            zero_fill_g; acquire qk_ready; matmul(q, qk, g); release;
            acquire v_ready;
            apply_causal_mask(g_2d, counter[0]+q_col, chunk + stage*chunks_per_stage)
            fused_softmax(g, up, s, r)
            mul_r_gp(r, gp); matmul_g_b(g, v, gp);
            accum_sp_r_s(sp, r, s); vector_copy(0, s, sp); release v;
          }
          [stage == NS-1: cascade send (gp, up, sp)]
          [stage in 1..NS-2: cascade receive + merge + cascade send]
          [stage == 0: cascade receive + merge + div_gp_sp + release out_dma_acquire]
          counter[2] += 1; if counter[2] >= 16: counter[0] += 4, counter[2] = 0
        }

    Notes on the cached MLIR ordering:
      * stage 0: ``acquire out_ready`` happens BEFORE the zero_fill_gp init
        (line 2234 in flash_attn.npu.air.mlir), but stage>0 tiles don't
        have an out_ready lock at all.
      * counter[0] is the cumulative q_block base (advances by NQ=4 per
        head-group cycle of NUM_HEAD_GROUPS=16).
      * counter[1] is the one-shot boot flag.
      * counter[2] is the head-local counter (mod NUM_HEAD_GROUPS).
    """
    @core(spec.tile)
    def compute_core_body():
        c0 = arith.constant(0, index=True)
        c1 = arith.constant(1, index=True)
        c2 = arith.constant(2, index=True)
        c0_i32 = arith.constant(0, T.i32())
        c1_i32 = arith.constant(1, T.i32())
        c_nq_i32 = arith.constant(NUM_Q_TILES, T.i32())
        c_chunks_i32 = arith.constant(CHUNKS_PER_STAGE, T.i32())
        c_total_hg_i32 = arith.constant(NUM_HEAD_GROUPS, T.i32())
        c_stage_off_i32 = arith.constant(spec.stage * CHUNKS_PER_STAGE, T.i32())
        c_qcol_i32 = arith.constant(spec.q_col, T.i32())
        zero_bf16 = arith.constant(0.0, T.bf16())

        for _ in range_(sys.maxsize):
            # Stage 0 acquires the output buffer lock BEFORE re-initializing.
            if spec.stage == 0:
                use_lock(spec.out_ready, LockAction.AcquireGreaterEqual, value=1)

            kernels.zero_fill_gp(spec.gp)
            kernels.zero_fill_sp(spec.sp)
            kernels.neg_inf_fill_up(spec.up)

            # Boot the causal counter: counter[1] is a one-shot flag.
            boot = memref.load(spec.counter, [c1])
            is_first = arith.cmpi("eq", boot, c0_i32)
            from aie.dialects import scf as _scf

            if_first = _scf.IfOp(is_first)
            with InsertionPoint(if_first.then_block):
                memref.store(c0_i32, spec.counter, [c0])
                memref.store(c1_i32, spec.counter, [c1])
                memref.store(c0_i32, spec.counter, [c2])
                _scf.YieldOp([])

            # Selective Q capture (4 acquire/release lock cycles, with the
            # copy_tile happening at iteration == q_col).
            _emit_selective_q_capture(spec, kernels)

            # Inner chunk loop (CHUNKS_PER_STAGE = 8).
            from aie.extras.dialects.arith import index_cast
            for arg0_idx in range_(0, CHUNKS_PER_STAGE, 1, iter_args=None):
                g_flat_ty = _collapsed_memref_type(spec.g, G_TILE_SIZE)
                g_flat = memref.collapse_shape(g_flat_ty, spec.g, [[0, 1]])
                kernels.zero_fill_g(g_flat)

                use_lock(spec.qk_ready, LockAction.AcquireGreaterEqual, value=1)
                kernels.matmul_a_b(spec.q, spec.qk, g_flat)
                use_lock(spec.qk_dma_acquire, LockAction.Release, value=1)

                use_lock(spec.v_ready, LockAction.AcquireGreaterEqual, value=1)

                # Causal mask uses q_block = counter[0] + q_col, kv_block =
                # arg0 + stage*chunks_per_stage.  When stage*chunks_per_stage
                # == 0 the addition is folded away (matches the cached IR for
                # stage 0, which omits the addi).
                arg0 = index_cast(arg0_idx, to=T.i32())
                q_base = memref.load(spec.counter, [c0])
                q_block = arith.addi(q_base, c_qcol_i32)
                if spec.stage == 0:
                    kv_block = arg0
                else:
                    kv_block = arith.addi(arg0, c_stage_off_i32)
                kernels.apply_causal_mask(spec.g, q_block, kv_block)

                kernels.fused_softmax(g_flat, spec.up, spec.s, spec.r)
                kernels.mul_r_gp(spec.r, spec.gp)
                kernels.matmul_g_b(g_flat, spec.v, spec.gp)
                kernels.accum_sp_r_s(spec.sp, spec.r, spec.s)
                kernels.vector_copy_32(c0_i32, spec.s, spec.sp)
                use_lock(spec.v_dma_acquire, LockAction.Release, value=1)

            if spec.stage == NUM_CASCADE_STAGES - 1:
                _emit_cascade_send(spec.gp, OUTPUT_TILE_SIZE, zero_bf16, c0)
                _emit_cascade_send(spec.up, ROW_SIZE, zero_bf16, c0)
                _emit_cascade_send(spec.sp, ROW_SIZE, zero_bf16, c0)
                _emit_counter_increment(spec, c0, c2, c0_i32, c1_i32, c_total_hg_i32)
                continue

            # stage in 0 .. NS-2: receive cascade and merge.
            _emit_cascade_receive(spec.merged_gp, OUTPUT_TILE_SIZE, c0)
            _emit_cascade_receive(spec.merged_up, ROW_SIZE, c0)
            _emit_cascade_receive(spec.merged_sp, ROW_SIZE, c0)

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
                _emit_counter_increment(spec, c0, c2, c0_i32, c1_i32, c_total_hg_i32)
                continue

            _emit_cascade_send(spec.merged_gp, OUTPUT_TILE_SIZE, zero_bf16, c0)
            _emit_cascade_send(spec.up, ROW_SIZE, zero_bf16, c0)
            _emit_cascade_send(spec.merged_sp, ROW_SIZE, zero_bf16, c0)
            _emit_counter_increment(spec, c0, c2, c0_i32, c1_i32, c_total_hg_i32)

    # The link_with attribute is added by aie-assign-core-link-files based
    # on the external_func declarations' link_with -- no manual attachment
    # needed here.


def _emit_counter_increment(spec, c0_idx, c2_idx, c0_i32, c1_i32, c_total_hg_i32):
    """Increment counter[2]; if >= NUM_HEAD_GROUPS, wrap to 0 and bump counter[0] by NQ."""
    from aie.dialects import scf as _scf

    c_nq_i32 = arith.constant(NUM_Q_TILES, T.i32())

    head_cur = memref.load(spec.counter, [c2_idx])
    head_next = arith.addi(head_cur, c1_i32)
    wrapped = arith.cmpi("sge", head_next, c_total_hg_i32)
    if_wrap = _scf.IfOp(wrapped)
    with InsertionPoint(if_wrap.then_block):
        q_cur = memref.load(spec.counter, [c0_idx])
        q_next = arith.addi(q_cur, c_nq_i32)
        memref.store(q_next, spec.counter, [c0_idx])
        memref.store(c0_i32, spec.counter, [c2_idx])
        _scf.YieldOp([])

    not_wrapped = arith.cmpi("slt", head_next, c_total_hg_i32)
    if_no_wrap = _scf.IfOp(not_wrapped)
    with InsertionPoint(if_no_wrap.then_block):
        memref.store(head_next, spec.counter, [c2_idx])
        _scf.YieldOp([])


# ---------------------------------------------------------------------------
# Runtime sequence — 128 q_groups, 32 dma_configure_task_for each.
# ---------------------------------------------------------------------------
def _emit_runtime_sequence(
    allocations: dict[str, list[list[object]]],
    q_host_ty,
    k_host_ty,
    v_host_ty,
    out_host_ty,
    sym_name: str,
) -> None:
    """Emit the attn_seg_sequence runtime sequence.

    For each q_group ∈ [0, Q_GROUPS=128):
      - lq_iter = q_group // NUM_HEAD_GROUPS  (8 values, 0..7)
      - head_pair = q_group % NUM_HEAD_GROUPS  (16 values, 0..15)
      - For each segment ∈ {0, 1} (= head_local):
        - head = head_pair * 2 + segment
        - kv_head = head // GQA_GROUP_SIZE
        - Q offset = lq_iter * LQ_ITER_STRIDE + head * Q_HEAD_STRIDE
        - K stage_s offset = s * KV_STAGE_STRIDE + kv_head * KV_HEAD_STRIDE
        - V stage_s offset = s * KV_STAGE_STRIDE + kv_head * KV_HEAD_STRIDE
        - Out q_col offset = lq_iter * LQ_ITER_STRIDE + q_col * OUT_QCOL_STRIDE + head * Q_HEAD_STRIDE

    Tasks per q_group: 8 Q+K (4 stages × 2 starts: Q then K reuse the
    same air_QKIn shim) + 4 V + 4 outputs = 16 per segment × 2 = 32.
    """
    @runtime_sequence(q_host_ty, k_host_ty, v_host_ty, out_host_ty,
                      sym_name=sym_name)
    def attn_seg_sequence(q, k, v, out):
        for q_group in range(Q_GROUPS):
            lq_iter = q_group // NUM_HEAD_GROUPS
            head_pair = q_group % NUM_HEAD_GROUPS

            q_tasks: dict[tuple[int, int], object] = {}
            k_tasks: dict[tuple[int, int], object] = {}
            v_tasks: dict[tuple[int, int], object] = {}
            out_tasks: dict[tuple[int, int], object] = {}

            for segment in range(NUM_SEGMENTS):
                head = head_pair * 2 + segment
                kv_head = head // GQA_GROUP_SIZE

                q_offset = lq_iter * LQ_ITER_STRIDE + head * Q_HEAD_STRIDE
                kv_head_offset = kv_head * KV_HEAD_STRIDE

                # 4 Q dispatches and 4 K dispatches (8 total) per segment.
                # Both ride on the same air_QKIn_s_seg_0_0 shim alloc.
                for stage in range(NUM_CASCADE_STAGES):
                    q_task = dma_configure_task_for(
                        allocations["qk"][segment][stage]
                    )
                    with bds(q_task) as bd:
                        with bd[0]:
                            dma_bd(
                                q,
                                offset=q_offset,
                                len=LQP * DK,
                                dimensions=[(LQP, EMB_DIM_Q), (DK, 1)],
                            )
                            EndOp()
                    dma_start_task(q_task)
                    q_tasks[(segment, stage)] = q_task

                    k_offset = stage * KV_STAGE_STRIDE + kv_head_offset
                    k_task = dma_configure_task_for(
                        allocations["qk"][segment][stage]
                    )
                    with bds(k_task) as bd:
                        with bd[0]:
                            dma_bd(
                                k,
                                offset=k_offset,
                                len=LK_PER_STAGE * DK,
                                dimensions=[(LK_PER_STAGE, EMB_DIM_KV), (DK, 1)],
                            )
                            EndOp()
                    dma_start_task(k_task)
                    k_tasks[(segment, stage)] = k_task

                # 4 V dispatches per segment.
                for stage in range(NUM_CASCADE_STAGES):
                    v_offset = stage * KV_STAGE_STRIDE + kv_head_offset
                    v_task = dma_configure_task_for(
                        allocations["v"][segment][stage]
                    )
                    with bds(v_task) as bd:
                        with bd[0]:
                            dma_bd(
                                v,
                                offset=v_offset,
                                len=LK_PER_STAGE * DV,
                                dimensions=[(LK_PER_STAGE, EMB_DIM_KV), (DV, 1)],
                            )
                            EndOp()
                    dma_start_task(v_task)
                    v_tasks[(segment, stage)] = v_task

                # 4 output dispatches per segment (one per q_col).
                for q_col in range(NUM_Q_TILES):
                    out_offset = (
                        lq_iter * LQ_ITER_STRIDE
                        + q_col * OUT_QCOL_STRIDE
                        + head * Q_HEAD_STRIDE
                    )
                    out_task = dma_configure_task_for(
                        allocations["out"][segment][q_col],
                        issue_token=True,
                    )
                    with bds(out_task) as bd:
                        with bd[0]:
                            dma_bd(
                                out,
                                offset=out_offset,
                                len=OUTPUT_TILE_SIZE,
                                dimensions=[(Q_TILE_ROWS, EMB_DIM_OUT), (DV, 1)],
                            )
                            EndOp()
                    dma_start_task(out_task)
                    out_tasks[(segment, q_col)] = out_task

            # Match the AIR-lowered task lifetime ordering. Order taken from
            # the cached IR's per-q_group free/await block (lines 5309-5340
            # of flash_attn.npu.air.mlir).
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


# ---------------------------------------------------------------------------
# Device assembly: the inner @attn_seg device and the outer dispatcher.
# ---------------------------------------------------------------------------
def _emit_attn_seg_device(q_host_ty, k_host_ty, v_host_ty, out_host_ty) -> None:
    """Emit the inner ``aie.device(npu2) @attn_seg { ... }`` block.

    Contains 48 tiles (8 cols × 6 rows), 192 locks, ~512 L1+L2 buffers,
    32 cores, 96 flows, 8 memtile DMAs, 24 shim allocations, and 24
    cascade flows.
    """
    tile_ty = _bf16_memref(Q_TILE_ROWS, DK, memory_space=2)
    v_tile_ty = _bf16_memref(LKP, DV, memory_space=2)
    gp_ty = _bf16_memref(Q_TILE_ROWS, DV, memory_space=2)
    row_ty = _bf16_memref(Q_TILE_ROWS, 1, memory_space=2)
    g_tile_ty = _bf16_memref(Q_TILE_ROWS, LKP, memory_space=2)
    l2_tile_ty = _bf16_memref(Q_TILE_ROWS, DK, memory_space=1)
    counter_ty = _i32_memref(3, memory_space=2)

    @device(AIEDevice.npu2, sym_name="attn_seg")
    def _device():
        external_buffer(q_host_ty, name="Q")
        external_buffer(k_host_ty, name="K")
        external_buffer(v_host_ty, name="V")
        external_buffer(out_host_ty, name="Out")

        kernels = _declare_kernels()

        shim_tiles = {col: tile(col, 0) for col in range(8)}
        mem_tiles = {col: tile(col, 1) for col in range(8)}
        compute_tiles = {(col, row): tile(col, row)
                         for col in range(8) for row in range(2, 6)}

        _emit_flows(shim_tiles, mem_tiles, compute_tiles)
        allocations = _emit_shim_allocations(shim_tiles)

        # Mem tile specs (one per column).
        mem_specs: dict[int, MemTileSpec] = {}
        for col in range(8):
            mt = mem_tiles[col]
            mem_specs[col] = MemTileSpec(
                segment=col // 4,
                index=col % 4,
                tile=mt,
                qk=buffer(mt, datatype=l2_tile_ty, name=f"qk_l2_col{col}"),
                v=buffer(mt, datatype=l2_tile_ty, name=f"v_l2_col{col}"),
                out=buffer(mt, datatype=gp_ty, name=f"out_l2_col{col}"),
                out_wait=lock(mt, lock_id=0, init=0),
                out_ready=lock(mt, lock_id=1, init=1),
                qk_wait=lock(mt, lock_id=2, init=0),
                qk_ready=lock(mt, lock_id=3, init=1),
                v_wait=lock(mt, lock_id=4, init=0),
                v_ready=lock(mt, lock_id=5, init=1),
            )

        # Compute tile specs (one per (segment, q_col, stage)).
        compute_specs: list[ComputeTileSpec] = []
        for segment in range(NUM_SEGMENTS):
            base = segment * 4
            for q_col in range(NUM_Q_TILES):
                for stage in range(NUM_CASCADE_STAGES):
                    row = 2 + stage
                    tile_ref = compute_tiles[(base + q_col, row)]

                    # Locks shared by all stages.
                    qk_dma_acquire = lock(tile_ref, lock_id=3 if stage == 0 else 1, init=1)
                    qk_ready = lock(tile_ref, lock_id=2 if stage == 0 else 0, init=0)
                    v_dma_acquire = lock(tile_ref, lock_id=5 if stage == 0 else 3, init=1)
                    v_ready = lock(tile_ref, lock_id=4 if stage == 0 else 2, init=0)

                    if stage == 0:
                        out_dma_acquire = lock(tile_ref, lock_id=0, init=0)
                        out_ready = lock(tile_ref, lock_id=1, init=1)
                    else:
                        out_dma_acquire = None
                        out_ready = None

                    # Buffer allocations.
                    qk_b = buffer(tile_ref, datatype=tile_ty,
                                  name=f"qk_seg{segment}_s{stage}_q{q_col}")
                    q_b = buffer(tile_ref, datatype=tile_ty,
                                 name=f"q_seg{segment}_s{stage}_q{q_col}")
                    v_b = buffer(tile_ref, datatype=v_tile_ty,
                                 name=f"v_seg{segment}_s{stage}_q{q_col}")
                    g_b = buffer(tile_ref, datatype=g_tile_ty,
                                 name=f"g_seg{segment}_s{stage}_q{q_col}")
                    gp_b = buffer(tile_ref, datatype=gp_ty,
                                  name=f"gp_seg{segment}_s{stage}_q{q_col}")
                    up_b = buffer(tile_ref, datatype=row_ty,
                                  name=f"up_seg{segment}_s{stage}_q{q_col}")
                    sp_b = buffer(tile_ref, datatype=row_ty,
                                  name=f"sp_seg{segment}_s{stage}_q{q_col}")
                    s_b = buffer(tile_ref, datatype=row_ty,
                                 name=f"s_seg{segment}_s{stage}_q{q_col}")
                    r_b = buffer(tile_ref, datatype=row_ty,
                                 name=f"r_seg{segment}_s{stage}_q{q_col}")
                    counter_b = buffer(tile_ref, datatype=counter_ty,
                                       name=f"ctr_seg{segment}_s{stage}_q{q_col}")

                    if stage == NUM_CASCADE_STAGES - 1:
                        merged_gp = merged_up = merged_sp = None
                        prev_up = r_from_cascade = r_from_local = tmp_sp = None
                    else:
                        merged_gp = buffer(tile_ref, datatype=gp_ty,
                                           name=f"merged_gp_seg{segment}_s{stage}_q{q_col}")
                        merged_up = buffer(tile_ref, datatype=row_ty,
                                           name=f"merged_up_seg{segment}_s{stage}_q{q_col}")
                        merged_sp = buffer(tile_ref, datatype=row_ty,
                                           name=f"merged_sp_seg{segment}_s{stage}_q{q_col}")
                        prev_up = buffer(tile_ref, datatype=row_ty,
                                         name=f"prev_up_seg{segment}_s{stage}_q{q_col}")
                        r_from_cascade = buffer(tile_ref, datatype=row_ty,
                                                name=f"r_cas_seg{segment}_s{stage}_q{q_col}")
                        r_from_local = buffer(tile_ref, datatype=row_ty,
                                              name=f"r_loc_seg{segment}_s{stage}_q{q_col}")
                        tmp_sp = buffer(tile_ref, datatype=row_ty,
                                        name=f"tmp_sp_seg{segment}_s{stage}_q{q_col}")

                    compute_specs.append(ComputeTileSpec(
                        segment=segment, stage=stage, q_col=q_col,
                        tile=tile_ref,
                        qk=qk_b, q=q_b, v=v_b, g=g_b, gp=gp_b,
                        up=up_b, sp=sp_b, s=s_b, r=r_b, counter=counter_b,
                        merged_gp=merged_gp, merged_up=merged_up,
                        merged_sp=merged_sp, prev_up=prev_up,
                        r_from_cascade=r_from_cascade,
                        r_from_local=r_from_local, tmp_sp=tmp_sp,
                        out_dma_acquire=out_dma_acquire,
                        out_ready=out_ready,
                        qk_dma_acquire=qk_dma_acquire,
                        qk_ready=qk_ready,
                        v_dma_acquire=v_dma_acquire,
                        v_ready=v_ready,
                    ))

        for spec in mem_specs.values():
            _emit_memtile_dma(spec)
        for spec in compute_specs:
            _emit_compute_mem(spec)
            _emit_compute_core(spec, kernels)

        _emit_runtime_sequence(
            allocations, q_host_ty, k_host_ty, v_host_ty, out_host_ty,
            sym_name="attn_seg_sequence",
        )


def _emit_dispatcher_device(q_host_ty, k_host_ty, v_host_ty, out_host_ty) -> None:
    """Emit the outer ``aie.device(npu2) { ... }`` dispatcher.

    Only contains an ``aiex.runtime_sequence @attention_bf16`` that fires
    the inner ``@attn_seg_sequence`` via ``aiex.configure`` + ``aiex.run``.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(q_host_ty, k_host_ty, v_host_ty, out_host_ty,
                          sym_name="attention_bf16")
        def _outer(*args):
            cfg = ConfigureOp(symbol="attn_seg")
            blk = cfg.body.blocks.append()
            with InsertionPoint(blk):
                RunOp(
                    runtime_sequence_symbol="attn_seg_sequence",
                    args=list(args),
                )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_flash_attn_module(
    seq_len: int = SEQ_LEN,
    n_heads: int = NUM_HEADS,
    n_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    *,
    verbose: bool = False,
) -> str:
    """Build the prefill flash-attention ``aie/aiex``-dialect module.

    Currently shape-locked to the Llama-3.2-1B prefill values:
        seq_len=2048, n_heads=32, n_kv_heads=8, head_dim=64.
    """
    del verbose
    if (seq_len, n_heads, n_kv_heads, head_dim) != (SEQ_LEN, NUM_HEADS,
                                                    NUM_KV_HEADS, HEAD_DIM):
        raise ValueError(
            f"flash_attn builder is fixed to seq_len={SEQ_LEN}, "
            f"n_heads={NUM_HEADS}, n_kv_heads={NUM_KV_HEADS}, "
            f"head_dim={HEAD_DIM}; got seq_len={seq_len}, "
            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
            f"head_dim={head_dim}."
        )

    q_host_ty = _bf16_np(LQ, NUM_HEADS * DK)
    k_host_ty = _bf16_np(LK, NUM_KV_HEADS * DK)
    v_host_ty = _bf16_np(LK, NUM_KV_HEADS * DV)
    out_host_ty = _bf16_np(LQ, NUM_HEADS * DV)

    with mlir_mod_ctx() as ctx:
        _emit_attn_seg_device(q_host_ty, k_host_ty, v_host_ty, out_host_ty)
        _emit_dispatcher_device(q_host_ty, k_host_ty, v_host_ty, out_host_ty)
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs cached MLIR).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="Output path (default: stdout)",
                        default=None)
    args = parser.parse_args()
    text = build_flash_attn_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
