# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the GQA-batched BFP576 decode-attention kernel.

Validation ladder (see ``ATTN_DECODE_GQA_SCOPE.md`` and
``tools/test_decode_attn_npu.py``):

  * ``build_tiling_probe_module()``  -- ITERATION 1: pure DMA round-trip of
    one 64x64 bf16 tile through the column-block-major 8x8-tiled L1 and back
    out, single core (0,2), no compute.  Verifies the tiling DMA dims.

  * ``build_decode_attn_module(seq_len)`` -- ITERATIONS 2-4: the actual
    decode attention.  Single GQA group per dispatch:
        zero_fill_g -> matmul_a_b_bf16(q, k) -> fused_softmax
                    -> matmul_g_b_bf16(g, v) -> [div] -> un-tile out
    q tile = GROUP_SIZE=4 real heads (rows 0-3) + zero-padded rows 4-63;
    k/v tiles = 64 KV positions x 64 head_dim.  No causal mask.  The
    1/sqrt(head_dim) scale is folded into fused_softmax's exp2 (log2e/8),
    so the host feeds q UNSCALED.

Modeled structurally on ``builders/awq_matvec.py`` (single-core builder:
locks, mem block, core forever-loop, packet flows, shim_dma_allocation,
runtime_sequence, dispatcher device).  The tiling DMA dims come from
``builders/flash_attn.py`` (memtile MM2S in = ``[(8,8),(64,64),(8,1)]``;
compute-tile MM2S out un-tile = ``[(64,8),(8,512),(8,1)]``).
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects import memref, vector
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    buffer,
    core,
    device,
    dma_bd,
    dma_start,
    external_buffer,
    external_func,
    flow,
    lock,
    mem,
    next_bd,
    packetflow,
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
from aie.ir import InsertionPoint, MemRefType

from ._emit import attach_loop_annotation_to_all_scf_for


# ---------------------------------------------------------------------------
# Fixed config (Llama-3.2-1B decode attention).
# ---------------------------------------------------------------------------
HEAD_DIM = 64
GROUP_SIZE = 4                       # 32 q-heads / 8 kv-heads
TILE_ROWS = 64                       # M dim of the BFP576 tile (4 used)
KVP = 64                             # KV positions per tile
TILE_SIZE = TILE_ROWS * HEAD_DIM     # 4096

KERNEL_OBJECT = "attn_pythoc.o"

# Tiling DMA dims.  In mlir-aie the ``dimensions`` on a shim ``dma_bd``
# describe the access pattern into the HOST buffer; the L1 side (aie.mem
# dma_bd) is linear.  So:
#   in : L1[linear] = host_in[TILE_IN_DIMS-offset]
#   out: host_out[TILE_OUT_DIMS-offset] = L1[linear]
# For a direct host round-trip through the column-block-major tiled L1 the
# un-tile dims must be the *same* host access pattern as the in dims (both
# enumerate (col_block, row, col_in_block) over the host natural offset
# row*64 + col_block*8 + col_in_block).  VERIFIED by numpy AND on HW.
#
# NOTE: flash_attn.py's compute-tile output un-tile ``[(64,8),(8,512),(8,1)]``
# is NOT this -- it applies to a different (multi-level L2) buffer, not a
# direct host round-trip.  Do not reuse it here.
TILE_IN_DIMS = [(8, 8), (64, 64), (8, 1)]
TILE_OUT_DIMS = [(8, 8), (64, 64), (8, 1)]


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
def _bf16_memref(*shape, memory_space=None):
    from aie.ir import IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


# ===========================================================================
# ITERATION 1 -- tiling probe.
# ===========================================================================
def _emit_tiling_probe_seg() -> None:
    """One compute tile (0,2): S2MM (tiled-in) one 64x64 tile, then MM2S
    (un-tiled-out) the same tile.  No compute -- a pure round-trip identity.
    """
    from aie.ir import UnitAttr  # noqa: F401 (kept for parity / future use)

    @device(AIEDevice.npu2, sym_name="decode_attn_seg")
    def _seg():
        shim_tile = tile(0, 0)
        ct = tile(0, 2)

        # Locks: buf empty/full pair.
        lk_in_avail = lock(ct, lock_id=3, init=1)   # L1 free to receive
        lk_in_ready = lock(ct, lock_id=2, init=0)   # L1 holds tiled input
        lk_out_done = lock(ct, lock_id=1, init=1)   # output drained
        lk_out_full = lock(ct, lock_id=0, init=0)   # output ready to drain

        l1_ty = _bf16_memref(TILE_SIZE, memory_space=2)
        buf = buffer(ct, datatype=l1_ty, name="buf_tile")

        external_buffer(_bf16_np(TILE_SIZE), name="__ext_in")
        external_buffer(_bf16_np(TILE_SIZE), name="__ext_out")

        # --- aie.mem block: MM2S0 (out) + S2MM0 (in) ------------------
        @mem(ct)
        def _core_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(lk_out_full, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf, offset=0, len=TILE_SIZE)
                use_lock(lk_out_done, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(lk_in_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf, offset=0, len=TILE_SIZE)
                use_lock(lk_in_ready, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        # --- aie.core: just shuffle the lock tokens (identity) --------
        @core(ct)
        def _core_body():
            for _ in range_(sys.maxsize):
                use_lock(lk_out_done, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_in_ready, LockAction.AcquireGreaterEqual, value=1)
                # No compute: buf already holds the tiled input; just hand it
                # to the output DMA, which un-tiles it on the way out.
                use_lock(lk_in_avail, LockAction.Release, value=1)
                use_lock(lk_out_full, LockAction.Release, value=1)

        # --- Flows ----------------------------------------------------
        # Input: shim DMA:0 (MM2S) -> tile DMA:0 (S2MM), packet routed.
        packetflow(
            pkt_id=0,
            source=shim_tile,
            source_port=WireBundle.DMA,
            source_channel=0,
            dests={"dest": ct, "port": WireBundle.DMA, "channel": 0},
        )
        # Output: tile DMA:0 (MM2S) -> shim DMA:0 (S2MM), circuit switched.
        flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        # --- Shim DMA allocations -------------------------------------
        shim_dma_allocation("air_out", shim_tile, DMAChannelDir.S2MM, 0)
        shim_dma_allocation("air_in", shim_tile, DMAChannelDir.MM2S, 0)

        # --- Runtime sequence -----------------------------------------
        @runtime_sequence(
            _bf16_np(TILE_SIZE),
            _bf16_np(TILE_SIZE),
            sym_name="decode_attn_seg_sequence",
        )
        def _seq(arg_in, arg_out):
            in_task = dma_configure_task_for("air_in")
            with bds(in_task) as bd:
                with bd[0]:
                    dma_bd(arg_in, offset=0, len=TILE_SIZE,
                           dimensions=TILE_IN_DIMS, packet=(0, 0))
                    EndOp()
            dma_start_task(in_task)

            out_task = dma_configure_task_for("air_out", issue_token=True)
            with bds(out_task) as bd:
                with bd[0]:
                    dma_bd(arg_out, offset=0, len=TILE_SIZE,
                           dimensions=TILE_OUT_DIMS)
                    EndOp()
            dma_start_task(out_task)

            dma_await_task(out_task)
            dma_free_task(in_task)


# ===========================================================================
# ITERATIONS 2-4 -- decode attention.
# ===========================================================================
def _declare_attn_kernels():
    """Declare the external_funcs from attn_pythoc.o used by decode attn."""
    from aie.ir import UnitAttr

    g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)        # 4096
    qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)  # (64,64) q/k
    v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)         # (64,64) v
    gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)  # (64,64) ctx
    row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)        # (64,1)

    def _ef(name, inputs):
        fn = external_func(name, inputs=inputs, link_with=KERNEL_OBJECT)
        fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return fn

    return {
        "zero_fill_g": _ef("zero_fill_g_bf16", [g_flat_ty]),
        "zero_fill_gp": _ef("zero_fill_gp_bf16", [gp_ty]),
        "neg_inf_fill_up": _ef("neg_inf_fill_up_bf16", [row_ty]),
        "matmul_a_b": _ef("matmul_a_b_bf16", [qk_ty, qk_ty, g_flat_ty]),
        "matmul_g_b": _ef("matmul_g_b_bf16", [g_flat_ty, v_ty, gp_ty]),
        "fused_softmax": _ef("fused_softmax",
                             [g_flat_ty, row_ty, row_ty, row_ty]),
        "div_gp_sp": _ef("div_gp_sp", [row_ty, gp_ty]),
    }


def _emit_decode_attn_seg() -> None:
    """One compute tile (0,2) running single-KV-tile GQA decode attention.

    DMA: q/k/v tiled in (3 packet flows), gp tiled out (1 circuit flow).
    Compute (single KV tile, no online rescale needed):
        zero_fill_g(g); zero_fill_gp(gp); neg_inf_fill_up(up)
        matmul_a_b(q, k, g)        # g = scores (q . k^T)
        fused_softmax(g, up, sp, r) # g = exp numerators, sp = row sums
        matmul_g_b(g, v, gp)       # gp = numer . v   (gp was zeroed)
        div_gp_sp(sp, gp)          # gp /= row sum  -> normalized context
    """
    @device(AIEDevice.npu2, sym_name="decode_attn_seg")
    def _seg():
        shim_tile = tile(0, 0)
        ct = tile(0, 2)
        kernels = _declare_attn_kernels()

        # Locks: 3 input pairs (q,k,v) + 1 output pair.
        lk_q_avail = lock(ct, lock_id=7, init=1)
        lk_q_ready = lock(ct, lock_id=6, init=0)
        lk_k_avail = lock(ct, lock_id=5, init=1)
        lk_k_ready = lock(ct, lock_id=4, init=0)
        lk_v_avail = lock(ct, lock_id=3, init=1)
        lk_v_ready = lock(ct, lock_id=2, init=0)
        lk_o_done = lock(ct, lock_id=1, init=1)
        lk_o_full = lock(ct, lock_id=0, init=0)

        qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)
        gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        g_ty = _bf16_memref(TILE_ROWS, KVP, memory_space=2)
        row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)

        buf_q = buffer(ct, datatype=qk_ty, name="buf_q")
        buf_k = buffer(ct, datatype=qk_ty, name="buf_k")
        buf_v = buffer(ct, datatype=v_ty, name="buf_v")
        buf_gp = buffer(ct, datatype=gp_ty, name="buf_gp")
        buf_g = buffer(ct, datatype=g_ty, name="buf_g")
        buf_up = buffer(ct, datatype=row_ty, name="buf_up")
        buf_sp = buffer(ct, datatype=row_ty, name="buf_sp")
        buf_r = buffer(ct, datatype=row_ty, name="buf_r")

        external_buffer(_bf16_np(TILE_SIZE), name="__ext_q")
        external_buffer(_bf16_np(TILE_SIZE), name="__ext_k")
        external_buffer(_bf16_np(TILE_SIZE), name="__ext_v")
        external_buffer(_bf16_np(TILE_SIZE), name="__ext_out")

        # --- aie.mem: MM2S0 (gp out) + S2MM0 (q->k->v 3-BD cyclic chain) --
        # A compute tile has only 2 S2MM channels, so all three inputs ride
        # S2MM channel 0, demuxed by packet id (q=0, k=1, v=2).  This mirrors
        # the AWQ builder's 3-input single-channel chain.
        @mem(ct)
        def _core_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(lk_o_full, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_gp, offset=0, len=TILE_SIZE)
                use_lock(lk_o_done, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[6])
            with block[3]:
                use_lock(lk_q_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_q, offset=0, len=TILE_SIZE)
                use_lock(lk_q_ready, LockAction.Release, value=1)
                next_bd(block[4])
            with block[4]:
                use_lock(lk_k_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_k, offset=0, len=TILE_SIZE)
                use_lock(lk_k_ready, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(lk_v_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_v, offset=0, len=TILE_SIZE)
                use_lock(lk_v_ready, LockAction.Release, value=1)
                next_bd(block[3])
            with block[6]:
                EndOp()

        # --- aie.core ------------------------------------------------
        @core(ct)
        def _core_body():
            g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)
            for _ in range_(sys.maxsize):
                use_lock(lk_o_done, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_q_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_k_ready, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_v_ready, LockAction.AcquireGreaterEqual, value=1)

                g_flat = memref.collapse_shape(g_flat_ty, buf_g, [[0, 1]])
                kernels["zero_fill_g"](g_flat)
                kernels["zero_fill_gp"](buf_gp)
                kernels["neg_inf_fill_up"](buf_up)
                kernels["matmul_a_b"](buf_q, buf_k, g_flat)
                kernels["fused_softmax"](g_flat, buf_up, buf_sp, buf_r)
                kernels["matmul_g_b"](g_flat, buf_v, buf_gp)
                kernels["div_gp_sp"](buf_sp, buf_gp)

                use_lock(lk_q_avail, LockAction.Release, value=1)
                use_lock(lk_k_avail, LockAction.Release, value=1)
                use_lock(lk_v_avail, LockAction.Release, value=1)
                use_lock(lk_o_full, LockAction.Release, value=1)

        # --- Flows ----------------------------------------------------
        # 3 inputs multiplexed on shim MM2S 0 -> compute S2MM 0 (pkt 0/1/2).
        for pkt_id in (0, 1, 2):
            packetflow(
                pkt_id=pkt_id,
                source=shim_tile,
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": ct, "port": WireBundle.DMA, "channel": 0},
            )
        flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        # --- Shim DMA allocations -------------------------------------
        shim_dma_allocation("air_out", shim_tile, DMAChannelDir.S2MM, 0)
        shim_dma_allocation("air_q", shim_tile, DMAChannelDir.MM2S, 0)
        shim_dma_allocation("air_k", shim_tile, DMAChannelDir.MM2S, 0)
        shim_dma_allocation("air_v", shim_tile, DMAChannelDir.MM2S, 0)

        # --- Runtime sequence -----------------------------------------
        @runtime_sequence(
            _bf16_np(TILE_SIZE), _bf16_np(TILE_SIZE),
            _bf16_np(TILE_SIZE), _bf16_np(TILE_SIZE),
            sym_name="decode_attn_seg_sequence",
        )
        def _seq(arg_q, arg_k, arg_v, arg_out):
            q_task = dma_configure_task_for("air_q")
            with bds(q_task) as bd:
                with bd[0]:
                    dma_bd(arg_q, offset=0, len=TILE_SIZE,
                           dimensions=TILE_IN_DIMS, packet=(0, 0))
                    EndOp()
            dma_start_task(q_task)

            k_task = dma_configure_task_for("air_k")
            with bds(k_task) as bd:
                with bd[0]:
                    dma_bd(arg_k, offset=0, len=TILE_SIZE,
                           dimensions=TILE_IN_DIMS, packet=(0, 1))
                    EndOp()
            dma_start_task(k_task)

            v_task = dma_configure_task_for("air_v")
            with bds(v_task) as bd:
                with bd[0]:
                    dma_bd(arg_v, offset=0, len=TILE_SIZE,
                           dimensions=TILE_IN_DIMS, packet=(0, 2))
                    EndOp()
            dma_start_task(v_task)

            out_task = dma_configure_task_for("air_out", issue_token=True)
            with bds(out_task) as bd:
                with bd[0]:
                    dma_bd(arg_out, offset=0, len=TILE_SIZE,
                           dimensions=TILE_OUT_DIMS)
                    EndOp()
            dma_start_task(out_task)

            dma_await_task(out_task)
            dma_free_task(q_task)
            dma_free_task(k_task)
            dma_free_task(v_task)


# ---------------------------------------------------------------------------
# ITERATION 4 -- online-softmax KV tiling (seq_len > 64).
# ---------------------------------------------------------------------------
def _declare_attn_kernels_online():
    """Kernels for the online-softmax (multi-chunk) path."""
    from aie.ir import UnitAttr

    g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)
    qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
    v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)
    gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
    row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)

    def _ef(name, inputs):
        fn = external_func(name, inputs=inputs, link_with=KERNEL_OBJECT)
        fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return fn

    return {
        "zero_fill_g": _ef("zero_fill_g_bf16", [g_flat_ty]),
        "zero_fill_gp": _ef("zero_fill_gp_bf16", [gp_ty]),
        "zero_fill_sp": _ef("zero_fill_sp_bf16", [row_ty]),
        "neg_inf_fill_up": _ef("neg_inf_fill_up_bf16", [row_ty]),
        "matmul_a_b": _ef("matmul_a_b_bf16", [qk_ty, qk_ty, g_flat_ty]),
        "matmul_g_b": _ef("matmul_g_b_bf16", [g_flat_ty, v_ty, gp_ty]),
        "fused_softmax": _ef("fused_softmax",
                             [g_flat_ty, row_ty, row_ty, row_ty]),
        "mul_r_gp": _ef("mul_r_gp", [row_ty, gp_ty]),
        "accum_sp_r_s": _ef("accum_sp_r_s", [row_ty, row_ty, row_ty]),
        "vector_copy_32": _ef("vector_copy_32elems",
                              [T.i32(), row_ty, row_ty]),
        "div_gp_sp": _ef("div_gp_sp", [row_ty, gp_ty]),
    }


def _emit_mask_invalid_cols(buf_g, n_valid: int, c0) -> None:
    """Write bf16 -inf (0xff80) into the tiled-G columns [n_valid, 64).

    buf_g is the (64,64) column-block-major 8x8-tiled L1 buffer.  A column
    ``col`` lives at flat offset ``(col//8)*512 + row*8 + (col%8)``.  We mask
    in two parts: whole 8-col blocks beyond the boundary block (each is a
    contiguous 512-elem run = all 64 rows x 8 cols) via v32 stores, and the
    partial boundary block (cols [n_valid, 8*ceil) per row) via scalar
    stores.  All offsets are build-time constants (seq_len is baked).
    """
    if n_valid >= KVP:
        return
    from aie.ir import IntegerType, VectorType
    # bf16 -inf via i16 0xff80 (= -128 as signed i16) bitcast.
    i16_ty = IntegerType.get_signless(16)
    neg_bits = arith.constant(-128, i16_ty)
    v32_i16 = VectorType.get([32], i16_ty)
    v32_bf = VectorType.get([32], T.bf16())
    splat_i16 = vector.broadcast(v32_i16, neg_bits)
    neg_vec = arith.bitcast(v32_bf, splat_i16)

    g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)
    g_flat = memref.collapse_shape(g_flat_ty, buf_g, [[0, 1]])

    boundary_blk = n_valid // 8          # column-block containing the boundary
    rem = n_valid % 8                    # cols still valid inside boundary blk

    from aie.ir import AffineDimExpr, AffineMap
    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])

    # Whole masked column-blocks: each is a contiguous 512-elem run.
    first_full = boundary_blk + (1 if rem else 0)
    for cb in range(first_full, 8):
        base = cb * 512
        for off in range(base, base + 512, 32):
            chunk = memref.subview(g_flat, [off], [32], [1])
            vector.transfer_write(None, neg_vec, chunk, [c0],
                                  permutation_map=perm, in_bounds=[True])

    # Partial boundary block: scalar-mask cols [rem, 8) for every row.
    if rem:
        # Scalar bf16 -inf: lane 0 of the -inf vector.
        from aie.dialects import vector as _vec
        neg_scalar = _vec.extract(neg_vec, [], [0])
        cb = boundary_blk
        for row in range(64):
            for cib in range(rem, 8):
                off = cb * 512 + row * 8 + cib
                idx = arith.constant(off, index=True)
                memref.store(neg_scalar, g_flat, [idx])


def build_decode_attn_module(seq_len: int = 64, *, verbose: bool = False) -> str:
    """Build the decode-attention module for a given context length.

    seq_len == 64  -> single KV tile, no online rescale (stages 0/1).
    seq_len  > 64  -> online-softmax over ceil(seq_len/64) 64-wide KV chunks
                      (stage 2).  The last partial chunk is masked: the host
                      zero-pads K/V and the device masks invalid score columns
                      to bf16 -inf before softmax.
    """
    if verbose:
        print(f"  [attn_decode] building decode-attn module seq_len={seq_len}")
    n_chunks = (seq_len + KVP - 1) // KVP
    # Validated on HW for n_chunks<=4 (seq_len<=256).  Beyond that the
    # current single-shared-channel per-chunk KV DMA either exhausts shim BD
    # IDs (n_chunks>=8) or wedges the packet stream (n_chunks 5-6).  A memtile
    # broadcast/objectfifo-repeat KV feed is the scalable fix (TODO).
    MAX_CHUNKS = 4
    if n_chunks > MAX_CHUNKS:
        raise NotImplementedError(
            f"decode_attn online KV tiling validated for n_chunks<={MAX_CHUNKS}"
            f" (seq_len<={MAX_CHUNKS * KVP}); got seq_len={seq_len} "
            f"(n_chunks={n_chunks}).  Larger contexts need a memtile-staged "
            f"KV feed (see report)."
        )
    kv_size = n_chunks * TILE_SIZE
    host_tys = (_bf16_np(TILE_SIZE), _bf16_np(kv_size),
                _bf16_np(kv_size), _bf16_np(TILE_SIZE))
    with mlir_mod_ctx() as ctx:
        if seq_len == KVP:
            _emit_decode_attn_seg()
        else:
            last_valid = seq_len - (n_chunks - 1) * KVP
            _emit_decode_attn_online_seg(n_chunks, last_valid)
        _emit_dispatcher(host_tys, "decode_attn",
                         "decode_attn_seg", "decode_attn_seg_sequence")
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


def _emit_decode_attn_online_seg(n_chunks: int, last_valid: int) -> None:
    """One compute tile (0,2) running online-softmax GQA decode attention
    over ``n_chunks`` 64-wide KV chunks.  The last chunk has ``last_valid``
    valid columns; the rest are masked to -inf after the QK matmul.

    Online recurrence per chunk (standard flash):
        matmul_a_b(q, k_c, g)             # scores for chunk c
        [mask cols >= last_valid on last chunk]
        fused_softmax(g, up, sp, r)       # up=run max, g=exp numer,
                                          # sp=chunk sum, r=alpha (rescale)
        mul_r_gp(r, gp)                   # gp *= alpha  (rescale old ctx)
        matmul_g_b(g, v_c, gp)            # gp += numer . v
        accum_sp_r_s(sp_run, r, sp)       # sp = alpha*sp_run + chunk_sum
        vector_copy(sp -> sp_run)         # publish new running denom
    After all chunks: div_gp_sp(sp_run, gp).
    """
    @device(AIEDevice.npu2, sym_name="decode_attn_seg")
    def _seg():
        shim_tile = tile(0, 0)
        ct = tile(0, 2)
        kernels = _declare_attn_kernels_online()

        # Locks.
        lk_q_avail = lock(ct, lock_id=9, init=1)
        lk_q_ready = lock(ct, lock_id=8, init=0)
        lk_k_avail = lock(ct, lock_id=7, init=1)
        lk_k_ready = lock(ct, lock_id=6, init=0)
        lk_v_avail = lock(ct, lock_id=5, init=1)
        lk_v_ready = lock(ct, lock_id=4, init=0)
        lk_o_done = lock(ct, lock_id=1, init=1)
        lk_o_full = lock(ct, lock_id=0, init=0)

        qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)
        gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        g_ty = _bf16_memref(TILE_ROWS, KVP, memory_space=2)
        row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)

        buf_q = buffer(ct, datatype=qk_ty, name="buf_q")
        buf_k = buffer(ct, datatype=qk_ty, name="buf_k")
        buf_v = buffer(ct, datatype=v_ty, name="buf_v")
        buf_gp = buffer(ct, datatype=gp_ty, name="buf_gp")
        buf_g = buffer(ct, datatype=g_ty, name="buf_g")
        buf_up = buffer(ct, datatype=row_ty, name="buf_up")
        buf_sp = buffer(ct, datatype=row_ty, name="buf_sp")
        buf_r = buffer(ct, datatype=row_ty, name="buf_r")
        buf_sprun = buffer(ct, datatype=row_ty, name="buf_sprun")

        external_buffer(_bf16_np(TILE_SIZE), name="__ext_q")
        external_buffer(_bf16_np(n_chunks * TILE_SIZE), name="__ext_k")
        external_buffer(_bf16_np(n_chunks * TILE_SIZE), name="__ext_v")
        external_buffer(_bf16_np(TILE_SIZE), name="__ext_out")

        # --- aie.mem ---------------------------------------------------
        # MM2S0  : gp output (1 BD).
        # S2MM0  : q (1 BD) then k chunks (cyclic 1 BD).  q and k share the
        #          channel since they never overlap in time (q once, then
        #          n_chunks k's).
        # S2MM1  : v chunks (cyclic 1 BD).
        # K and V are on SEPARATE channels so each channel carries a single
        # ordered packet stream -- mixing them on one shared channel wedges
        # at n_chunks>=4 (packet interleave breaks the cyclic k,v,k,v chain).
        @mem(ct)
        def _core_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(lk_o_full, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_gp, offset=0, len=TILE_SIZE)
                use_lock(lk_o_done, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[5])
            with block[3]:
                use_lock(lk_q_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_q, offset=0, len=TILE_SIZE)
                use_lock(lk_q_ready, LockAction.Release, value=1)
                next_bd(block[4])
            with block[4]:
                use_lock(lk_k_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_k, offset=0, len=TILE_SIZE)
                use_lock(lk_k_ready, LockAction.Release, value=1)
                next_bd(block[4])   # cycle on k for every chunk
            with block[5]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[7])
            with block[6]:
                use_lock(lk_v_avail, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_v, offset=0, len=TILE_SIZE)
                use_lock(lk_v_ready, LockAction.Release, value=1)
                next_bd(block[6])   # cycle on v for every chunk
            with block[7]:
                EndOp()

        # --- aie.core ------------------------------------------------
        @core(ct)
        def _core_body():
            g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)
            c0 = arith.constant(0, index=True)
            c0_i32 = arith.constant(0, T.i32())
            for _ in range_(sys.maxsize):
                use_lock(lk_o_done, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk_q_ready, LockAction.AcquireGreaterEqual, value=1)

                g_flat = memref.collapse_shape(g_flat_ty, buf_g, [[0, 1]])
                kernels["zero_fill_gp"](buf_gp)
                kernels["zero_fill_sp"](buf_sprun)
                kernels["neg_inf_fill_up"](buf_up)

                for c in range(n_chunks):
                    use_lock(lk_k_ready, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(lk_v_ready, LockAction.AcquireGreaterEqual, value=1)
                    kernels["zero_fill_g"](g_flat)
                    kernels["matmul_a_b"](buf_q, buf_k, g_flat)
                    if c == n_chunks - 1 and last_valid < KVP:
                        _emit_mask_invalid_cols(buf_g, last_valid, c0)
                    kernels["fused_softmax"](g_flat, buf_up, buf_sp, buf_r)
                    kernels["mul_r_gp"](buf_r, buf_gp)
                    kernels["matmul_g_b"](g_flat, buf_v, buf_gp)
                    kernels["accum_sp_r_s"](buf_sprun, buf_r, buf_sp)
                    kernels["vector_copy_32"](c0_i32, buf_sp, buf_sprun)
                    use_lock(lk_k_avail, LockAction.Release, value=1)
                    use_lock(lk_v_avail, LockAction.Release, value=1)

                kernels["div_gp_sp"](buf_sprun, buf_gp)

                use_lock(lk_q_avail, LockAction.Release, value=1)
                use_lock(lk_o_full, LockAction.Release, value=1)

        # --- Flows -----------------------------------------------------
        # q (pkt 0) + k (pkt 1): shim MM2S 0 -> compute S2MM 0.
        # v (pkt 2)           : shim MM2S 1 -> compute S2MM 1.
        for pkt_id in (0, 1):
            packetflow(
                pkt_id=pkt_id,
                source=shim_tile, source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": ct, "port": WireBundle.DMA, "channel": 0},
            )
        packetflow(
            pkt_id=2,
            source=shim_tile, source_port=WireBundle.DMA, source_channel=1,
            dests={"dest": ct, "port": WireBundle.DMA, "channel": 1},
        )
        flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        shim_dma_allocation("air_out", shim_tile, DMAChannelDir.S2MM, 0)
        shim_dma_allocation("air_q", shim_tile, DMAChannelDir.MM2S, 0)
        shim_dma_allocation("air_k", shim_tile, DMAChannelDir.MM2S, 0)
        shim_dma_allocation("air_v", shim_tile, DMAChannelDir.MM2S, 1)

        # --- Runtime sequence: q once, then per-chunk k then v -------
        @runtime_sequence(
            _bf16_np(TILE_SIZE), _bf16_np(n_chunks * TILE_SIZE),
            _bf16_np(n_chunks * TILE_SIZE), _bf16_np(TILE_SIZE),
            sym_name="decode_attn_seg_sequence",
        )
        def _seq(arg_q, arg_k, arg_v, arg_out):
            q_task = dma_configure_task_for("air_q")
            with bds(q_task) as bd:
                with bd[0]:
                    dma_bd(arg_q, offset=0, len=TILE_SIZE,
                           dimensions=TILE_IN_DIMS, packet=(0, 0))
                    EndOp()
            dma_start_task(q_task)

            # Per-chunk K/V tasks (proven for n_chunks<=4).  Each task uses
            # one shim BD ID and is freed only after the output await, so the
            # live-ID count is q + 2*n_chunks + out.  The shim has 16 IDs, so
            # this supports n_chunks<=6 (seq_len up to 384); seq_len>=448 hits
            # the allocator limit (see build_decode_attn_module guard).
            k_tasks = []
            v_tasks = []
            for c in range(n_chunks):
                k_task = dma_configure_task_for("air_k")
                with bds(k_task) as bd:
                    with bd[0]:
                        dma_bd(arg_k, offset=c * TILE_SIZE, len=TILE_SIZE,
                               dimensions=TILE_IN_DIMS, packet=(0, 1))
                        EndOp()
                dma_start_task(k_task)
                k_tasks.append(k_task)

                v_task = dma_configure_task_for("air_v")
                with bds(v_task) as bd:
                    with bd[0]:
                        dma_bd(arg_v, offset=c * TILE_SIZE, len=TILE_SIZE,
                               dimensions=TILE_IN_DIMS, packet=(0, 2))
                        EndOp()
                dma_start_task(v_task)
                v_tasks.append(v_task)

            out_task = dma_configure_task_for("air_out", issue_token=True)
            with bds(out_task) as bd:
                with bd[0]:
                    dma_bd(arg_out, offset=0, len=TILE_SIZE,
                           dimensions=TILE_OUT_DIMS)
                    EndOp()
            dma_start_task(out_task)

            dma_await_task(out_task)
            dma_free_task(q_task)
            for t in k_tasks:
                dma_free_task(t)
            for t in v_tasks:
                dma_free_task(t)


# ===========================================================================
# BATCHED single-dispatch decode attention (additive; production path
# untouched).  N GQA groups on N cores, one per column, all driven by ONE
# runtime_sequence => ONE host dispatch.  Mirrors flash_attn's multi-column
# topology (shim tile(col,0), compute tile(col,2), per-column shim DMA so each
# column's shim has its own BD-ID budget).  Each group runs the EXACT same
# online-softmax logic as _emit_decode_attn_online_seg, copied per-core.
#
# Host BO layout (4 args, independent of n_groups):
#   q_all  : (n_groups * TILE_SIZE,)   group g at offset g*TILE_SIZE
#   k_all  : (n_groups * kv_size,)     group g at offset g*kv_size
#   v_all  : (n_groups * kv_size,)     group g at offset g*kv_size
#   out_all: (n_groups * TILE_SIZE,)   group g at offset g*TILE_SIZE
# ===========================================================================
def _emit_group_mem(ct, n_chunks, buf_gp, buf_q, buf_k, buf_v,
                    lk_o_full, lk_o_done,
                    lk_q_avail, lk_q_ready,
                    lk_k_avail, lk_k_ready,
                    lk_v_avail, lk_v_ready) -> None:
    """Emit one group's aie.mem block (verbatim from the online seg).

    MM2S0: gp out (1 BD).  S2MM0: q then cyclic-k.  S2MM1: cyclic-v.
    """
    @mem(ct)
    def _core_mem(block):
        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(lk_o_full, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buf_gp, offset=0, len=TILE_SIZE)
            use_lock(lk_o_done, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[5])
        with block[3]:
            use_lock(lk_q_avail, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buf_q, offset=0, len=TILE_SIZE)
            use_lock(lk_q_ready, LockAction.Release, value=1)
            next_bd(block[4])
        with block[4]:
            use_lock(lk_k_avail, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buf_k, offset=0, len=TILE_SIZE)
            use_lock(lk_k_ready, LockAction.Release, value=1)
            next_bd(block[4])
        with block[5]:
            dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[7])
        with block[6]:
            use_lock(lk_v_avail, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buf_v, offset=0, len=TILE_SIZE)
            use_lock(lk_v_ready, LockAction.Release, value=1)
            next_bd(block[6])
        with block[7]:
            EndOp()


def _emit_batched_decode_attn_seg(n_groups: int, n_chunks: int,
                                  last_valid: int) -> None:
    """N-group online-softmax decode attention; one core per column."""
    @device(AIEDevice.npu2, sym_name="decode_attn_seg")
    def _seg():
        kernels = _declare_attn_kernels_online()

        qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)
        gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        g_ty = _bf16_memref(TILE_ROWS, KVP, memory_space=2)
        row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)
        g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)

        shim_tiles = []
        for g in range(n_groups):
            col = g  # one group per column
            shim_tile = tile(col, 0)
            ct = tile(col, 2)
            shim_tiles.append(shim_tile)

            lk_q_avail = lock(ct, lock_id=9, init=1)
            lk_q_ready = lock(ct, lock_id=8, init=0)
            lk_k_avail = lock(ct, lock_id=7, init=1)
            lk_k_ready = lock(ct, lock_id=6, init=0)
            lk_v_avail = lock(ct, lock_id=5, init=1)
            lk_v_ready = lock(ct, lock_id=4, init=0)
            lk_o_done = lock(ct, lock_id=1, init=1)
            lk_o_full = lock(ct, lock_id=0, init=0)

            buf_q = buffer(ct, datatype=qk_ty, name=f"buf_q_{g}")
            buf_k = buffer(ct, datatype=qk_ty, name=f"buf_k_{g}")
            buf_v = buffer(ct, datatype=v_ty, name=f"buf_v_{g}")
            buf_gp = buffer(ct, datatype=gp_ty, name=f"buf_gp_{g}")
            buf_g = buffer(ct, datatype=g_ty, name=f"buf_g_{g}")
            buf_up = buffer(ct, datatype=row_ty, name=f"buf_up_{g}")
            buf_sp = buffer(ct, datatype=row_ty, name=f"buf_sp_{g}")
            buf_r = buffer(ct, datatype=row_ty, name=f"buf_r_{g}")
            buf_sprun = buffer(ct, datatype=row_ty, name=f"buf_sprun_{g}")

            _emit_group_mem(
                ct, n_chunks, buf_gp, buf_q, buf_k, buf_v,
                lk_o_full, lk_o_done,
                lk_q_avail, lk_q_ready,
                lk_k_avail, lk_k_ready,
                lk_v_avail, lk_v_ready)

            def _make_core(buf_g=buf_g, buf_gp=buf_gp, buf_sprun=buf_sprun,
                           buf_up=buf_up, buf_q=buf_q, buf_k=buf_k, buf_v=buf_v,
                           buf_sp=buf_sp, buf_r=buf_r,
                           lk_o_done=lk_o_done, lk_o_full=lk_o_full,
                           lk_q_avail=lk_q_avail, lk_q_ready=lk_q_ready,
                           lk_k_avail=lk_k_avail, lk_k_ready=lk_k_ready,
                           lk_v_avail=lk_v_avail, lk_v_ready=lk_v_ready,
                           ct=ct):
                @core(ct)
                def _core_body():
                    c0 = arith.constant(0, index=True)
                    c0_i32 = arith.constant(0, T.i32())
                    for _ in range_(sys.maxsize):
                        use_lock(lk_o_done, LockAction.AcquireGreaterEqual, value=1)
                        use_lock(lk_q_ready, LockAction.AcquireGreaterEqual, value=1)

                        g_flat = memref.collapse_shape(g_flat_ty, buf_g, [[0, 1]])
                        kernels["zero_fill_gp"](buf_gp)
                        kernels["zero_fill_sp"](buf_sprun)
                        kernels["neg_inf_fill_up"](buf_up)

                        for c in range(n_chunks):
                            use_lock(lk_k_ready, LockAction.AcquireGreaterEqual, value=1)
                            use_lock(lk_v_ready, LockAction.AcquireGreaterEqual, value=1)
                            kernels["zero_fill_g"](g_flat)
                            kernels["matmul_a_b"](buf_q, buf_k, g_flat)
                            if c == n_chunks - 1 and last_valid < KVP:
                                _emit_mask_invalid_cols(buf_g, last_valid, c0)
                            kernels["fused_softmax"](g_flat, buf_up, buf_sp, buf_r)
                            kernels["mul_r_gp"](buf_r, buf_gp)
                            kernels["matmul_g_b"](g_flat, buf_v, buf_gp)
                            kernels["accum_sp_r_s"](buf_sprun, buf_r, buf_sp)
                            kernels["vector_copy_32"](c0_i32, buf_sp, buf_sprun)
                            use_lock(lk_k_avail, LockAction.Release, value=1)
                            use_lock(lk_v_avail, LockAction.Release, value=1)

                        kernels["div_gp_sp"](buf_sprun, buf_gp)

                        use_lock(lk_q_avail, LockAction.Release, value=1)
                        use_lock(lk_o_full, LockAction.Release, value=1)
            _make_core()

            # Flows for this column (all column-local routes; packet ids may
            # repeat across columns since each route is on a distinct tile).
            for pkt_id in (0, 1):
                packetflow(
                    pkt_id=pkt_id,
                    source=shim_tile, source_port=WireBundle.DMA, source_channel=0,
                    dests={"dest": ct, "port": WireBundle.DMA, "channel": 0},
                )
            packetflow(
                pkt_id=2,
                source=shim_tile, source_port=WireBundle.DMA, source_channel=1,
                dests={"dest": ct, "port": WireBundle.DMA, "channel": 1},
            )
            flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

            shim_dma_allocation(f"air_out_{g}", shim_tile, DMAChannelDir.S2MM, 0)
            shim_dma_allocation(f"air_q_{g}", shim_tile, DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_k_{g}", shim_tile, DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_v_{g}", shim_tile, DMAChannelDir.MM2S, 1)

        # --- One runtime sequence drives ALL groups (ONE dispatch) --------
        kv_size = n_chunks * TILE_SIZE
        host_q = _bf16_np(n_groups * TILE_SIZE)
        host_kv = _bf16_np(n_groups * kv_size)
        host_out = _bf16_np(n_groups * TILE_SIZE)

        @runtime_sequence(
            host_q, host_kv, host_kv, host_out,
            sym_name="decode_attn_seg_sequence",
        )
        def _seq(arg_q, arg_k, arg_v, arg_out):
            all_tasks = []
            out_tasks = []
            for g in range(n_groups):
                q_off = g * TILE_SIZE
                kv_base = g * kv_size

                q_task = dma_configure_task_for(f"air_q_{g}")
                with bds(q_task) as bd:
                    with bd[0]:
                        dma_bd(arg_q, offset=q_off, len=TILE_SIZE,
                               dimensions=TILE_IN_DIMS, packet=(0, 0))
                        EndOp()
                dma_start_task(q_task)
                all_tasks.append(q_task)

                for c in range(n_chunks):
                    k_task = dma_configure_task_for(f"air_k_{g}")
                    with bds(k_task) as bd:
                        with bd[0]:
                            dma_bd(arg_k, offset=kv_base + c * TILE_SIZE,
                                   len=TILE_SIZE,
                                   dimensions=TILE_IN_DIMS, packet=(0, 1))
                            EndOp()
                    dma_start_task(k_task)
                    all_tasks.append(k_task)

                    v_task = dma_configure_task_for(f"air_v_{g}")
                    with bds(v_task) as bd:
                        with bd[0]:
                            dma_bd(arg_v, offset=kv_base + c * TILE_SIZE,
                                   len=TILE_SIZE,
                                   dimensions=TILE_IN_DIMS, packet=(0, 2))
                            EndOp()
                    dma_start_task(v_task)
                    all_tasks.append(v_task)

                out_task = dma_configure_task_for(f"air_out_{g}", issue_token=True)
                with bds(out_task) as bd:
                    with bd[0]:
                        dma_bd(arg_out, offset=q_off, len=TILE_SIZE,
                               dimensions=TILE_OUT_DIMS)
                        EndOp()
                dma_start_task(out_task)
                out_tasks.append(out_task)

            for t in out_tasks:
                dma_await_task(t)
            for t in all_tasks:
                dma_free_task(t)


def build_decode_attn_batched_module(seq_len: int = 64, n_groups: int = 8,
                                     *, verbose: bool = False) -> str:
    """Batched single-dispatch decode attention: ``n_groups`` GQA groups on
    ``n_groups`` cores (one per column), all in ONE runtime_sequence.

    seq_len scope: n_chunks<=4 (seq_len<=256), same as the single-group path.
    n_groups scope: <=8 (NPU2 has 8 columns).
    """
    if verbose:
        print(f"  [attn_decode] building BATCHED decode-attn module "
              f"seq_len={seq_len} n_groups={n_groups}")
    n_chunks = (seq_len + KVP - 1) // KVP
    MAX_CHUNKS = 4
    if n_chunks > MAX_CHUNKS:
        raise NotImplementedError(
            f"batched decode_attn validated for n_chunks<={MAX_CHUNKS} "
            f"(seq_len<={MAX_CHUNKS * KVP}); got seq_len={seq_len}.")
    if n_groups > 8:
        raise NotImplementedError(f"n_groups<=8 (8 columns); got {n_groups}")

    last_valid = seq_len - (n_chunks - 1) * KVP
    kv_size = n_chunks * TILE_SIZE
    host_tys = (_bf16_np(n_groups * TILE_SIZE), _bf16_np(n_groups * kv_size),
                _bf16_np(n_groups * kv_size), _bf16_np(n_groups * TILE_SIZE))
    with mlir_mod_ctx() as ctx:
        _emit_batched_decode_attn_seg(n_groups, n_chunks, last_valid)
        _emit_dispatcher(host_tys, "decode_attn",
                         "decode_attn_seg", "decode_attn_seg_sequence")
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


def _emit_dispatcher(host_tys, sym_name, seg_sym, seg_seq_sym) -> None:
    """Outer dispatcher device firing the seg sequence via configure+run."""
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(*host_tys, sym_name=sym_name)
        def _outer(*args):
            cfg = ConfigureOp(symbol=seg_sym)
            blk = cfg.body.blocks.append()
            with InsertionPoint(blk):
                RunOp(runtime_sequence_symbol=seg_seq_sym, args=list(args))


def build_tiling_probe_module(*, verbose: bool = False) -> str:
    """ITERATION 1: build the pure tiling round-trip module.

    Host I/O: in (TILE_SIZE,) natural -> out (TILE_SIZE,) natural; expect
    out == in.
    """
    if verbose:
        print("  [attn_decode] building tiling-probe module")
    host_tys = (_bf16_np(TILE_SIZE), _bf16_np(TILE_SIZE))
    with mlir_mod_ctx() as ctx:
        _emit_tiling_probe_seg()
        _emit_dispatcher(host_tys, "decode_attn",
                         "decode_attn_seg", "decode_attn_seg_sequence")
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit a module to stdout (for diffing).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "attn"], default="probe")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    if args.mode == "probe":
        text = build_tiling_probe_module(verbose=True)
    else:
        text = build_decode_attn_module(args.seq_len, verbose=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
