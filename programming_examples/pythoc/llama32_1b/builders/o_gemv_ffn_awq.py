# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b ``o_gemv_ffn_awq`` decode kernel.

This is the Stage-3 AWQ counterpart of ``builders/o_gemv_ffn.py``. The 8
fused-decode launches are identical in topology; the 4 GEMV launches
(og, gg, ug, dg) swap their BF16 weight DMAs for packed-uint4 + groupwise
parameter (AWQ) weight DMAs and call AWQ external kernels:

    og/gg/ug:  awq_matvec_vectorized_u4_bf16 + awq_linalg_fill_bf16
               link_with "awq_mv_pythoc.o"
               weight memref: ui8[M, K/2 + 4*(K/group_size)]   (K=2048, row=1088)
    dg     :  dg_awq_matvec_vectorized_u4_bf16 + dg_awq_linalg_fill_bf16
               link_with "awq_mv_k8192_pythoc.o"
               weight memref: ui8[M, K/2 + 4*(K/group_size)]   (K=8192, row=4352)

The 4 non-GEMV launches (a1 add, rm rms, sw silu_mul, a2 add) are
structurally unchanged from the BF16 sibling. We copy those helpers here
rather than importing them because they hardcode the BF16
``o_gemv_ffn_host_arg_types()`` host signature — and the AWQ dispatcher
needs ``memref<2048x1088xui8>`` / ``memref<8192x1088xui8>`` /
``memref<2048x4352xui8>`` for the four AWQ weight args (0, 7, 9, 12).
See the report from the subagent for details on this copy-vs-reuse
trade-off.

References:
  * ``reference_mlir/o_gemv_ffn_awq.npu.air.mlir`` — ground truth (7,964 lines).
  * ``builders/o_gemv_ffn.py`` — BF16 sibling template.
  * ``kernels/awq_mv.py``, ``kernels/awq_mv_k8192.py`` — external kernel ABI.
  * ``llama32_1b_awq_runtime.py`` lines 204-228 — runtime arg order +
    output_indices / static_input_indices / intermediate_indices.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from ml_dtypes import bfloat16

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
    memtile_dma,
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
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.ir import InsertionPoint, UnitAttr

from ._emit import (
    attach_loop_annotation_to_all_scf_for,
    bf16_memref,
    bf16_np,
)
# c2_attn wave-0 softmax-mask helpers (weight-free, shared verbatim with the
# BF16 c2_attn device so the on-NPU attention mask cannot drift between paths).
from .o_gemv_ffn import (
    _c2attn_mask_invalid_cols,
    _c2attn_mask_invalid_cols_rtp,
)


# ---------------------------------------------------------------------------
# Constants matching the cached AWQ AIR-stitched IR for Llama-3.2-1B.
# ---------------------------------------------------------------------------
EMB_DIM = 2048      # model hidden size
HIDDEN_DIM = 8192   # FFN hidden size
GROUP_SIZE = 128    # AWQ group size baked into kernels/awq_mv.py
N_COLS = 8          # 8 compute columns in the matvec herd
K_TILE = 8          # inner K tiling factor for the K=2048 AWQ matvec
M_TILE = 8          # rows processed per K=2048 matvec call
# K_TILE = M_TILE => K-loop is a single iter. See rms_gemv_rope_awq.py
# for the rationale; mirrors the K_TILE_K8192=2 change on dg AWQ.

# A/B toggle: double-buffer the L1 weight buffer on the K=2048 matvec
# (og/gg/ug) and overlap the next row-group's L1 fill with this group's
# compute by unrolling the outer row-group loop by 2.  See
# _emit_awq_matvec_seg_k2048 and PINGPONG_STATUS.md.  Off = current
# single-buffered baseline.
PINGPONG_W_K2048 = False

# A/B toggle: same outer-loop L1 W double-buffer on the K=8192 down-proj
# (dg).  Independent of the K=2048 flag so dg can be isolated.  dg has the
# largest matvec span + highest lock_stall in the AWQ trace, so it's the
# strongest W-prefetch candidate.  Env-overridable for A/B sweeps
# (PYTHOC_PP_W_DG=0/1); literal is the default when the env is unset.
import os as _os
PINGPONG_W_DG = False # _os.environ.get("PYTHOC_PP_W_DG", "1") == "1"

# A/B toggle: L2 (memtile) W double-buffer on dg -- lets shim->L2 overlap
# L2->L1, attacking the W-stream starvation at the L2 hop (the trace's
# dominant W-starvation source).  Independent of PINGPONG_W_DG (L1 hop);
# both can be on for a full DDR->L2->L1 depth-2 W pipeline.  Plumbed into
# both the unpacked dg seg and the d4 packed emitter.  Env-overridable
# (PYTHOC_PP_W_L2_DG=0/1).
PINGPONG_W_L2_DG = True #_os.environ.get("PYTHOC_PP_W_L2_DG", "1") == "1"

# Down-projection (K=8192) tiling.
K_TILE_K8192 = 2    # inner K factor for the K=8192 matvec
M_TILE_K8192 = 2    # rows processed per K=8192 matvec call
# K_TILE_K8192 = M_TILE_K8192 => K-loop is a single iter (no looping).
# This removes the per-K-iter lock cycle, at the cost of 2x the per-call
# work (matvec_fn processes 2 output rows instead of 1) and 2x the W L1
# tile size (8.5 KB instead of 4.25 KB). L1 still has plenty of headroom.

# Inline-add per-tile chunk size (256 bf16 elements).
ADD_CHUNK = 256

# SwiGLU per-tile buffer size.
SWIGLU_CHUNK = 1024

# Per-segment kernel object filenames.
KO_AWQ_MV = "awq_mv_pythoc.ll"  # inlined (alwaysinline IR-merge): ~12% faster AWQ decode
KO_AWQ_MV_K8192 = "awq_mv_k8192_pythoc.ll"  # inlined
KO_SWIGLU = "silu_and_mul_bf16.o"
KO_RMS = "rms_norm_2048_bf16.o"
KO_MATVEC_RMS = "matvec_rms_pythoc.ll"  # inlined (alwaysinline IR-merge)  # fused packed-RMS prologue (air 3-device fold)


def _combined_row_bytes(k: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes per AWQ row: K/2 packed uint4 + 4*(K/group_size) param bytes."""
    return k // 2 + 4 * (k // group_size)


def _ui8_memref(*shape, memory_space=None):
    """Module-level ui8 memref helper (AWQ packed-uint4 weight buffers)."""
    from aie.extras import types as T
    from aie.ir import IntegerAttr, IntegerType, MemRefType
    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.ui8(), None, ms)


# ---------------------------------------------------------------------------
# Channel-number map (verbatim from cached
# reference_mlir/o_gemv_ffn_awq.npu.air.mlir shim_dma_allocations).
# Note: AWQ-tree channel numbers are different from BF16-tree numbers.
# ---------------------------------------------------------------------------
_CHANNELS: Dict[str, Dict[str, object]] = {
    # Phase 1: OG AWQ GEMV (out_rows=2048, awq_mv_pythoc.o)
    "og_awq_matvec_0":   {"weight_base": 33, "out_base": 32, "input": 1},
    # Phase 2: Residual add 1 (inline 256-elt herd)
    "a1_eltwise_add_seg": {"in0": 5, "in1": 6, "out": 7},
    # Phase 3: FFN RMSNorm (single tile)
    "rm_rms_seg":         {"in0": 8, "in1": 9, "out": 10},
    # Phase 4: GG AWQ GEMV (out_rows=8192, awq_mv_pythoc.o)
    "gg_awq_matvec_0":   {"weight_base": 36, "out_base": 35, "input": 12},
    # Phase 5: UG AWQ GEMV (out_rows=8192, awq_mv_pythoc.o)
    "ug_awq_matvec_0":   {"weight_base": 39, "out_base": 34, "input": 17},
    # Phase 6: SwiGLU (8 tiles, 1024-elt buffers)
    "sw_silu_mul_seg":    {"in0": 21, "in1": 22, "out": 23},
    # Phase 7: DG AWQ GEMV (out_rows=2048, K=8192, awq_mv_k8192_pythoc.o)
    "dg_awq_matvec_0":   {"weight_base": 38, "out_base": 37, "input": 25},
    # Phase 8: Residual add 2 (inline 256-elt herd)
    "a2_eltwise_add_seg": {"in0": 29, "in1": 30, "out": 31},
}


# ---------------------------------------------------------------------------
# 15-arg host signature for the AWQ dispatcher and every segment.
# ARG ORDER (matches llama32_1b_awq_runtime.py:204-220):
#   0  : memref<emb_dim x row2048 x ui8>     wo_w (AWQ packed)
#   1  : memref<emb_dim x bf16>              attn_out
#   2  : memref<emb_dim x bf16>              proj_buf
#   3  : memref<emb_dim x bf16>              x_residual
#   4  : memref<emb_dim x bf16>              res1_buf
#   5  : memref<emb_dim x bf16>              ffn_norm
#   6  : memref<emb_dim x bf16>              normed2_buf
#   7  : memref<hidden_dim x row2048 x ui8>  wgate_w (AWQ packed)
#   8  : memref<hidden_dim x bf16>           gate_buf
#   9  : memref<hidden_dim x row2048 x ui8>  wup_w (AWQ packed)
#  10  : memref<hidden_dim x bf16>           up_buf
#  11  : memref<hidden_dim x bf16>           swiglu_buf
#  12  : memref<emb_dim x row8192 x ui8>     wdown_w (AWQ packed, K=8192)
#  13  : memref<emb_dim x bf16>              down_buf
#  14  : memref<emb_dim x bf16>              output_buf
# ---------------------------------------------------------------------------
def _awq_host_arg_types(emb_dim: int = EMB_DIM,
                        hidden_dim: int = HIDDEN_DIM,
                        group_size: int = GROUP_SIZE) -> List:
    row2048 = _combined_row_bytes(emb_dim, group_size)        # 1088
    row8192 = _combined_row_bytes(hidden_dim, group_size)     # 4352
    u8 = np.uint8
    return [
        np.ndarray[(emb_dim, row2048), np.dtype[u8]],       #  0 wo_w
        bf16_np(emb_dim),                                    #  1 attn_out
        bf16_np(emb_dim),                                    #  2 proj_buf
        bf16_np(emb_dim),                                    #  3 x_residual
        bf16_np(emb_dim),                                    #  4 res1_buf
        bf16_np(emb_dim),                                    #  5 ffn_norm
        bf16_np(emb_dim),                                    #  6 normed2_buf
        np.ndarray[(hidden_dim, row2048), np.dtype[u8]],    #  7 wgate_w
        bf16_np(hidden_dim),                                 #  8 gate_buf
        np.ndarray[(hidden_dim, row2048), np.dtype[u8]],    #  9 wup_w
        bf16_np(hidden_dim),                                 # 10 up_buf
        bf16_np(hidden_dim),                                 # 11 swiglu_buf
        np.ndarray[(emb_dim, row8192), np.dtype[u8]],       # 12 wdown_w
        bf16_np(emb_dim),                                    # 13 down_buf
        bf16_np(emb_dim),                                    # 14 output_buf
    ]


def _awq_attn_n_chunks(resident: bool) -> int:
    """KV chunk ceiling for the AWQ c2_attn device (matches the BF16 path).

    Default 4 (seq<=256); MEMKV lifts it to PYTHOC_C2_ATTN_MAX_CHUNKS so the
    host BO sizing / runtime-L fold stay in lockstep with the device geometry.
    Non-resident single-chunk is unused for AWQ (always resident).
    """
    import os as _os_nc
    if not resident:
        return 1
    if _os_nc.environ.get("PYTHOC_C2_ATTN_MEMKV", "0") == "1":
        return int(_os_nc.environ.get("PYTHOC_C2_ATTN_MAX_CHUNKS", "8"))
    return 4


def _awq_c2_attn_host_arg_types(emb_dim: int = EMB_DIM,
                                hidden_dim: int = HIDDEN_DIM,
                                group_size: int = GROUP_SIZE,
                                *, n_groups: int = N_COLS,
                                resident: bool = False) -> List:
    """Extended ABI for the AWQ ``c2_attn`` device (18 args).

    The AWQ counterpart of ``builders/_emit.py::c2_attn_host_arg_types``: the
    base AWQ 15-arg layout (uint8 packed weights at 0/7/9/12) but arg1
    (attn_out) is WIDENED to the per-group attention-output scratch
    (n_groups*4096), and three bf16 attention inputs are appended:

        arg15 : q_all  (n_groups*4096)
        arg16 : k_all  (n_groups*n_chunks*4096)
        arg17 : v_all  (n_groups*n_chunks*4096)

    Resident folds the runtime valid-length L into q's padding (no extra arg).
    """
    tile_size = 64 * 64  # A_TILE_ROWS * A_HEAD_DIM
    n_chunks = _awq_attn_n_chunks(resident)
    kv_size = n_chunks * tile_size
    base = _awq_host_arg_types(emb_dim, hidden_dim, group_size)
    base[1] = bf16_np(n_groups * tile_size)   # widen attn_out scratch
    return base + [
        bf16_np(n_groups * tile_size),        # arg15 q_all
        bf16_np(n_groups * kv_size),          # arg16 k_all
        bf16_np(n_groups * kv_size),          # arg17 v_all
    ]


# ---------------------------------------------------------------------------
# external_buffer triples emitted per device. AIR uses these as opaque
# metadata; aiecc treats them as references. We mirror the order/shapes
# the cached MLIR uses so the structural diff stays minimal.
# ``shapes`` is a list of (shape_tuple, dtype) pairs. dtype is "bf16" or "ui8".
# ---------------------------------------------------------------------------
def _emit_external_buffers(*shapes_with_dtype):
    names = ["__air_external_buffer", "__air_external_buffer_1",
             "__air_external_buffer_2"]
    for nm, (shp, dt) in zip(names, shapes_with_dtype):
        if dt == "bf16":
            ty = bf16_np(*shp)
        elif dt == "ui8":
            ty = np.ndarray[shp, np.dtype[np.uint8]]
        else:
            raise ValueError(f"unknown dtype {dt}")
        external_buffer(ty, name=nm)


def _emit_external_buffers_bf16(*shapes):
    """Backward-compatible helper for all-bf16 external buffer triples."""
    _emit_external_buffers(*[(s, "bf16") for s in shapes])


# ---------------------------------------------------------------------------
# AWQ GEMV matvec segment (K=2048, awq_mv_pythoc.o). Shared by og, gg, ug.
# Structurally identical to BF16 matvec_seg_k2048 except weight buffer
# memrefs are packed uint8 with row width = combined_row_bytes(K=2048) = 1088.
# ---------------------------------------------------------------------------
def _emit_awq_matvec_seg_k2048(sym: str, weight_arg_idx: int, input_arg_idx: int,
                                output_arg_idx: int, out_rows: int,
                                group_size: int = GROUP_SIZE,
                                pingpong_w: bool = False) -> None:
    """Emit an AWQ [8,1] matvec herd device with K=2048.

    ``out_rows``  -- 2048 (O proj) or 8192 (gate/up projections).
    n_outer = out_rows // 1024. Each outer iteration delivers 1024 rows
    across the 8 columns (128 per column).

    ``pingpong_w=True`` double-buffers the L1 weight buffer so the
    memtile->L1 fill of row-group N+1 overlaps the compute of row-group N.
    At K_TILE=M_TILE=8 the inner output-row sub-loop is a single iter, so
    the overlap is recovered by unrolling the *outer* row-group loop by 2
    (group A reads ``_wb``, group B reads ``_wb1``); ``w_avail`` becomes
    init=2 and the L1-receive DMA becomes a 2-BD ring.  (X/Y stay
    single-buffered.)  This differs from the inner-sub-loop unroll used by
    ``pingpong_x`` / the stale rms ``pingpong_w``, which assume 2 sub-loop
    trips and are inert at K_TILE=8.
    """
    chans = _CHANNELS[sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024
    if pingpong_w:
        # Outer-loop unroll-by-2 must divide the row-groups/col/outer evenly
        # so the core never blocks on a phantom (odd) group.
        rows_per_col = 1024 // N_COLS
        assert rows_per_col % (2 * M_TILE) == 0, (
            f"pingpong_w outer unroll-by-2 needs rows_per_col "
            f"({rows_per_col}) divisible by 2*M_TILE ({2 * M_TILE})"
        )

    row_bytes = _combined_row_bytes(EMB_DIM, group_size)    # 1088
    # Output DMA shape per col: 128 elts arranged as (16,64),(8,1).
    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    x_repeat_count = 31
    # Weight DMA shape per col task: 16 mini-rows of 16 chunks of 544 bytes
    # = 139264 bytes = 8 rows * 1088 bytes * 16 mem-tile cycles.
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE * row_bytes              # 8 * 1088 = 8704
    weight_outer_stride = 1024 * row_bytes              # 1024 * 1088 = 1_114_112
    output_col_stride = M_TILE                          # 8
    output_outer_stride = 1024

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Mem tile locks (4 ids 3..0, AIR descending col order).
        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        # Compute tile locks (6 ids 5..0, ascending col).
        core_locks = {}
        _w_avail_init = 2 if pingpong_w else 1
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=_w_avail_init),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffer types.
        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.extras import types as T
        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(
                    IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE, row_bytes, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)

        # Mem tile buffers (descending col order to match AIR emit).
        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        core_buf_y = {}
        core_buf_w = {}
        core_buf_w1 = {}  # only when pingpong_w
        core_buf_x = {}
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)
            if pingpong_w:
                core_buf_w1[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)

        # External buffers: weight (M, row_bytes) ui8, input (K,) bf16,
        # output (M,) bf16.  Order matches cached MLIR.
        _emit_external_buffers(
            ((out_rows, row_bytes), "ui8"),
            ((EMB_DIM,), "bf16"),
            ((out_rows,), "bf16"),
        )

        # Declare external_funcs.
        from ml_dtypes import bfloat16 as _bf16
        fill_fn = external_func(
            "awq_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # Compute tile mem + core blocks (descending col).
        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            x_buf = core_buf_x[col]
            w_buf1 = core_buf_w1.get(col)  # None unless pingpong_w

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_xb, offset=0, len=EMB_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # W L1 ping-pong: 2-BD ring filling wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_core_mem(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb, _wb1):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())

                    def _group(_w):
                        # One row-group: zero y, matvec M_TILE rows, ship y.
                        # K_TILE=M_TILE so the inner sub-loop is a single iter
                        # (row_offset=0); the matvec consumes the whole _w tile.
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE, K_TILE):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _w, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)

                    if _wb1 is None:
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                    else:
                        # W L1 ping-pong: unroll the outer row-group loop by 2,
                        # alternating wb0/wb1 so the L1 fill of the next group
                        # (into the other buffer) overlaps this group's compute.
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                            _group(_wb1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

        # Flows (shim<->mem; shim->compute(input); mem<->compute).
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)

        # Mem tile DMAs (ascending col).
        def _make_memtile_dma(_col, _ml, _w, _y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                with block[4]:
                    use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["w_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
        for col in range(N_COLS):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col], mem_buf_y[col])

        # Shim DMA allocations.
        out_base = chans["out_base"]
        weight_base = chans["weight_base"]
        input_chan = chans["input"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{weight_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        shim_dma_allocation(
            f"air_channel_{input_chan}",
            shim_tiles[0],
            DMAChannelDir.MM2S,
            1,
        )

        # Runtime sequence.
        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_y = args[output_arg_idx]
            for outer in range(n_outer):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{weight_base}_{col}",
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len,
                                dimensions=w_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(
                    f"air_channel_{input_chan}",
                    repeat_count=x_repeat_count,
                )
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_x,
                            offset=0,
                            len=EMB_DIM,
                            dimensions=[(4, 512), (512, 1)],
                        )
                        EndOp()
                dma_start_task(x_task)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_base}_{col}",
                        issue_token=True,
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_y,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len,
                                dimensions=y_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


# ---------------------------------------------------------------------------
# AWQ GEMV matvec segment (K=8192, awq_mv_k8192_pythoc.o) -- Down projection.
# Compute tile: M_TILE_K8192=2 output rows per call, K_TILE_K8192=1 inner k.
# Weight L1 buffer: 1 row of row_bytes=4352 ui8.
# Output rows = EMB_DIM = 2048 across 8 outer iters
#   (each outer covers 256 rows = 8 cols * 32 elts).
# ---------------------------------------------------------------------------
def _emit_awq_matvec_seg_k8192(sym: str, weight_arg_idx: int,
                                input_arg_idx: int, output_arg_idx: int,
                                group_size: int = GROUP_SIZE,
                                pingpong_x: bool = False,
                                pingpong_w_l2: bool = False,
                                pingpong_w: bool = False) -> None:
    """K=8192 down-projection AWQ matvec [8,1] herd, awq_mv_k8192_pythoc.o.

    Output rows: 2048 across 8 outer iters (each outer covers 256 rows =
    8 cols * 32 elts each).  Weight has same access pattern as K=2048
    case in elements (w_dims=[(16,69632),(16,544),(544,1)], len=139264),
    but the down-projection's weight memref has row_bytes=4352 and
    outer stride 256 rows * 4352 = 1_114_112.

    ``pingpong_x=True`` doubles the L1 X buffer (~16 KB each: AWQ dg's X
    is HIDDEN_DIM=8192 bf16 == 16 KB), turns the X DMA BD chain into a
    2-BD ring (xb0/xb1 alternating), and raises ``x_avail`` to init=2.
    The K-loop (M_TILE_K8192/K_TILE_K8192 = 2 iters) is unrolled so iter
    0 reads xb0 and iter 1 reads xb1. L1 footprint after enable:
    W 4.25 KB + 2*X 32 KB + Y 4 B ~= 36 KB, fits in the 64 KB cap.

    Rationale: the kernel-local trace shows X (per BD) is 4x bigger than
    W (16 KB vs 4.25 KB) on the AWQ K=8192 path -- X is the dominant
    DMA cost. W ping-pong on AWQ moved starv0 (X) the wrong direction
    because the two channels share upstream bandwidth; tackling X
    directly is the higher-leverage move.

    ``pingpong_w_l2=True`` doubles the L2 memtile W buffer (8.5 KB each
    for AWQ K=8192), splits both memtile chains (S2MM ch 0 shim->L2 and
    MM2S ch 1 L2->L1) into 2-BD rings, and raises ``w_dma_done`` to
    init=2. Same pattern as the BF16 dg L2 W PP (commit bb8ddd4ab).
    Independent of pingpong_x; both can be enabled together.
    """
    chans = _CHANNELS[sym]
    out_rows = EMB_DIM
    n_outer = out_rows // 256  # 8
    if pingpong_w:
        # Outer-loop unroll-by-2 must divide the row-groups/col/outer evenly.
        rows_per_col = 256 // N_COLS                  # 32
        assert rows_per_col % (2 * M_TILE_K8192) == 0, (
            f"pingpong_w outer unroll-by-2 needs rows_per_col "
            f"({rows_per_col}) divisible by 2*M_TILE_K8192 ({2 * M_TILE_K8192})"
        )
        assert not pingpong_x, (
            "pingpong_w (outer-loop W ring, single X) and pingpong_x "
            "(inner K-loop X ring) are mutually exclusive on dg"
        )

    row_bytes = _combined_row_bytes(HIDDEN_DIM, group_size)    # 4352
    y_dims = [(16, 16), (2, 1)]
    y_len = 32
    x_repeat_count = 31
    x_dims = [(16, 512), (512, 1)]
    x_len = HIDDEN_DIM
    # Per-col weight task: 16 mini-rows of 16 chunks of 544 bytes = 139264.
    # That matches the K=2048 K-block size in bytes (1088/4) -- structurally
    # the mem tile cycles 16 times per outer iter for each of 16 sub-chunks.
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE_K8192 * row_bytes              # 2*4352 = 8704
    weight_outer_stride = 256 * row_bytes                     # 256*4352 = 1_114_112
    output_col_stride = M_TILE_K8192                          # 2
    output_outer_stride = 256

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        mem_locks = {}
        _w_dma_done_init = 2 if pingpong_w_l2 else 1
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=_w_dma_done_init),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        core_locks = {}
        _x_avail_init = 2 if pingpong_x else 1
        _w_avail_init = 2 if pingpong_w else 1
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=_w_avail_init),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=_x_avail_init),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffer types (K=8192 variant).
        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.extras import types as T
        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(
                    IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE_K8192, row_bytes, memory_space=2)
        _X_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE_K8192, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE_K8192, memory_space=1)

        mem_buf_w = {}
        mem_buf_w1 = {}  # only when pingpong_w_l2
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
            if pingpong_w_l2:
                mem_buf_w1[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        core_buf_y = {}
        core_buf_w = {}
        core_buf_w1 = {}  # only when pingpong_w
        core_buf_x = {}
        core_buf_x1 = {}  # only when pingpong_x
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)
            if pingpong_w:
                core_buf_w1[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            if pingpong_x:
                core_buf_x1[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)

        # External buffers: weight (EMB_DIM, row_bytes) ui8, input
        # (HIDDEN_DIM,) bf16, output (EMB_DIM,) bf16.
        _emit_external_buffers(
            ((EMB_DIM, row_bytes), "ui8"),
            ((HIDDEN_DIM,), "bf16"),
            ((EMB_DIM,), "bf16"),
        )

        from ml_dtypes import bfloat16 as _bf16
        fill_fn = external_func(
            "dg_awq_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_AWQ_MV_K8192,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "dg_awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_AWQ_MV_K8192,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            w_buf1 = core_buf_w1.get(col)  # None unless pingpong_w
            x_buf = core_buf_x[col]
            x_buf1 = core_buf_x1.get(col)  # None unless pingpong_x

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb, _xb1, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE_K8192)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    if _xb1 is None:
                        with block[4]:
                            use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_xb, offset=0, len=HIDDEN_DIM)
                            use_lock(_cl["x_ready"], LockAction.Release, value=1)
                            next_bd(block[4])
                    else:
                        # X ping-pong: 2-BD ring writing xb0 then xb1.
                        with block[4]:
                            use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_xb, offset=0, len=HIDDEN_DIM)
                            use_lock(_cl["x_ready"], LockAction.Release, value=1)
                            next_bd(block[9])
                        with block[9]:
                            use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_xb1, offset=0, len=HIDDEN_DIM)
                            use_lock(_cl["x_ready"], LockAction.Release, value=1)
                            next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # W L1 ping-pong: 2-BD ring filling wb0 then wb1.
                        # (block[7]/[8] are free; X ping-pong uses block[9].)
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_core_mem(ct_op, cl, y_buf, w_buf, x_buf, x_buf1, w_buf1)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb, _xb1, _wb1):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(HIDDEN_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE_K8192, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())

                    if _wb1 is not None:
                        # W L1 ping-pong: unroll the outer row-group loop by 2,
                        # alternating wb0/wb1. X stays single (both groups read
                        # _xb); requires pingpong_x off (asserted at emit).
                        def _group(_w):
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                                k_i32 = index_cast(k_idx, to=T.i32())
                                use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, k_i32, _w, _xb, _yb)
                                use_lock(_cl["x_avail"], LockAction.Release, value=1)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                            _group(_wb1)
                        return

                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        if _xb1 is None:
                            for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                                k_i32 = index_cast(k_idx, to=T.i32())
                                use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                                use_lock(_cl["x_avail"], LockAction.Release, value=1)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        else:
                            # X ping-pong: unroll 2 K-iters, K-iter 0 reads
                            # xb0, K-iter 1 reads xb1. M_TILE_K8192/K_TILE_K8192
                            # must be 2.
                            assert M_TILE_K8192 // K_TILE_K8192 == 2, (
                                f"X pingpong unroll assumes M_TILE_K8192/"
                                f"K_TILE_K8192==2, got "
                                f"{M_TILE_K8192}/{K_TILE_K8192}"
                            )
                            k_i32_0 = arith.constant(0, T.i32())
                            k_i32_1 = arith.constant(K_TILE_K8192, T.i32())
                            # K-iter 0: xb0
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            # K-iter 1: xb1
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_1, _wb, _xb1, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf, x_buf1, w_buf1)

        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)

        def _make_memtile_dma(_col, _ml, _w, _w1, _y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE_K8192)
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                if _w1 is None:
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                else:
                    # L2 W ping-pong: 2-BD rings on both MM2S ch 1
                    # (L2->L1) and S2MM ch 0 (shim->L2).
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[9])
                    with block[9]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[10])
                    with block[10]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE_K8192)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
        for col in range(N_COLS):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col],
                              mem_buf_w1.get(col), mem_buf_y[col])

        out_base = chans["out_base"]
        weight_base = chans["weight_base"]
        input_chan = chans["input"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{weight_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        shim_dma_allocation(
            f"air_channel_{input_chan}",
            shim_tiles[0],
            DMAChannelDir.MM2S,
            1,
        )

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_y = args[output_arg_idx]
            for outer in range(n_outer):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{weight_base}_{col}",
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len,
                                dimensions=w_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(
                    f"air_channel_{input_chan}",
                    repeat_count=x_repeat_count,
                )
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_x,
                            offset=0,
                            len=x_len,
                            dimensions=x_dims,
                        )
                        EndOp()
                dma_start_task(x_task)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_base}_{col}",
                        issue_token=True,
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_y,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len,
                                dimensions=y_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


# ---------------------------------------------------------------------------
# Eltwise-add segment (inline arith.addf, no link_with).
# Structurally identical to the BF16 sibling's helper -- the dispatcher's
# host arg types just need to be the AWQ 15-arg signature.
# ---------------------------------------------------------------------------
def _emit_eltwise_add_seg(sym: str, in0_arg_idx: int, in1_arg_idx: int,
                          out_arg_idx: int, group_size: int = GROUP_SIZE) -> None:
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "in2_avail": lock(ct, lock_id=5, init=1),
                "in2_ready": lock(ct, lock_id=4, init=0),
                "in1_avail": lock(ct, lock_id=3, init=1),
                "in1_ready": lock(ct, lock_id=2, init=0),
                "out_done":  lock(ct, lock_id=1, init=1),
                "out_full":  lock(ct, lock_id=0, init=0),
            }

        _BUF_TY = bf16_memref(ADD_CHUNK, memory_space=2)

        core_buf_out = {}
        core_buf_in2 = {}
        core_buf_in1 = {}
        for col in reversed(range(N_COLS)):
            core_buf_out[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in2[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in1[col] = buffer(compute_tiles[col], datatype=_BUF_TY)

        _emit_external_buffers_bf16((EMB_DIM,), (EMB_DIM,), (EMB_DIM,))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineMap, AffineDimExpr

        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            buf_out = core_buf_out[col]
            buf_in2 = core_buf_in2[col]
            buf_in1 = core_buf_in1[col]

            def _make_core_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=ADD_CHUNK)
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b1, offset=0, len=ADD_CHUNK)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b2, offset=0, len=ADD_CHUNK)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(ct_op, cl, buf_out, buf_in2, buf_in1)

            def _make_core_body(_ct, _cl, _bo, _b2, _b1):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    c0 = arith.constant(0, T.index())
                    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                    vec_ty = T.vector(16, T.bf16())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        for i in range_(0, ADD_CHUNK, 16):
                            sub1 = memref.subview(_b1, [i], [16], [1])
                            sub2 = memref.subview(_b2, [i], [16], [1])
                            subo = memref.subview(_bo, [i], [16], [1])
                            v1 = vector.transfer_read(
                                vec_ty, sub1, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            v2 = vector.transfer_read(
                                vec_ty, sub2, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            vsum = arith.addf(v1, v2)
                            vector.transfer_write(
                                None, vsum, subo, [c0],
                                permutation_map=perm, in_bounds=[True])
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, buf_out, buf_in2, buf_in1)

        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        out_chan = chans["out"]
        in0_chan = chans["in0"]
        in1_chan = chans["in1"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{in0_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{in1_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                1,
            )

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_in1 = args[in0_arg_idx]
            arg_in2 = args[in1_arg_idx]
            arg_out = args[out_arg_idx]
            in1_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(f"air_channel_{in0_chan}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_in1,
                            offset=col * ADD_CHUNK,
                            len=ADD_CHUNK,
                            dimensions=[(ADD_CHUNK, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                in1_tasks.append(t)
            in2_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(f"air_channel_{in1_chan}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_in2,
                            offset=col * ADD_CHUNK,
                            len=ADD_CHUNK,
                            dimensions=[(ADD_CHUNK, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                in2_tasks.append(t)
            out_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(
                    f"air_channel_{out_chan}_{col}",
                    issue_token=True,
                )
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_out,
                            offset=col * ADD_CHUNK,
                            len=ADD_CHUNK,
                            dimensions=[(ADD_CHUNK, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                out_tasks.append(t)

            for t in reversed(out_tasks):
                dma_await_task(t)
            for t in reversed(in2_tasks):
                dma_free_task(t)
            for t in reversed(in1_tasks):
                dma_free_task(t)


# ---------------------------------------------------------------------------
# RMSNorm segment (single compute tile, external rms_norm kernel).
# Identical to BF16 sibling apart from the AWQ 15-arg dispatcher signature.
# ---------------------------------------------------------------------------
def _emit_rm_rms_seg(group_size: int = GROUP_SIZE) -> None:
    sym = "rm_rms_seg"
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim = tile(0, 0)
        ct = tile(0, 2)

        lk5 = lock(ct, lock_id=5, init=1)
        lk4 = lock(ct, lock_id=4, init=0)
        lk3 = lock(ct, lock_id=3, init=1)
        lk2 = lock(ct, lock_id=2, init=0)
        lk1 = lock(ct, lock_id=1, init=1)
        lk0 = lock(ct, lock_id=0, init=0)

        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        _BF16_16_L1 = bf16_memref(16, memory_space=2)
        buf_x = buffer(ct, datatype=_BF16_2048_L1)
        buf_y = buffer(ct, datatype=_BF16_2048_L1)
        buf_w = buffer(ct, datatype=_BF16_2048_L1)
        buf_s = buffer(ct, datatype=_BF16_16_L1)

        _emit_external_buffers_bf16((EMB_DIM,), (EMB_DIM,), (EMB_DIM,))

        @mem(ct)
        def _core_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(lk0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_y, offset=0, len=EMB_DIM)
                use_lock(lk1, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                EndOp()
            with block[3]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
            with block[4]:
                use_lock(lk3, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_w, offset=0, len=EMB_DIM)
                use_lock(lk2, LockAction.Release, value=1)
                next_bd(block[4])
            with block[5]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
            with block[6]:
                use_lock(lk5, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_x, offset=0, len=EMB_DIM)
                use_lock(lk4, LockAction.Release, value=1)
                next_bd(block[6])

        from aie.extras import types as T

        rms_fn = external_func(
            "rms_norm_2048_bf16",
            inputs=[_BF16_2048_L1, _BF16_2048_L1, _BF16_2048_L1, _BF16_16_L1],
            link_with=KO_RMS,
        )
        rms_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        import sys as _sys

        @core(ct)
        def _core_body():
            for _ in range_(_sys.maxsize):
                use_lock(lk1, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk2, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk4, LockAction.AcquireGreaterEqual, value=1)
                rms_fn(buf_x, buf_w, buf_y, buf_s)
                use_lock(lk5, LockAction.Release, value=1)
                use_lock(lk0, LockAction.Release, value=1)
                use_lock(lk3, LockAction.Release, value=1)

        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[5]
            arg_x = args[4]
            arg_y = args[6]
            t_w = dma_configure_task_for(f"air_channel_{chans['in0']}")
            with bds(t_w) as bd:
                with bd[0]:
                    dma_bd(arg_w, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_w)
            t_x = dma_configure_task_for(f"air_channel_{chans['in1']}")
            with bds(t_x) as bd:
                with bd[0]:
                    dma_bd(arg_x, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_x)
            t_y = dma_configure_task_for(f"air_channel_{chans['out']}", issue_token=True)
            with bds(t_y) as bd:
                with bd[0]:
                    dma_bd(arg_y, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_y)
            dma_await_task(t_y)
            dma_free_task(t_w)
            dma_free_task(t_x)


# ---------------------------------------------------------------------------
# SwiGLU segment.  Same as BF16 sibling apart from dispatcher arg types.
# ---------------------------------------------------------------------------
def _emit_sw_silu_mul_seg(group_size: int = GROUP_SIZE) -> None:
    sym = "sw_silu_mul_seg"
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "in2_avail": lock(ct, lock_id=5, init=1),
                "in2_ready": lock(ct, lock_id=4, init=0),
                "in1_avail": lock(ct, lock_id=3, init=1),
                "in1_ready": lock(ct, lock_id=2, init=0),
                "out_done":  lock(ct, lock_id=1, init=1),
                "out_full":  lock(ct, lock_id=0, init=0),
            }

        _BUF_TY = bf16_memref(SWIGLU_CHUNK, memory_space=2)

        core_buf_out = {}
        core_buf_in2 = {}
        core_buf_in1 = {}
        for col in reversed(range(N_COLS)):
            core_buf_out[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in2[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in1[col] = buffer(compute_tiles[col], datatype=_BUF_TY)

        _emit_external_buffers_bf16((HIDDEN_DIM,), (HIDDEN_DIM,), (HIDDEN_DIM,))

        from aie.extras import types as T
        silu_fn = external_func(
            "silu_and_mul_bf16",
            inputs=[_BUF_TY, _BUF_TY, _BUF_TY, np.int32],
            link_with=KO_SWIGLU,
        )
        silu_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            buf_out = core_buf_out[col]
            buf_in2 = core_buf_in2[col]
            buf_in1 = core_buf_in1[col]

            def _make_core_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=SWIGLU_CHUNK)
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b1, offset=0, len=SWIGLU_CHUNK)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b2, offset=0, len=SWIGLU_CHUNK)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(ct_op, cl, buf_out, buf_in2, buf_in1)

            def _make_core_body(_ct, _cl, _bo, _b2, _b1):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    n_c = arith.constant(SWIGLU_CHUNK, T.i32())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        silu_fn(_b1, _b2, _bo, n_c)
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, buf_out, buf_in2, buf_in1)

        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        out_chan = chans["out"]
        in0_chan = chans["in0"]
        in1_chan = chans["in1"]
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{out_chan}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{in0_chan}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{in1_chan}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 1)

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_gate = args[8]
            arg_up = args[10]
            arg_out = args[11]
            in1_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(f"air_channel_{in0_chan}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_gate,
                            offset=col * SWIGLU_CHUNK,
                            len=SWIGLU_CHUNK,
                            dimensions=[(2, 512), (512, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                in1_tasks.append(t)
            in2_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(f"air_channel_{in1_chan}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_up,
                            offset=col * SWIGLU_CHUNK,
                            len=SWIGLU_CHUNK,
                            dimensions=[(2, 512), (512, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                in2_tasks.append(t)
            out_tasks = []
            for col in range(N_COLS):
                t = dma_configure_task_for(
                    f"air_channel_{out_chan}_{col}",
                    issue_token=True,
                )
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_out,
                            offset=col * SWIGLU_CHUNK,
                            len=SWIGLU_CHUNK,
                            dimensions=[(2, 512), (512, 1)],
                        )
                        EndOp()
                dma_start_task(t)
                out_tasks.append(t)

            for t in reversed(out_tasks):
                dma_await_task(t)
            for t in reversed(in2_tasks):
                dma_free_task(t)
            for t in reversed(in1_tasks):
                dma_free_task(t)


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
_DEFAULT_AWQ_DISPATCH_SEQUENCE = (
    "og_awq_matvec_0", "a1_eltwise_add_seg",
    "rm_rms_seg", "gg_awq_matvec_0",
    "ug_awq_matvec_0", "sw_silu_mul_seg",
    "dg_awq_matvec_0", "a2_eltwise_add_seg",
)


# ---------------------------------------------------------------------------
# C2 merged device (AWQ): full call-2 collapse, ported from the BF16 builder.
# ---------------------------------------------------------------------------
def _emit_awq_call2_c2(sym: str, with_down: bool,
                       group_size: int = GROUP_SIZE, *,
                       attn_wave0: bool = False, seq_len: int = 64,
                       n_groups: int = 8,
                       attn_resident: bool = False) -> None:
    """C2 (collapse plan): the C1 merged device, evolved per the C2 row map.

    ``attn_wave0`` (c2_attn): prepend GQA decode attention as WAVE 0 on the
    row-3 (add) herd -- the AWQ counterpart of ``o_gemv_ffn.py::_emit_call2_c2``
    attn_wave0.  The attention compute is WEIGHT-FREE (BFP576 BF16
    ``attn_pythoc.o`` kernels), so the wave-0 block is identical to the BF16
    device regardless of the uint4 surrounding matvecs.  The 8 add cores are
    idle until add1 (waits for the O wave), so attention reuses them with the
    SAME channel map (MM2S0 out, S2MM0/S2MM1 in).  Attention writes per-group
    context (64x64 untiled) to a wide DDR scratch (arg1, widened to
    n_groups*4096); the O wave gathers rows 0..3 of each group head-major into
    ``normed`` so O/add1/gate/up/swiglu/down/add2 stay byte-identical.  ONE
    device / ONE configure / 1 LoadPDI.  All attention code is gated behind
    ``attn_wave0`` so the default c2_merged emission is untouched.

    vs C1: the standalone rms tile/stage is gone -- gate/up waves run the
    proven d1d3d4_rms fold (packed [res1|norm_w] delivered once per wave,
    `rms_norm_packed_bf16` into a resident `normed`, then 128 matvec chunks).
    The O wave activation is also delivered ONCE per token into `normed`
    (resident reuse, no per-chunk x stream). add herd row 3 runs TWO waves
    (add1, add2). swiglu on row 4. ``with_down`` adds the K=8192 down herd
    on row 5 (D4's core/mem copied, x resident-once) and the mem tiles carry
    a second W chain (MM2S2/S2MM2) -- call 2 = ONE configure.

    Stages: O / add1 / gate / up / swiglu [/ down / add2].
    Packet IDs are DISTINCT SINGLE BITS so no two roles can alias under any
    subset mask the pathfinder picks: matvec W/x = 1, add = 2, swiglu = 4,
    down = 8; ALL outputs converge to the shim on id 1 (no demux needed).
    (Earlier {8,12,13} aliased: on shared shim ports the router emitted
    rule(mask=27, val=8) which drops bit 2, merging add=8 and swiglu=12, so
    col-0's add input -- the only column also carrying the X broadcast --
    starved. Single-bit ids force the mask to include each role's bit.)
    """
    row_bytes = _combined_row_bytes(EMB_DIM, group_size)       # 1088 (K=2048)
    row_bytes8192 = _combined_row_bytes(HIDDEN_DIM, group_size) # 4352 (K=8192)
    W_CH, A0_CH, SG_CH = 60, 61, 62                     # MM2S 0 demux
    X_CH, A1_CH, SU_CH = 64, 65, 66                     # MM2S 1 demux
    YO_CH, AO_CH, SO_CH = 68, 69, 70                    # S2MM 0 mux
    DW_CH, DX_CH, DO_CH = 72, 73, 74                    # down (with_down)
    # c2_attn wave-0 attention channels (row 3 / add tiles), disjoint from the
    # AWQ c2 channels (60-74).  q+k ride the add in1 channel (shim MM2S0), v
    # rides add in2 (shim MM2S1), gp out rides add out (shim S2MM0).  The waves
    # are time-disjoint so they share the physical channels.  Mirrors the BF16
    # o_gemv_ffn.py wave-0 channel map verbatim.
    AQ_CH, AK_CH, AV_CH, APO_CH = 90, 91, 92, 93
    AL_CH = 94                                          # resident L (MM2S0)

    # Attention geometry (mirrors builders/o_gemv_ffn.py::_emit_call2_c2).
    A_HEAD_DIM = 64
    A_GROUP_SIZE = 4
    A_TILE_ROWS = 64
    A_KVP = 64
    A_TILE_SIZE = A_TILE_ROWS * A_HEAD_DIM            # 4096
    import os as _os_geo
    # MEMKV: lifts the 256-token cap by feeding the FULL per-group K (and V) in
    # ONE shim BD each over the proven shim->add routing; the add-tile fill ring
    # backpressures the single stream into the L1 double-buffer, so shim BD-task
    # usage is CONSTANT (q+k+v+out = 4/group) and context length decouples from
    # the shim BD budget.  Default OFF (4-chunk path).
    _A_MEMKV = (attn_resident
                and _os_geo.environ.get("PYTHOC_C2_ATTN_MEMKV", "0") == "1")
    A_MAX_CHUNKS = 4                                  # seq_len <= 256
    if _A_MEMKV:
        A_MAX_CHUNKS = int(_os_geo.environ.get("PYTHOC_C2_ATTN_MAX_CHUNKS", "8"))
    if attn_resident:
        A_N_CHUNKS = A_MAX_CHUNKS
        A_LAST_VALID = A_KVP                            # unused (runtime mask)
        A_CHUNKS_PER_BUF = 2                            # double-buffer 2 chunks
        A_N_BUF_FILLS = A_N_CHUNKS // A_CHUNKS_PER_BUF  # refills/token
    else:
        A_N_CHUNKS = (seq_len + A_KVP - 1) // A_KVP
        A_LAST_VALID = seq_len - (A_N_CHUNKS - 1) * A_KVP
    A_KV_SIZE = A_N_CHUNKS * A_TILE_SIZE
    if attn_wave0 and not attn_resident and A_N_CHUNKS != 1:
        raise NotImplementedError(
            f"AWQ c2_attn wave-0 attention is wired for seq_len<=64 "
            f"(n_chunks=1) in the non-resident path; got seq_len={seq_len} "
            f"(n_chunks={A_N_CHUNKS}).  Use the resident path.")
    A_TILE_IN_DIMS = [(8, 8), (64, 64), (8, 1)]
    A_TILE_OUT_DIMS = [(8, 8), (64, 64), (8, 1)]

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE * row_bytes
    weight_outer_stride = 1024 * row_bytes
    output_col_stride = M_TILE
    output_outer_stride = 1024
    # down (K=8192) geometry, verbatim from _emit_matvec_add_pack_k8192
    d_n_outer = EMB_DIM // 256
    d_y_dims = [(16, 16), (2, 1)]
    d_y_len = 32
    d_w_col_stride = M_TILE_K8192 * row_bytes8192
    d_w_outer_stride = 256 * row_bytes8192

    # Debug knob: plain gate/up waves (normed2 from DDR, no on-core RMS).
    import os as _os
    _pg = int(_os.environ.get("PYTHOC_C2_PLAINGATE", "0"))
    _plain_gate = _pg == 1          # plain normed2 BD, no rms
    _skip_rms = _pg == 2            # packed BD delivered, rms call skipped
    _alt_rms = _pg == 3             # packed BD; call KO_RMS kernel instead

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        mat_tiles = [tile(c, 2) for c in range(N_COLS)]
        add_tiles = [tile(c, 3) for c in range(N_COLS)]
        sw_tiles = [tile(c, 4) for c in range(N_COLS)]
        dn_tiles = [tile(c, 5) for c in range(N_COLS)] if with_down else None
        import os as _os
        _xcol = int(_os.environ.get("PYTHOC_C2_XCOL", "0"))  # X-broadcast src col
        # FIX: deliver the mat activation (X) per-column via each column's own
        # mem-tile (shim[c] -> mem[c] -> mat[c]) instead of a shim-row broadcast
        # fan from shim[0]. The fan shared the MM2S1 lane with per-column add1
        # in1 and starved the fan's terminal columns (see test_c2_add_starve).
        # Per-column delivery has no E/W fan, so MM2S1 traffic is all local.
        # Mem X ring uses odd channel 5 (BD pool 24-47, clear of the W/y even
        # chains). Disabled for with_down (c2_merged) -- its mem channels 2/3
        # are taken by the down W/y chains; that path keeps the old broadcast.
        _memx = (not with_down) and _os.environ.get("PYTHOC_C2_MEMX", "1") == "1"

        # PYTHOC_C2_WRELAY2: 2-slot L2 W-relay ping-pong (the warm-reuse fix).
        # The AWQ c2 W-relay is SINGLE-BUFFERED (w_dma_done/w_ready init=1/0) --
        # the exact 1-credit relay the BF16 path had BEFORE the WRELAY2 fix.  It
        # works for c2_merged (which reloads the PDI between dispatches) but the
        # resident c2_attn path reuses ONE PDI, so the relay's parked prefetch
        # becomes observable as cross-token warm-reuse drift.  Auto-ON for the
        # resident path; default OFF for plain c2_merged (production IR stays
        # byte-identical).  Explicit "1"/"0" overrides.  CRITICAL: credit MUST
        # be 1, not 2 (init=2 races two slots ahead -> nondeterministic gate/up
        # residual amplified by down).  Mirrors o_gemv_ffn.py's WRELAY2.
        _wrelay2_env = _os.environ.get("PYTHOC_C2_WRELAY2")
        if _wrelay2_env is not None:
            _wrelay2 = _wrelay2_env == "1"
        else:
            _wrelay2 = attn_resident
        # Independently gate the DOWN (K=8192) relay ping-pong for isolation.
        _wrelay2_dn = _wrelay2 and _os.environ.get("PYTHOC_C2_WRELAY2_DN", "1") == "1"

        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3,
                                   init=int(_os.environ.get("PYTHOC_C2_WRELAY2_CR", "1"))
                                   if _wrelay2 else 1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }
            if with_down:
                mem_locks[col].update({
                    "dw_dma_done": lock(mt, lock_id=7,
                                        init=int(_os.environ.get("PYTHOC_C2_WRELAY2_CR", "1"))
                                        if _wrelay2_dn else 1),
                    "dw_ready":    lock(mt, lock_id=6, init=0),
                    "dy_done":     lock(mt, lock_id=5, init=1),
                    "dy_ready":    lock(mt, lock_id=4, init=0),
                })
            if _memx:
                # X relay ring (ids 8/9 clear of the w/y ids 0-3).
                mem_locks[col].update({
                    "x_empty": lock(mt, lock_id=9, init=1),
                    "x_full":  lock(mt, lock_id=8, init=0),
                })

        def _six_locks(t):
            return {
                "w_avail": lock(t, lock_id=5, init=1),
                "w_ready": lock(t, lock_id=4, init=0),
                "x_avail": lock(t, lock_id=3, init=1),
                "x_ready": lock(t, lock_id=2, init=0),
                "y_done":  lock(t, lock_id=1, init=1),
                "y_full":  lock(t, lock_id=0, init=0),
            }

        def _io_locks(t):
            return {
                "in2_avail": lock(t, lock_id=5, init=1),
                "in2_ready": lock(t, lock_id=4, init=0),
                "in1_avail": lock(t, lock_id=3, init=1),
                "in1_ready": lock(t, lock_id=2, init=0),
                "out_done":  lock(t, lock_id=1, init=1),
                "out_full":  lock(t, lock_id=0, init=0),
            }

        mat_locks = {c: _six_locks(mat_tiles[c]) for c in range(N_COLS)}
        add_locks = {c: _io_locks(add_tiles[c]) for c in range(N_COLS)}
        sw_locks = {c: _io_locks(sw_tiles[c]) for c in range(N_COLS)}
        dn_locks = ({c: _six_locks(dn_tiles[c]) for c in range(N_COLS)}
                    if with_down else None)

        _W_L1_TY = _ui8_memref(K_TILE, row_bytes, memory_space=2)
        _XP_L1_TY = bf16_memref(2 * EMB_DIM, memory_space=2)   # [res1|norm_w]
        _NORMED_TY = bf16_memref(EMB_DIM, memory_space=2)
        _RSCR_TY = bf16_memref(16, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)
        _ADD_TY = bf16_memref(ADD_CHUNK, memory_space=2)
        _SW_TY = bf16_memref(SWIGLU_CHUNK, memory_space=2)
        _DW_L1_TY = _ui8_memref(K_TILE_K8192, row_bytes8192, memory_space=2)
        _DX_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _DY_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _DW_L2_TY = _ui8_memref(1, M_TILE_K8192, row_bytes8192, memory_space=1)
        _DY_L2_TY = bf16_memref(1, M_TILE_K8192, memory_space=1)

        mem_buf_w = {}
        mem_buf_y = {}
        mem_buf_dw = {}
        mem_buf_dy = {}
        mem_buf_x = {}
        _MX_L2_TY = bf16_memref(2 * EMB_DIM, memory_space=1)   # holds packed X
        if _memx:
            for col in reversed(range(N_COLS)):
                mem_buf_x[col] = buffer(mem_tiles[col], datatype=_MX_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        mem_buf_w1 = {}
        if _wrelay2:
            for col in reversed(range(N_COLS)):
                mem_buf_w1[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)
        mem_buf_dw1 = {}
        if with_down:
            for col in reversed(range(N_COLS)):
                mem_buf_dw[col] = buffer(mem_tiles[col], datatype=_DW_L2_TY)
            if _wrelay2_dn:
                for col in reversed(range(N_COLS)):
                    mem_buf_dw1[col] = buffer(mem_tiles[col], datatype=_DW_L2_TY)
            for col in reversed(range(N_COLS)):
                mem_buf_dy[col] = buffer(mem_tiles[col], datatype=_DY_L2_TY)

        mat_buf_y = {}
        mat_buf_w = {}
        mat_buf_xp = {}
        mat_buf_normed = {}
        mat_buf_rscr = {}
        add_buf_out = {}
        add_buf_in2 = {}
        add_buf_in1 = {}
        sw_buf_out = {}
        sw_buf_in2 = {}
        sw_buf_in1 = {}
        dn_buf_y = {}
        dn_buf_w = {}
        dn_buf_x = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_xp[col] = buffer(mat_tiles[col], datatype=_XP_L1_TY)
            mat_buf_normed[col] = buffer(mat_tiles[col], datatype=_NORMED_TY)
            mat_buf_rscr[col] = buffer(mat_tiles[col], datatype=_RSCR_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_in2[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_in1[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            sw_buf_out[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_in2[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_in1[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            if with_down:
                dn_buf_y[col] = buffer(dn_tiles[col], datatype=_DY_L1_TY)
                dn_buf_w[col] = buffer(dn_tiles[col], datatype=_DW_L1_TY)
                dn_buf_x[col] = buffer(dn_tiles[col], datatype=_DX_L1_TY)

        _emit_external_buffers(((HIDDEN_DIM, row_bytes), "ui8"), ((EMB_DIM,), "bf16"), ((HIDDEN_DIM,), "bf16"))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        fill_fn = external_func(
            "awq_linalg_fill_bf16", inputs=[_bf16, _Y_L1_TY], link_with=KO_AWQ_MV)
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _NORMED_TY, _Y_L1_TY],
            link_with=KO_AWQ_MV)
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if _alt_rms:
            rms_alt_fn = external_func(
                "rms_norm_2048_bf16",
                inputs=[_NORMED_TY, _NORMED_TY, _NORMED_TY, _RSCR_TY],
                link_with=KO_RMS)
            rms_alt_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        rms_fn = external_func(
            "rms_norm_packed_bf16",
            inputs=[_XP_L1_TY, _NORMED_TY, _RSCR_TY],
            link_with=KO_MATVEC_RMS)
        rms_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        silu_fn = external_func(
            "silu_and_mul_bf16",
            inputs=[_SW_TY, _SW_TY, _SW_TY, np.int32],
            link_with=KO_SWIGLU)
        silu_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if with_down:
            dn_fill_fn = external_func(
                "dg_awq_linalg_fill_bf16", inputs=[_bf16, _DY_L1_TY],
                link_with=KO_AWQ_MV_K8192)
            dn_fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            dn_matvec_fn = external_func(
                "dg_awq_matvec_vectorized_u4_bf16",
                inputs=[np.int32, np.int32, np.int32, _DW_L1_TY, _DX_L1_TY,
                        _DY_L1_TY],
                link_with=KO_AWQ_MV_K8192)
            dn_matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # --- c2_attn: attention kernels + per-column buffers/locks on the
        #     row-3 (add) tiles. All gated behind attn_wave0.  Transplanted
        #     verbatim from o_gemv_ffn.py::_emit_call2_c2 (weight-free BFP576
        #     BF16 attn_pythoc.o kernels -- identical for the uint4 device). ---
        attn_kernels = {}
        a_buf = {}          # per-col dict of attention L1 buffers
        a_lock = {}         # per-col dict of attention locks
        a_rtp = {}          # per-col runtime valid-length L RTP (resident mode)
        if attn_wave0:
            KO_ATTN = "attn_pythoc.o"
            _A_QK_TY = bf16_memref(A_TILE_ROWS, A_HEAD_DIM, memory_space=2)
            _A_V_TY = bf16_memref(A_KVP, A_HEAD_DIM, memory_space=2)
            if attn_resident:
                from aie.ir import (MemRefType as _MRT, IntegerType as _IT,
                                    IntegerAttr as _IA)
                _i8 = _IT.get_signless(8)
                _ms2 = _IA.get(_IT.get_signless(32), 2)
                _A_KALL_TY = _MRT.get([A_CHUNKS_PER_BUF * A_TILE_SIZE * 2],
                                      _i8, None, _ms2)
            _A_GP_TY = bf16_memref(A_TILE_ROWS, A_HEAD_DIM, memory_space=2)
            _A_G_TY = bf16_memref(A_TILE_ROWS, A_KVP, memory_space=2)
            _A_ROW_TY = bf16_memref(A_TILE_ROWS, 1, memory_space=2)
            _A_GFLAT_TY = bf16_memref(A_TILE_SIZE, memory_space=2)

            def _aef(name, inputs):
                fn = external_func(name, inputs=inputs, link_with=KO_ATTN)
                fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
                return fn

            attn_kernels = {
                "zero_fill_g": _aef("zero_fill_g_bf16", [_A_GFLAT_TY]),
                "zero_fill_gp": _aef("zero_fill_gp_bf16", [_A_GP_TY]),
                "zero_fill_sp": _aef("zero_fill_sp_bf16", [_A_ROW_TY]),
                "neg_inf_fill_up": _aef("neg_inf_fill_up_bf16", [_A_ROW_TY]),
                "matmul_a_b": _aef("matmul_a_b_bf16",
                                   [_A_QK_TY, _A_QK_TY, _A_GFLAT_TY]),
                "matmul_g_b": _aef("matmul_g_b_bf16",
                                   [_A_GFLAT_TY, _A_V_TY, _A_GP_TY]),
                "fused_softmax": _aef("fused_softmax",
                                      [_A_GFLAT_TY, _A_ROW_TY, _A_ROW_TY,
                                       _A_ROW_TY]),
                "mul_r_gp": _aef("mul_r_gp", [_A_ROW_TY, _A_GP_TY]),
                "accum_sp_r_s": _aef("accum_sp_r_s",
                                     [_A_ROW_TY, _A_ROW_TY, _A_ROW_TY]),
                "vector_copy_32": _aef("vector_copy_32elems",
                                       [np.int32, _A_ROW_TY, _A_ROW_TY]),
                "div_gp_sp": _aef("div_gp_sp", [_A_ROW_TY, _A_GP_TY]),
            }
            for col in range(n_groups):
                at = add_tiles[col]
                _k_ty = _A_KALL_TY if attn_resident else _A_QK_TY
                _v_ty = _A_KALL_TY if attn_resident else _A_V_TY
                a_buf[col] = {
                    "q": buffer(at, datatype=_A_QK_TY, name=f"a_q_{col}"),
                    "k": buffer(at, datatype=_k_ty, name=f"a_k_{col}"),
                    "v": buffer(at, datatype=_v_ty, name=f"a_v_{col}"),
                    "gp": buffer(at, datatype=_A_GP_TY, name=f"a_gp_{col}"),
                    "g": buffer(at, datatype=_A_G_TY, name=f"a_g_{col}"),
                    "up": buffer(at, datatype=_A_ROW_TY, name=f"a_up_{col}"),
                    "sp": buffer(at, datatype=_A_ROW_TY, name=f"a_sp_{col}"),
                    "r": buffer(at, datatype=_A_ROW_TY, name=f"a_r_{col}"),
                    "sprun": buffer(at, datatype=_A_ROW_TY,
                                    name=f"a_sprun_{col}"),
                }
                # Attention locks (ids 6-13 clear of add's 0-5).
                a_lock[col] = {
                    "q_avail": lock(at, lock_id=13, init=1),
                    "q_ready": lock(at, lock_id=12, init=0),
                    "k_avail": lock(at, lock_id=11, init=1),
                    "k_ready": lock(at, lock_id=10, init=0),
                    "v_avail": lock(at, lock_id=9, init=1),
                    "v_ready": lock(at, lock_id=8, init=0),
                    "o_done":  lock(at, lock_id=7, init=1),
                    "o_full":  lock(at, lock_id=6, init=0),
                }
                if attn_resident:
                    a_rtp[col] = "q_padding"

        # --- matvec row 2: x BDs ring O(normed) -> gate(xp) -> up(xp) ---
        N_CHUNKS_O = EMB_DIM // N_COLS // M_TILE       # 32
        N_CHUNKS_GU = HIDDEN_DIM // N_COLS // M_TILE   # 128
        for col in reversed(range(N_COLS)):
            def _make_mat_mem(_ct, _cl, _yb, _wb, _xpb, _nb):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[7])
                    with block[4]:
                        # O wave: attn_out -> normed (matvec reads it in place)
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_nb, offset=0, len=EMB_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[5])
                    with block[5]:
                        # gate wave: packed [res1|norm_w] (or plain normed2)
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        if _plain_gate:
                            dma_bd(_nb, offset=0, len=EMB_DIM)
                        else:
                            dma_bd(_xpb, offset=0, len=2 * EMB_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                    with block[6]:
                        # up wave: packed [res1|norm_w] again (or plain)
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        if _plain_gate:
                            dma_bd(_nb, offset=0, len=EMB_DIM)
                        else:
                            dma_bd(_xpb, offset=0, len=2 * EMB_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[7]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                    with block[8]:
                        use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[8])
            _make_mat_mem(mat_tiles[col], mat_locks[col], mat_buf_y[col],
                          mat_buf_w[col], mat_buf_xp[col], mat_buf_normed[col])

            def _make_mat_core(_ct, _cl, _yb, _wb, _xpb, _nb, _scr):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_off = arith.constant(0, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    for _ in range_(_sys.maxsize):
                        # O wave: activation resident in normed for 32 chunks
                        use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        for _c in range_(N_CHUNKS_O):
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, zero_off, _wb, _nb, _yb)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
                        use_lock(_cl["x_avail"], LockAction.Release, value=1)
                        # gate, up waves: rms once, then 128 chunks each.
                        # Unrolled straight-line (not for _w in range_(2)):
                        # keeps the inlined rms at the same loop depth as the
                        # proven d3 fold (deeper nesting miscompiles).
                        for _ in range(2):
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if _alt_rms:
                                rms_alt_fn(_nb, _nb, _nb, _scr)
                            elif not _plain_gate and not _skip_rms:
                                rms_fn(_xpb, _nb, _scr)
                            for _c in range_(N_CHUNKS_GU):
                                use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                                fill_fn(zero_bf16, _yb)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, zero_off, _wb, _nb, _yb)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                                use_lock(_cl["y_full"], LockAction.Release, value=1)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
            _make_mat_core(mat_tiles[col], mat_locks[col], mat_buf_y[col],
                           mat_buf_w[col], mat_buf_xp[col], mat_buf_normed[col],
                           mat_buf_rscr[col])

        # --- add row 3 (C1 add herd; add1 then add2 waves).  Under attn_wave0
        #     the SAME tiles also run GQA decode attention as wave 0: the add
        #     mem block gains attention BDs as the first slot on each channel
        #     (gp out on MM2S0, q+k on S2MM0, v on S2MM1) and the add core runs
        #     attention once then TWO adds per token (add1, add2).  Transplanted
        #     verbatim from o_gemv_ffn.py::_emit_call2_c2. ---
        for col in reversed(range(N_COLS)):
            _al = a_lock.get(col)
            _ab = a_buf.get(col)

            def _make_add_mem(_ct, _cl, _bo, _b2, _b1, _al=_al, _ab=_ab,
                              _arL=a_rtp.get(col)):
                @mem(_ct)
                def _core_mem(block):
                    if _al is not None:
                        # MM2S0 ring: gp -> add_out -> add_out -> gp.
                        dma_start(DMAChannelDir.MM2S, 0, dest=block[1],
                                  chain=block[10])
                        with block[1]:
                            use_lock(_al["o_full"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_ab["gp"], offset=0, len=A_TILE_SIZE,
                                   packet=(0, 16))
                            use_lock(_al["o_done"], LockAction.Release, value=1)
                            next_bd(block[2])
                        with block[2]:
                            use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_bo, offset=0, len=ADD_CHUNK, packet=(0, 5))
                            use_lock(_cl["out_done"], LockAction.Release, value=1)
                            next_bd(block[4])
                        with block[4]:
                            use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_bo, offset=0, len=ADD_CHUNK, packet=(0, 5))
                            use_lock(_cl["out_done"], LockAction.Release, value=1)
                            next_bd(block[1])
                        with block[3]:
                            EndOp()
                        # S2MM0 ring: q -> [k-fill]xN -> add_in1 -> add_in1 -> q.
                        with block[10]:
                            dma_start(DMAChannelDir.S2MM, 0, dest=block[11],
                                      chain=block[20])
                        _fb = A_N_BUF_FILLS if _arL is not None else 1
                        _kvlen = (A_CHUNKS_PER_BUF * A_TILE_SIZE * 2
                                  if _arL is not None else A_TILE_SIZE)
                        with block[11]:
                            use_lock(_al["q_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_ab["q"], offset=0, len=A_TILE_SIZE)
                            use_lock(_al["q_ready"], LockAction.Release, value=1)
                            next_bd(block[12])
                        for _kf in range(_fb):
                            _nxt = (block[12 + _kf + 1] if _kf < _fb - 1
                                    else block[12 + _fb])      # -> add_in1
                            with block[12 + _kf]:
                                use_lock(_al["k_avail"], LockAction.AcquireGreaterEqual, value=1)
                                dma_bd(_ab["k"], offset=0, len=_kvlen)
                                use_lock(_al["k_ready"], LockAction.Release, value=1)
                                next_bd(_nxt)
                        _a1 = 12 + _fb
                        with block[_a1]:
                            use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b1, offset=0, len=ADD_CHUNK)
                            use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                            next_bd(block[_a1 + 1])
                        with block[_a1 + 1]:
                            use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b1, offset=0, len=ADD_CHUNK)
                            use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                            next_bd(block[11])
                        # S2MM1 ring: [v-fill]xN -> add_in2 -> add_in2 -> v.
                        with block[20]:
                            dma_start(DMAChannelDir.S2MM, 1, dest=block[21],
                                      chain=block[3])
                        for _vf in range(_fb):
                            _nxt = (block[21 + _vf + 1] if _vf < _fb - 1
                                    else block[21 + _fb])      # -> add_in2
                            with block[21 + _vf]:
                                use_lock(_al["v_avail"], LockAction.AcquireGreaterEqual, value=1)
                                dma_bd(_ab["v"], offset=0, len=_kvlen)
                                use_lock(_al["v_ready"], LockAction.Release, value=1)
                                next_bd(_nxt)
                        _a2 = 21 + _fb
                        with block[_a2]:
                            use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b2, offset=0, len=ADD_CHUNK)
                            use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                            next_bd(block[_a2 + 1])
                        with block[_a2 + 1]:
                            use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_b2, offset=0, len=ADD_CHUNK)
                            use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                            next_bd(block[21])
                        return
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=ADD_CHUNK, packet=(0, 5))
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b1, offset=0, len=ADD_CHUNK)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b2, offset=0, len=ADD_CHUNK)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_add_mem(add_tiles[col], add_locks[col], add_buf_out[col],
                          add_buf_in2[col], add_buf_in1[col])

            def _emit_one_add(_cl, _bo, _b2, _b1, zero_bf16, c0, perm, vec_ty):
                use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                for i in range_(0, ADD_CHUNK, 16):
                    sub1 = memref.subview(_b1, [i], [16], [1])
                    sub2 = memref.subview(_b2, [i], [16], [1])
                    subo = memref.subview(_bo, [i], [16], [1])
                    v1 = vector.transfer_read(
                        vec_ty, sub1, [c0], permutation_map=perm,
                        padding=zero_bf16, in_bounds=[True])
                    v2 = vector.transfer_read(
                        vec_ty, sub2, [c0], permutation_map=perm,
                        padding=zero_bf16, in_bounds=[True])
                    vsum = arith.addf(v1, v2)
                    vector.transfer_write(None, vsum, subo, [c0],
                                          permutation_map=perm, in_bounds=[True])
                use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                use_lock(_cl["out_full"], LockAction.Release, value=1)

            def _make_add_core(_ct, _cl, _bo, _b2, _b1, _al=_al, _ab=_ab,
                               _artp=a_rtp.get(col)):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    c0 = arith.constant(0, T.index())
                    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                    vec_ty = T.vector(16, T.bf16())
                    if _al is not None:
                        c0_i32 = arith.constant(0, T.i32())
                        _i32 = T.i32()
                        _c64 = arith.constant(64, _i32)
                        _c0i = arith.constant(0, _i32)
                        for _ in range_(_sys.maxsize):
                            # WAVE 0: GQA decode attention (online softmax).
                            use_lock(_al["o_done"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_al["q_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if _artp is not None:
                                from aie.dialects import arith as _ar0
                                _qflat = memref.collapse_shape(
                                    bf16_memref(A_TILE_SIZE, memory_space=2),
                                    _ab["q"], [[0, 1]])
                                _Lidx = arith.constant(32, T.index())
                                _Lbf = memref.load(_qflat, [_Lidx])
                                _L = _ar0.fptosi(_i32, _Lbf)
                            g_flat = memref.collapse_shape(
                                bf16_memref(A_TILE_SIZE, memory_space=2),
                                _ab["g"], [[0, 1]])
                            attn_kernels["zero_fill_gp"](_ab["gp"])
                            attn_kernels["zero_fill_sp"](_ab["sprun"])
                            attn_kernels["neg_inf_fill_up"](_ab["up"])
                            _fills = A_N_BUF_FILLS if _artp is not None else 1
                            _cpb = A_CHUNKS_PER_BUF if _artp is not None else 1
                            for _f in range(_fills):
                                use_lock(_al["k_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_al["v_ready"], LockAction.AcquireGreaterEqual, value=1)
                                for _ci in range(_cpb):
                                    _c = _f * _cpb + _ci
                                    attn_kernels["zero_fill_g"](g_flat)
                                    if _artp is not None:
                                        _shift = arith.constant(
                                            _ci * A_TILE_SIZE * 2, T.index())
                                        _kc = memref.view(_A_QK_TY, _ab["k"],
                                                          _shift, [])
                                        _vc = memref.view(_A_V_TY, _ab["v"],
                                                          _shift, [])
                                    else:
                                        _kc, _vc = _ab["k"], _ab["v"]
                                    attn_kernels["matmul_a_b"](_ab["q"], _kc, g_flat)
                                    if _artp is not None:
                                        from aie.dialects import arith as _ar
                                        _off = arith.constant(64 * _c, _i32)
                                        _rem = _ar.subi(_L, _off)
                                        _b = _ar.minsi(_rem, _c64)
                                        _b = _ar.maxsi(_b, _c0i)
                                        _c2attn_mask_invalid_cols_rtp(_ab["g"], _b, c0)
                                    elif _c == A_N_CHUNKS - 1 and A_LAST_VALID < A_KVP:
                                        _c2attn_mask_invalid_cols(_ab["g"], A_LAST_VALID, c0)
                                    attn_kernels["fused_softmax"](g_flat, _ab["up"], _ab["sp"], _ab["r"])
                                    attn_kernels["mul_r_gp"](_ab["r"], _ab["gp"])
                                    attn_kernels["matmul_g_b"](g_flat, _vc, _ab["gp"])
                                    attn_kernels["accum_sp_r_s"](_ab["sprun"], _ab["r"], _ab["sp"])
                                    attn_kernels["vector_copy_32"](c0_i32, _ab["sp"], _ab["sprun"])
                                use_lock(_al["k_avail"], LockAction.Release, value=1)
                                use_lock(_al["v_avail"], LockAction.Release, value=1)
                            attn_kernels["div_gp_sp"](_ab["sprun"], _ab["gp"])
                            use_lock(_al["q_avail"], LockAction.Release, value=1)
                            use_lock(_al["o_full"], LockAction.Release, value=1)
                            # WAVE 1 (add1) and WAVE 2 (add2).
                            _emit_one_add(_cl, _bo, _b2, _b1, zero_bf16, c0, perm, vec_ty)
                            _emit_one_add(_cl, _bo, _b2, _b1, zero_bf16, c0, perm, vec_ty)
                        return
                    for _ in range_(_sys.maxsize):
                        _emit_one_add(_cl, _bo, _b2, _b1, zero_bf16, c0, perm, vec_ty)
            _make_add_core(add_tiles[col], add_locks[col], add_buf_out[col],
                           add_buf_in2[col], add_buf_in1[col])

        # --- swiglu row 4 (verbatim C1) ---
        for col in reversed(range(N_COLS)):
            def _make_sw_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=SWIGLU_CHUNK, packet=(0, 6))
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b1, offset=0, len=SWIGLU_CHUNK)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_b2, offset=0, len=SWIGLU_CHUNK)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_sw_mem(sw_tiles[col], sw_locks[col], sw_buf_out[col],
                         sw_buf_in2[col], sw_buf_in1[col])

            def _make_sw_core(_ct, _cl, _bo, _b2, _b1):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    n_c = arith.constant(SWIGLU_CHUNK, T.i32())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        silu_fn(_b1, _b2, _bo, n_c)
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_sw_core(sw_tiles[col], sw_locks[col], sw_buf_out[col],
                          sw_buf_in2[col], sw_buf_in1[col])

        # --- down row 5 (with_down): K=8192 herd, x resident once/token ---
        N_CHUNKS_DN = EMB_DIM // N_COLS // M_TILE_K8192   # 128
        if with_down:
            for col in reversed(range(N_COLS)):
                def _make_dn_mem(_ct, _cl, _yb, _wb, _xb):
                    @mem(_ct)
                    def _core_mem(block):
                        dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                        with block[1]:
                            use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_yb, offset=0, len=M_TILE_K8192)
                            use_lock(_cl["y_done"], LockAction.Release, value=1)
                            next_bd(block[1])
                        with block[2]:
                            EndOp()
                        with block[3]:
                            dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                        with block[4]:
                            use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_xb, offset=0, len=HIDDEN_DIM)
                            use_lock(_cl["x_ready"], LockAction.Release, value=1)
                            next_bd(block[4])
                        with block[5]:
                            dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * row_bytes8192)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                _make_dn_mem(dn_tiles[col], dn_locks[col], dn_buf_y[col],
                             dn_buf_w[col], dn_buf_x[col])

                def _make_dn_core(_ct, _cl, _yb, _wb, _xb):
                    import sys as _sys

                    @core(_ct)
                    def _core_body():
                        k_total = arith.constant(HIDDEN_DIM, T.i32())
                        k_tile_c = arith.constant(K_TILE_K8192, T.i32())
                        zero_off = arith.constant(0, T.i32())
                        zero_bf16 = arith.constant(0.0, T.bf16())
                        for _ in range_(_sys.maxsize):
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            for _c in range_(N_CHUNKS_DN):
                                use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                                dn_fill_fn(zero_bf16, _yb)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                dn_matvec_fn(k_tile_c, k_total, zero_off, _wb, _xb, _yb)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                                use_lock(_cl["y_full"], LockAction.Release, value=1)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                _make_dn_core(dn_tiles[col], dn_locks[col], dn_buf_y[col],
                              dn_buf_w[col], dn_buf_x[col])

        # --- routing ---
        if attn_wave0:
            # Attention wave-0 flows (pkt 16, distinct from 1/2/4/8 on MM2S0/1
            # and from output ids 1/5/6/7 on S2MM0). q+k: shim MM2S0 -> add
            # S2MM0; v: shim MM2S1 -> add S2MM1; gp out: add MM2S0 -> shim
            # S2MM0. Time-disjoint with the add/swiglu/matvec/down waves.
            for col in range(n_groups):
                packetflow(
                    pkt_id=16,
                    source=shim_tiles[col], source_port=WireBundle.DMA,
                    source_channel=0,
                    dests={"dest": add_tiles[col], "port": WireBundle.DMA,
                           "channel": 0},
                )
                packetflow(
                    pkt_id=16,
                    source=shim_tiles[col], source_port=WireBundle.DMA,
                    source_channel=1,
                    dests={"dest": add_tiles[col], "port": WireBundle.DMA,
                           "channel": 1},
                )
                packetflow(
                    pkt_id=16,
                    source=add_tiles[col], source_port=WireBundle.DMA,
                    source_channel=0,
                    dests={"dest": shim_tiles[col], "port": WireBundle.DMA,
                           "channel": 0},
                )
        for col in range(N_COLS):
            packetflow(
                pkt_id=1,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=2,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=4,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": sw_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            if with_down:
                packetflow(
                    pkt_id=8,
                    source=shim_tiles[col], source_port=WireBundle.DMA,
                    source_channel=0,
                    dests={"dest": mem_tiles[col], "port": WireBundle.DMA,
                           "channel": 2},
                )
        if _memx:
            # Per-column X: shim[c] MM2S1 -> mem[c] S2MM5 (pkt 16, local, no
            # E/W fan), then mem[c] MM2S5 -> mat[c] DMA0 (circuit). pkt 16 is a
            # distinct single bit so it never aliases add(2)/sw(4) on MM2S1.
            for col in range(N_COLS):
                packetflow(
                    pkt_id=16,
                    source=shim_tiles[col], source_port=WireBundle.DMA,
                    source_channel=1,
                    dests={"dest": mem_tiles[col], "port": WireBundle.DMA,
                           "channel": 5},
                )
        else:
            packetflow(
                pkt_id=1,
                source=shim_tiles[_xcol], source_port=WireBundle.DMA, source_channel=1,
                dests=[{"dest": mat_tiles[c], "port": WireBundle.DMA, "channel": 0}
                       for c in range(N_COLS)],
            )
        if with_down:
            packetflow(
                pkt_id=8,
                source=shim_tiles[0], source_port=WireBundle.DMA, source_channel=1,
                dests=[{"dest": dn_tiles[c], "port": WireBundle.DMA, "channel": 0}
                       for c in range(N_COLS)],
            )
        for col in range(N_COLS):
            packetflow(
                pkt_id=2,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=1,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
            packetflow(
                pkt_id=4,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=1,
                dests={"dest": sw_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
            if _memx:
                # X relay: mem[c] MM2S5 -> mat[c] DMA0 (the activation input).
                flow(mem_tiles[col], WireBundle.DMA, 5, mat_tiles[col], WireBundle.DMA, 0)
            if with_down:
                flow(mem_tiles[col], WireBundle.DMA, 2, dn_tiles[col], WireBundle.DMA, 1)
                flow(dn_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 3)
        for col in range(N_COLS):
            packetflow(
                pkt_id=1,
                source=mem_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=5,
                source=add_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=6,
                source=sw_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            if with_down:
                packetflow(
                    pkt_id=7,
                    source=mem_tiles[col], source_port=WireBundle.DMA,
                    source_channel=3,
                    dests={"dest": shim_tiles[col], "port": WireBundle.DMA,
                           "channel": 0},
                )

        # --- shim DMA allocations ---
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{W_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_channel_{A0_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_channel_{SG_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_channel_{A1_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 1)
            shim_dma_allocation(f"air_channel_{SU_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 1)
            shim_dma_allocation(f"air_channel_{YO_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
            shim_dma_allocation(f"air_channel_{AO_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
            shim_dma_allocation(f"air_channel_{SO_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
            if with_down:
                shim_dma_allocation(f"air_channel_{DW_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.MM2S, 0)
                shim_dma_allocation(f"air_channel_{DO_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.S2MM, 0)
        if _memx:
            for col in range(N_COLS):
                shim_dma_allocation(f"air_channel_{X_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.MM2S, 1)
        else:
            shim_dma_allocation(f"air_channel_{X_CH}",
                                shim_tiles[_xcol], DMAChannelDir.MM2S, 1)
        if with_down:
            shim_dma_allocation(f"air_channel_{DX_CH}",
                                shim_tiles[0], DMAChannelDir.MM2S, 1)
        if attn_wave0:
            for col in range(n_groups):
                shim_dma_allocation(f"air_channel_{AQ_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.MM2S, 0)
                shim_dma_allocation(f"air_channel_{AK_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.MM2S, 0)
                shim_dma_allocation(f"air_channel_{AV_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.MM2S, 1)
                shim_dma_allocation(f"air_channel_{APO_CH}_{col}",
                                    shim_tiles[col], DMAChannelDir.S2MM, 0)

        # --- mem tile DMAs: matvec W/y chains + (with_down) down W/y chains ---
        def _make_memtile_dma(_col, _ml, _w, _y, _dw, _dy, _mxb=None,
                              _w1=None, _dw1=None):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                end_blk = 2
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE, packet=(0, 1))
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                if _w1 is None:
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                else:
                    # WRELAY2: 2-slot L2 W-relay ping-pong (warm-reuse fix).
                    # MM2S1 drain alternates _w/_w1 (blocks 4,17); S2MM0 fill
                    # alternates _w/_w1 (blocks 6,18).  w_dma_done init=1 (single
                    # standing credit on a 2-slot ring) -> bit-identical warm
                    # reuse.  Even fill count per wave returns the BD to slot 0.
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[17])
                    with block[17]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[18])
                    with block[18]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                with block[7]:
                    _after_y = block[9] if (_dw is not None or _mxb is not None) \
                        else block[2]
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=_after_y)
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                if _mxb is not None:
                    # X relay ring: shim[c] -> mem[c] (S2MM5) -> mat[c] (MM2S5).
                    # 3 slots/token matching the mat DMA0 chain lengths:
                    # O=EMB (attn_out), gate=2*EMB, up=2*EMB ([res1|norm_w]).
                    # Odd channel 5 -> BD ids in the 24-47 pool (clear of the
                    # even W/y chains' low ids); pin them to avoid collisions.
                    _xlens = [EMB_DIM, 2 * EMB_DIM, 2 * EMB_DIM]
                    with block[9]:
                        dma_start(DMAChannelDir.S2MM, 5, dest=block[10],
                                  chain=block[13])
                    for _i, _ln in enumerate(_xlens):
                        with block[10 + _i]:
                            use_lock(_ml["x_empty"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_mxb, offset=0, len=_ln, bd_id=24 + _i)
                            use_lock(_ml["x_full"], LockAction.Release, value=1)
                            next_bd(block[10 + ((_i + 1) % 3)])
                    with block[13]:
                        dma_start(DMAChannelDir.MM2S, 5, dest=block[14],
                                  chain=block[2])
                    for _i, _ln in enumerate(_xlens):
                        with block[14 + _i]:
                            use_lock(_ml["x_full"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_mxb, offset=0, len=_ln, bd_id=27 + _i)
                            use_lock(_ml["x_empty"], LockAction.Release, value=1)
                            next_bd(block[14 + ((_i + 1) % 3)])
                if _dw is not None:
                    with block[9]:
                        dma_start(DMAChannelDir.MM2S, 2, dest=block[10], chain=block[11])
                    if _dw1 is None:
                        with block[10]:
                            use_lock(_ml["dw_ready"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_dma_done"], LockAction.Release, value=1)
                            next_bd(block[10])
                    else:
                        # WRELAY2 down relay: 2-slot ping-pong (blocks 10,19 drain;
                        # 14,20 fill).  Same single-credit even-count discipline.
                        with block[10]:
                            use_lock(_ml["dw_ready"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_dma_done"], LockAction.Release, value=1)
                            next_bd(block[19])
                        with block[19]:
                            use_lock(_ml["dw_ready"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw1, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_dma_done"], LockAction.Release, value=1)
                            next_bd(block[10])
                    with block[11]:
                        dma_start(DMAChannelDir.MM2S, 3, dest=block[12], chain=block[13])
                    with block[12]:
                        use_lock(_ml["dy_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_dy, offset=0, len=M_TILE_K8192, packet=(0, 7))
                        use_lock(_ml["dy_done"], LockAction.Release, value=1)
                        next_bd(block[12])
                    with block[13]:
                        dma_start(DMAChannelDir.S2MM, 2, dest=block[14], chain=block[15])
                    if _dw1 is None:
                        with block[14]:
                            use_lock(_ml["dw_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_ready"], LockAction.Release, value=1)
                            next_bd(block[14])
                    else:
                        with block[14]:
                            use_lock(_ml["dw_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_ready"], LockAction.Release, value=1)
                            next_bd(block[20])
                        with block[20]:
                            use_lock(_ml["dw_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_dw1, offset=0, len=M_TILE_K8192 * row_bytes8192)
                            use_lock(_ml["dw_ready"], LockAction.Release, value=1)
                            next_bd(block[14])
                    with block[15]:
                        dma_start(DMAChannelDir.S2MM, 3, dest=block[16], chain=block[2])
                    with block[16]:
                        use_lock(_ml["dy_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_dy, offset=0, len=M_TILE_K8192)
                        use_lock(_ml["dy_ready"], LockAction.Release, value=1)
                        next_bd(block[16])
        for col in range(N_COLS):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col], mem_buf_y[col],
                              mem_buf_dw.get(col), mem_buf_dy.get(col),
                              mem_buf_x.get(col),
                              _w1=mem_buf_w1.get(col),
                              _dw1=mem_buf_dw1.get(col))

        # --- runtime sequence ---
        if attn_wave0:
            _seq_host_tys = _awq_c2_attn_host_arg_types(
                group_size=group_size, n_groups=n_groups,
                resident=attn_resident)
        else:
            _seq_host_tys = _awq_host_arg_types(group_size=group_size)

        @runtime_sequence(*_seq_host_tys, sym_name=f"{sym}_sequence")
        def _seq(*args):
            def _attn_wave0():
                # WAVE 0: feed q (arg15), per-group k/v (arg16/arg17), receive
                # the per-group context into the wide attn_out scratch (arg1,
                # n_groups*4096). Each group writes its full 64x64 untiled tile
                # to scratch[g*4096 ..]; the O wave gathers rows 0..3 head-major.
                # Transplanted verbatim from o_gemv_ffn.py::_emit_call2_c2.
                arg_q, arg_k, arg_v = args[15], args[16], args[17]
                arg_attn_out = args[1]
                in_tasks, out_tasks = [], []
                for g in range(n_groups):
                    q_off = g * A_TILE_SIZE
                    kv_base = g * A_KV_SIZE
                    qt = dma_configure_task_for(f"air_channel_{AQ_CH}_{g}")
                    with bds(qt) as bd:
                        with bd[0]:
                            dma_bd(arg_q, offset=q_off, len=A_TILE_SIZE,
                                   dimensions=A_TILE_IN_DIMS, packet=(0, 16))
                            EndOp()
                    dma_start_task(qt)
                    in_tasks.append(qt)
                    if _A_MEMKV:
                        # MEMKV: the FULL per-group K/V cache streams in ONE shim
                        # BD each (pkt 16 -> add S2MM0/S2MM1, the proven routing).
                        kt = dma_configure_task_for(f"air_channel_{AK_CH}_{g}")
                        with bds(kt) as bd:
                            with bd[0]:
                                dma_bd(arg_k, offset=kv_base, len=A_KV_SIZE,
                                       packet=(0, 16))
                                EndOp()
                        dma_start_task(kt)
                        in_tasks.append(kt)
                        vt = dma_configure_task_for(f"air_channel_{AV_CH}_{g}")
                        with bds(vt) as bd:
                            with bd[0]:
                                dma_bd(arg_v, offset=kv_base, len=A_KV_SIZE,
                                       packet=(0, 16))
                                EndOp()
                        dma_start_task(vt)
                        in_tasks.append(vt)
                        continue_chunks = False
                    elif attn_resident:
                        # Resident: K/V fed as A_N_BUF_FILLS DMAs of
                        # A_CHUNKS_PER_BUF chunks each (host pre-tiles 8x8
                        # col-block-major -> FLAT copies; L folded in q padding).
                        _buf_elems = A_CHUNKS_PER_BUF * A_TILE_SIZE
                        for _f in range(A_N_BUF_FILLS):
                            kt = dma_configure_task_for(f"air_channel_{AK_CH}_{g}")
                            with bds(kt) as bd:
                                with bd[0]:
                                    dma_bd(arg_k,
                                           offset=kv_base + _f * _buf_elems,
                                           len=_buf_elems, packet=(0, 16))
                                    EndOp()
                            dma_start_task(kt)
                            in_tasks.append(kt)
                            vt = dma_configure_task_for(f"air_channel_{AV_CH}_{g}")
                            with bds(vt) as bd:
                                with bd[0]:
                                    dma_bd(arg_v,
                                           offset=kv_base + _f * _buf_elems,
                                           len=_buf_elems, packet=(0, 16))
                                    EndOp()
                            dma_start_task(vt)
                            in_tasks.append(vt)
                        continue_chunks = False
                    else:
                        continue_chunks = True
                    for c in range(A_N_CHUNKS if continue_chunks else 0):
                        kt = dma_configure_task_for(f"air_channel_{AK_CH}_{g}")
                        with bds(kt) as bd:
                            with bd[0]:
                                dma_bd(arg_k, offset=kv_base + c * A_TILE_SIZE,
                                       len=A_TILE_SIZE,
                                       dimensions=A_TILE_IN_DIMS, packet=(0, 16))
                                EndOp()
                        dma_start_task(kt)
                        in_tasks.append(kt)
                        vt = dma_configure_task_for(f"air_channel_{AV_CH}_{g}")
                        with bds(vt) as bd:
                            with bd[0]:
                                dma_bd(arg_v, offset=kv_base + c * A_TILE_SIZE,
                                       len=A_TILE_SIZE,
                                       dimensions=A_TILE_IN_DIMS, packet=(0, 16))
                                EndOp()
                        dma_start_task(vt)
                        in_tasks.append(vt)
                    ot = dma_configure_task_for(f"air_channel_{APO_CH}_{g}",
                                                issue_token=True)
                    with bds(ot) as bd:
                        with bd[0]:
                            dma_bd(arg_attn_out, offset=g * A_TILE_SIZE,
                                   len=A_TILE_SIZE,
                                   dimensions=A_TILE_OUT_DIMS)
                            EndOp()
                    dma_start_task(ot)
                    out_tasks.append(ot)
                for t in out_tasks:
                    dma_await_task(t)
                for t in in_tasks:
                    dma_free_task(t)

            def _x_once(chan_name, bd_emit, pid):
                t = dma_configure_task_for(chan_name, repeat_count=0)
                with bds(t) as bd:
                    bd_emit(bd, pid)
                dma_start_task(t)
                return t

            def _mat_wave(arg_w, arg_y, out_rows, x_emit):
                # X feed: per-column shim[c]->mem[c] (pkt 16) when _memx (the
                # fan-free fix); else the single-source shim broadcast (pkt 1).
                if _memx:
                    x_tasks = [_x_once(f"air_channel_{X_CH}_{c}", x_emit, 16)
                               for c in range(N_COLS)]
                else:
                    x_tasks = [_x_once(f"air_channel_{X_CH}", x_emit, 1)]
                n_outer = out_rows // 1024
                for outer in range(n_outer):
                    weight_tasks = []
                    for col in range(N_COLS):
                        t = dma_configure_task_for(f"air_channel_{W_CH}_{col}")
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(
                                    arg_w,
                                    offset=outer * weight_outer_stride
                                    + col * weight_col_stride,
                                    len=w_len, dimensions=w_dims, packet=(0, 1))
                                EndOp()
                        dma_start_task(t)
                        weight_tasks.append(t)
                    out_tasks = []
                    for col in range(N_COLS):
                        t = dma_configure_task_for(
                            f"air_channel_{YO_CH}_{col}", issue_token=True)
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(
                                    arg_y,
                                    offset=outer * output_outer_stride
                                    + col * output_col_stride,
                                    len=y_len, dimensions=y_dims)
                                EndOp()
                        dma_start_task(t)
                        out_tasks.append(t)
                    for t in reversed(out_tasks):
                        dma_await_task(t)
                    for t in reversed(weight_tasks):
                        dma_free_task(t)
                for t in reversed(x_tasks):
                    dma_free_task(t)

            def _eltwise_wave(in0_name, in1_name, out_name, arg_in0, arg_in1,
                              arg_out, chunk, dims, pkt_id):
                in0_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{in0_name}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_in0, offset=col * chunk, len=chunk,
                                   dimensions=dims, packet=(0, pkt_id))
                            EndOp()
                    dma_start_task(t)
                    in0_tasks.append(t)
                in1_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{in1_name}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_in1, offset=col * chunk, len=chunk,
                                   dimensions=dims, packet=(0, pkt_id))
                            EndOp()
                    dma_start_task(t)
                    in1_tasks.append(t)
                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_name}_{col}", issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_out, offset=col * chunk, len=chunk,
                                   dimensions=dims)
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)
                for t in reversed(out_tasks):
                    dma_await_task(t)
                for t in reversed(in1_tasks):
                    dma_free_task(t)
                for t in reversed(in0_tasks):
                    dma_free_task(t)

            def _o_x(bd, pid):
                if attn_wave0:
                    # arg1 is the wide attn_out scratch (n_groups*4096), each
                    # group's full 64x64 untiled tile. Gather rows 0..3 (the 4
                    # real GQA heads = 256 elems) of each group, head-major, to
                    # reconstruct the flat (2048,) O-proj activation: per group
                    # g, 256 contiguous from g*4096.
                    with bd[0]:
                        dma_bd(args[1], offset=0, len=EMB_DIM,
                               dimensions=[(8, 4096), (256, 1)], packet=(0, pid))
                        EndOp()
                    return
                with bd[0]:
                    dma_bd(args[1], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)], packet=(0, pid))
                    EndOp()

            def _packed_x(bd, pid):
                if _plain_gate:
                    with bd[0]:
                        dma_bd(args[6], offset=0, len=EMB_DIM,
                               dimensions=[(4, 512), (512, 1)], packet=(0, pid))
                        EndOp()
                    return
                with bd[0]:
                    dma_bd(args[4], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)], packet=(0, pid))
                    next_bd(bd[1])
                with bd[1]:
                    dma_bd(args[5], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)], packet=(0, pid))
                    EndOp()

            # Debug knob (deadlock bisect): number of stages to emit, 1..7.
            import os as _os
            _n_stages = int(_os.environ.get("PYTHOC_C2_STAGES", "7"))
            # WAVE 0 (c2_attn): GQA decode attention -> attn_out scratch (arg1).
            if attn_wave0:
                _attn_wave0()
            # 1: O proj  wo x attn_out -> proj
            _mat_wave(args[0], args[2], EMB_DIM, _o_x)
            if _n_stages < 2:
                return
            # 2: add1   proj + x_resid -> res1
            _eltwise_wave(A0_CH, A1_CH, AO_CH, args[2], args[3], args[4],
                          ADD_CHUNK, [(ADD_CHUNK, 1)], 2)
            if _n_stages < 3:
                return
            # 3/4: gate, up (rms fused on-core from [res1|norm_w])
            _mat_wave(args[7], args[8], HIDDEN_DIM, _packed_x)
            if _n_stages < 4:
                return
            _mat_wave(args[9], args[10], HIDDEN_DIM, _packed_x)
            if _n_stages < 5:
                return
            # 5: swiglu  SiLU(gate) * up -> swiglu
            _eltwise_wave(SG_CH, SU_CH, SO_CH, args[8], args[10], args[11],
                          SWIGLU_CHUNK, [(2, 512), (512, 1)], 4)
            if not with_down or _n_stages < 6:
                return
            # 6: down  wdown x swiglu -> down
            def _down_x(bd, pid):
                with bd[0]:
                    dma_bd(args[11], offset=0, len=HIDDEN_DIM,
                           dimensions=[(16, 512), (512, 1)], packet=(0, pid))
                    EndOp()
            dx_task = _x_once(f"air_channel_{DX_CH}", _down_x, 8)
            for outer in range(d_n_outer):
                dw_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{DW_CH}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                args[12],
                                offset=outer * d_w_outer_stride + col * d_w_col_stride,
                                len=w_len, dimensions=w_dims, packet=(0, 8))
                            EndOp()
                    dma_start_task(t)
                    dw_tasks.append(t)
                do_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{DO_CH}_{col}", issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(args[13], offset=outer * 256 + col * M_TILE_K8192,
                                   len=d_y_len, dimensions=d_y_dims)
                            EndOp()
                    dma_start_task(t)
                    do_tasks.append(t)
                for t in reversed(do_tasks):
                    dma_await_task(t)
                for t in reversed(dw_tasks):
                    dma_free_task(t)
            dma_free_task(dx_task)
            # 7: add2   down + res1 -> output
            _eltwise_wave(A0_CH, A1_CH, AO_CH, args[13], args[4], args[14],
                          ADD_CHUNK, [(ADD_CHUNK, 1)], 2)


def _emit_dispatcher_device(group_size: int = GROUP_SIZE,
                            dispatch_sequence=None, *,
                            attn_wave0: bool = False,
                            attn_resident: bool = False) -> None:
    """Emit the unnamed top-level dispatcher device.

    Fires the segments in pipeline order. Default (unpacked):
        og -> a1 -> rm -> gg -> ug -> sw -> dg -> a2.
    Packed modes pass a shorter ``dispatch_sequence`` naming the merged
    devices. All segments share the AWQ 15-arg host signature.

    Under ``attn_wave0`` (c2_attn) the dispatcher uses the EXTENDED 18-arg ABI
    (q/k/v appended, arg1 widened) so it can forward the attention inputs to the
    merged c2_attn device's runtime_sequence -- the AWQ counterpart of the BF16
    c2_attn dispatcher ABI.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    if dispatch_sequence is None:
        dispatch_sequence = _DEFAULT_AWQ_DISPATCH_SEQUENCE

    if attn_wave0:
        _disp_host_tys = _awq_c2_attn_host_arg_types(
            group_size=group_size, n_groups=N_COLS, resident=attn_resident)
    else:
        _disp_host_tys = _awq_host_arg_types(group_size=group_size)

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *_disp_host_tys,
            sym_name="o_gemv_ffn_awq",
        )
        def _outer(*args):
            for sym in dispatch_sequence:
                cfg = ConfigureOp(symbol=sym)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(
                        runtime_sequence_symbol=f"{sym}_sequence",
                        args=list(args),
                    )


# ---------------------------------------------------------------------------
# Device-packing emitters (AWQ counterparts of o_gemv_ffn.py's pack helpers).
# These merge an AWQ matvec phase with its following element-wise post-op into
# one aie.device, chaining the intermediate through L2 memtile instead of DDR
# and packet-routing the shim MM2S0 inputs. Structurally identical to the
# BF16 pack emitters except for the weight-side specifics (uint4 packed rows,
# row_bytes width, AWQ kernel symbols, byte-length weight DMAs).
# ---------------------------------------------------------------------------
def _emit_awq_matvec_add_pack_k2048(
    sym: str,
    matvec_sym: str,
    add_sym: str,
    *,
    weight_arg_idx: int,
    input_arg_idx: int,
    residual_arg_idx: int,
    output_arg_idx: int,
    out_rows: int = EMB_DIM,
    group_size: int = GROUP_SIZE,
    pingpong_w: bool = False,
) -> None:
    """Pack one K=2048 AWQ matvec with its following residual add (D1: og->a1).

    The add consumes the matvec's per-column strided output partition in L2 and
    writes the global output with the matvec output BD dimensions. Mirrors
    ``builders/o_gemv_ffn.py::_emit_matvec_add_pack_k2048`` with AWQ weights.

    ``pingpong_w`` double-buffers the matvec L1 weight buffer via outer-loop
    unroll (same scheme as ``_emit_awq_matvec_seg_k2048``); the add post-op is
    untouched.
    """
    mat_chans = _CHANNELS[matvec_sym]
    add_chans = _CHANNELS[add_sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024
    if pingpong_w:
        rows_per_col = 1024 // N_COLS
        assert rows_per_col % (2 * M_TILE) == 0, (
            f"pingpong_w outer unroll-by-2 needs rows_per_col "
            f"({rows_per_col}) divisible by 2*M_TILE ({2 * M_TILE})"
        )

    row_bytes = _combined_row_bytes(EMB_DIM, group_size)    # 1088
    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    post_chunk = y_len
    x_repeat_count = 31
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE * row_bytes          # 8 * 1088 = 8704
    weight_outer_stride = 1024 * row_bytes          # 1024 * 1088 = 1_114_112
    output_col_stride = M_TILE
    output_outer_stride = 1024

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        mat_tiles = [tile(c, 2) for c in range(N_COLS)]
        add_tiles = [tile(c, 3) for c in range(N_COLS)]

        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        mat_locks = {}
        add_locks = {}
        _w_avail_init = 2 if pingpong_w else 1
        for col in range(N_COLS):
            mt = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(mt, lock_id=5, init=_w_avail_init),
                "w_ready": lock(mt, lock_id=4, init=0),
                "x_avail": lock(mt, lock_id=3, init=1),
                "x_ready": lock(mt, lock_id=2, init=0),
                "y_done":  lock(mt, lock_id=1, init=1),
                "y_full":  lock(mt, lock_id=0, init=0),
            }
            at = add_tiles[col]
            add_locks[col] = {
                "in2_avail": lock(at, lock_id=5, init=1),
                "in2_ready": lock(at, lock_id=4, init=0),
                "in1_avail": lock(at, lock_id=3, init=1),
                "in1_ready": lock(at, lock_id=2, init=0),
                "out_done":  lock(at, lock_id=1, init=1),
                "out_full":  lock(at, lock_id=0, init=0),
            }

        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE, row_bytes, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)
        _ADD_TY = bf16_memref(post_chunk, memory_space=2)

        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        mat_buf_y = {}
        mat_buf_w = {}
        mat_buf_w1 = {}  # only when pingpong_w
        mat_buf_x = {}
        add_buf_out = {}
        add_buf_res = {}
        add_buf_proj = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)
            if pingpong_w:
                mat_buf_w1[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_res[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_proj[col] = buffer(add_tiles[col], datatype=_ADD_TY)

        _emit_external_buffers(
            ((out_rows, row_bytes), "ui8"),
            ((EMB_DIM,), "bf16"),
            ((out_rows,), "bf16"),
        )

        fill_fn = external_func(
            "awq_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = mat_tiles[col]
            cl = mat_locks[col]
            y_buf = mat_buf_y[col]
            w_buf = mat_buf_w[col]
            w_buf1 = mat_buf_w1.get(col)  # None unless pingpong_w
            x_buf = mat_buf_x[col]

            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_xb, offset=0, len=EMB_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # W L1 ping-pong: 2-BD ring filling wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_mat_mem(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _wb1):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())

                    def _group(_w):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE, K_TILE):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _w, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)

                    if _wb1 is None:
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                    else:
                        # W L1 ping-pong: unroll the outer row-group loop by 2.
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                            _group(_wb1)
            _make_mat_core(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

        for col in reversed(range(N_COLS)):
            ct_op = add_tiles[col]
            cl = add_locks[col]
            buf_out = add_buf_out[col]
            buf_res = add_buf_res[col]
            buf_proj = add_buf_proj[col]

            def _make_add_mem(_ct, _cl, _bo, _bres, _bproj):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=post_chunk)
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bproj, offset=0, len=post_chunk)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bres, offset=0, len=post_chunk)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_add_mem(ct_op, cl, buf_out, buf_res, buf_proj)

            def _make_add_core(_ct, _cl, _bo, _bres, _bproj):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    c0 = arith.constant(0, T.index())
                    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                    vec_ty = T.vector(16, T.bf16())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        for i in range_(0, post_chunk, 16):
                            sub1 = memref.subview(_bproj, [i], [16], [1])
                            sub2 = memref.subview(_bres, [i], [16], [1])
                            subo = memref.subview(_bo, [i], [16], [1])
                            v1 = vector.transfer_read(
                                vec_ty, sub1, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            v2 = vector.transfer_read(
                                vec_ty, sub2, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            vsum = arith.addf(v1, v2)
                            vector.transfer_write(
                                None, vsum, subo, [c0],
                                permutation_map=perm, in_bounds=[True])
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_add_core(ct_op, cl, buf_out, buf_res, buf_proj)

        # Packet-route shim MM2S0: pkt 0 feeds matvec weights, pkt 1 feeds the
        # residual add operand. The matvec input broadcast uses shim0 MM2S1.
        for col in range(N_COLS):
            packetflow(
                pkt_id=0,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=1,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, add_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(add_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        weight_base = mat_chans["weight_base"]
        input_chan = mat_chans["input"]
        residual_chan = add_chans["in1"]
        out_chan = add_chans["out"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{weight_base}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{residual_chan}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        shim_dma_allocation(
            f"air_channel_{input_chan}", shim_tiles[0], DMAChannelDir.MM2S, 1,
        )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col], DMAChannelDir.S2MM, 0,
            )

        def _make_memtile_dma(_col, _ml, _w, _y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                with block[4]:
                    use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["w_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
        for col in range(N_COLS):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col], mem_buf_y[col])

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_res = args[residual_arg_idx]
            arg_y = args[output_arg_idx]
            for outer in range(n_outer):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len,
                                dimensions=w_dims,
                                packet=(0, 0),
                            )
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(
                    f"air_channel_{input_chan}",
                    repeat_count=x_repeat_count,
                )
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(
                            arg_x, offset=0, len=EMB_DIM,
                            dimensions=[(4, 512), (512, 1)],
                        )
                        EndOp()
                dma_start_task(x_task)

                res_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{residual_chan}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_res,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len, dimensions=y_dims, packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    res_tasks.append(t)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}", issue_token=True,
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_y,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len, dimensions=y_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                for t in reversed(res_tasks):
                    dma_free_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


def _emit_awq_matvec_add_pack_k8192(
    sym: str,
    matvec_sym: str,
    add_sym: str,
    *,
    weight_arg_idx: int,
    input_arg_idx: int,
    residual_arg_idx: int,
    output_arg_idx: int,
    group_size: int = GROUP_SIZE,
    pingpong_w: bool = False,
    pingpong_w_l2: bool = False,
) -> None:
    """Pack the K=8192 AWQ down matvec with a2_add (D4: dg->a2).

    Mirrors ``builders/o_gemv_ffn.py::_emit_matvec_add_pack_k8192`` with AWQ
    packed-uint4 weights (row_bytes=4352, dg_awq_* kernels).

    ``pingpong_w`` double-buffers the matvec L1 weight buffer via outer-loop
    unroll (same scheme as the dg seg); the a2 add post-op is untouched.
    ``pingpong_w_l2`` doubles the L2 memtile W buffer and turns both memtile
    W chains (MM2S ch1 L2->L1, S2MM ch0 shim->L2) into 2-BD rings, so
    shim->L2 overlaps L2->L1.  Independent of ``pingpong_w``.
    """
    mat_chans = _CHANNELS[matvec_sym]
    add_chans = _CHANNELS[add_sym]
    n_outer = EMB_DIM // 256
    if pingpong_w:
        rows_per_col = 256 // N_COLS                  # 32
        assert rows_per_col % (2 * M_TILE_K8192) == 0, (
            f"pingpong_w outer unroll-by-2 needs rows_per_col "
            f"({rows_per_col}) divisible by 2*M_TILE_K8192 ({2 * M_TILE_K8192})"
        )

    row_bytes = _combined_row_bytes(HIDDEN_DIM, group_size)   # 4352
    y_dims = [(16, 16), (2, 1)]
    y_len = 32
    post_chunk = y_len
    x_repeat_count = 31
    x_dims = [(16, 512), (512, 1)]
    x_len = HIDDEN_DIM
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE_K8192 * row_bytes          # 2 * 4352 = 8704
    weight_outer_stride = 256 * row_bytes                 # 256 * 4352 = 1_114_112
    output_col_stride = M_TILE_K8192
    output_outer_stride = 256

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        mat_tiles = [tile(c, 2) for c in range(N_COLS)]
        add_tiles = [tile(c, 3) for c in range(N_COLS)]

        mem_locks = {}
        _w_dma_done_init = 2 if pingpong_w_l2 else 1
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=_w_dma_done_init),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        mat_locks = {}
        add_locks = {}
        _w_avail_init = 2 if pingpong_w else 1
        for col in range(N_COLS):
            mt = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(mt, lock_id=5, init=_w_avail_init),
                "w_ready": lock(mt, lock_id=4, init=0),
                "x_avail": lock(mt, lock_id=3, init=1),
                "x_ready": lock(mt, lock_id=2, init=0),
                "y_done":  lock(mt, lock_id=1, init=1),
                "y_full":  lock(mt, lock_id=0, init=0),
            }
            at = add_tiles[col]
            add_locks[col] = {
                "in2_avail": lock(at, lock_id=5, init=1),
                "in2_ready": lock(at, lock_id=4, init=0),
                "in1_avail": lock(at, lock_id=3, init=1),
                "in1_ready": lock(at, lock_id=2, init=0),
                "out_done":  lock(at, lock_id=1, init=1),
                "out_full":  lock(at, lock_id=0, init=0),
            }

        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE_K8192, row_bytes, memory_space=2)
        _X_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE_K8192, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE_K8192, memory_space=1)
        _ADD_TY = bf16_memref(post_chunk, memory_space=2)

        mem_buf_w = {}
        mem_buf_w1 = {}  # only when pingpong_w_l2
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
            if pingpong_w_l2:
                mem_buf_w1[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        mat_buf_y = {}
        mat_buf_w = {}
        mat_buf_w1 = {}  # only when pingpong_w
        mat_buf_x = {}
        add_buf_out = {}
        add_buf_res = {}
        add_buf_down = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)
            if pingpong_w:
                mat_buf_w1[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_res[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_down[col] = buffer(add_tiles[col], datatype=_ADD_TY)

        _emit_external_buffers(
            ((EMB_DIM, row_bytes), "ui8"),
            ((HIDDEN_DIM,), "bf16"),
            ((EMB_DIM,), "bf16"),
        )

        fill_fn = external_func(
            "dg_awq_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_AWQ_MV_K8192,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "dg_awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_AWQ_MV_K8192,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = mat_tiles[col]
            cl = mat_locks[col]
            y_buf = mat_buf_y[col]
            w_buf = mat_buf_w[col]
            w_buf1 = mat_buf_w1.get(col)  # None unless pingpong_w
            x_buf = mat_buf_x[col]

            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE_K8192)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_xb, offset=0, len=HIDDEN_DIM)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # W L1 ping-pong: 2-BD ring filling wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE_K8192 * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_mat_mem(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _wb1):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(HIDDEN_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE_K8192, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())

                    def _group(_w):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _w, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)

                    if _wb1 is None:
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                    else:
                        # W L1 ping-pong: unroll the outer row-group loop by 2.
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                            _group(_wb1)
            _make_mat_core(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

        for col in reversed(range(N_COLS)):
            ct_op = add_tiles[col]
            cl = add_locks[col]
            buf_out = add_buf_out[col]
            buf_res = add_buf_res[col]
            buf_down = add_buf_down[col]

            def _make_add_mem(_ct, _cl, _bo, _bres, _bdown):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=post_chunk)
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bdown, offset=0, len=post_chunk)
                        use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bres, offset=0, len=post_chunk)
                        use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_add_mem(ct_op, cl, buf_out, buf_res, buf_down)

            def _make_add_core(_ct, _cl, _bo, _bres, _bdown):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    c0 = arith.constant(0, T.index())
                    perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                    vec_ty = T.vector(16, T.bf16())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        for i in range_(0, post_chunk, 16):
                            sub1 = memref.subview(_bdown, [i], [16], [1])
                            sub2 = memref.subview(_bres, [i], [16], [1])
                            subo = memref.subview(_bo, [i], [16], [1])
                            v1 = vector.transfer_read(
                                vec_ty, sub1, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            v2 = vector.transfer_read(
                                vec_ty, sub2, [c0],
                                permutation_map=perm, padding=zero_bf16,
                                in_bounds=[True])
                            vsum = arith.addf(v1, v2)
                            vector.transfer_write(
                                None, vsum, subo, [c0],
                                permutation_map=perm, in_bounds=[True])
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_add_core(ct_op, cl, buf_out, buf_res, buf_down)

        for col in range(N_COLS):
            packetflow(
                pkt_id=0,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=1,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, add_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(add_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        weight_base = mat_chans["weight_base"]
        input_chan = mat_chans["input"]
        residual_chan = add_chans["in1"]
        out_chan = add_chans["out"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{weight_base}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{residual_chan}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        shim_dma_allocation(
            f"air_channel_{input_chan}", shim_tiles[0], DMAChannelDir.MM2S, 1,
        )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col], DMAChannelDir.S2MM, 0,
            )

        def _make_memtile_dma(_col, _ml, _w, _w1, _y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE_K8192)
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                if _w1 is None:
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                else:
                    # L2 W ping-pong: 2-BD rings on both MM2S ch 1 (L2->L1)
                    # and S2MM ch 0 (shim->L2), alternating w0/w1.
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[9])
                    with block[9]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[10])
                    with block[10]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * row_bytes)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE_K8192)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
        for col in range(N_COLS):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col],
                              mem_buf_w1.get(col), mem_buf_y[col])

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_res = args[residual_arg_idx]
            arg_y = args[output_arg_idx]
            for outer in range(n_outer):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len, dimensions=w_dims, packet=(0, 0),
                            )
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(
                    f"air_channel_{input_chan}", repeat_count=x_repeat_count,
                )
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(arg_x, offset=0, len=x_len, dimensions=x_dims)
                        EndOp()
                dma_start_task(x_task)

                res_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{residual_chan}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_res,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len, dimensions=y_dims, packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    res_tasks.append(t)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}", issue_token=True,
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_y,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len, dimensions=y_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                for t in reversed(res_tasks):
                    dma_free_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


def _emit_awq_gg_ug_swiglu_pack(
    sym: str = "d3_gg_ug_sw_pack",
    *,
    gg_sym: str = "gg_awq_matvec_0",
    ug_sym: str = "ug_awq_matvec_0",
    sw_sym: str = "sw_silu_mul_seg",
    gg_weight_arg_idx: int = 7,
    ug_weight_arg_idx: int = 9,
    input_arg_idx: int = 6,
    output_arg_idx: int = 11,
    out_rows: int = HIDDEN_DIM,
    group_size: int = GROUP_SIZE,
    pingpong_w: bool = False,
    rms_fused: bool = False,
    res1_arg_idx: int = 4,
    normw_arg_idx: int = 5,
) -> None:
    """Pack gate/up K=2048 AWQ matvecs with the following SwiGLU (D3).

    Gate runs on row 2, up on row 3, SwiGLU on row 4. Mirrors
    ``builders/o_gemv_ffn.py::_emit_gg_ug_swiglu_pack`` with AWQ packed-uint4
    weights. The normed2 broadcast is split across two shim columns (gg via
    shim0 MM2S1, ug via shim1 MM2S1) -- the fix for the 16-destination
    broadcast hang documented in DEVICE_PACKING_ANALYSIS.md §13.
    """
    gg_chans = _CHANNELS[gg_sym]
    ug_chans = _CHANNELS[ug_sym]
    sw_chans = _CHANNELS[sw_sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024
    if pingpong_w:
        rows_per_col = 1024 // N_COLS
        assert rows_per_col % (2 * M_TILE) == 0, (
            f"pingpong_w outer unroll-by-2 needs rows_per_col "
            f"({rows_per_col}) divisible by 2*M_TILE ({2 * M_TILE})"
        )

    row_bytes = _combined_row_bytes(EMB_DIM, group_size)    # 1088
    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    x_repeat_count = 31
    w_dims = [(16, 69632), (16, 544), (544, 1)]
    w_len = 16 * 16 * 544  # 139264
    weight_col_stride = M_TILE * row_bytes
    weight_outer_stride = 1024 * row_bytes
    output_col_stride = M_TILE
    output_outer_stride = 1024
    post_chunk = y_len

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        gg_tiles = [tile(c, 2) for c in range(N_COLS)]
        ug_tiles = [tile(c, 3) for c in range(N_COLS)]
        sw_tiles = [tile(c, 4) for c in range(N_COLS)]

        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "gg_w_dma_done": lock(mt, lock_id=7, init=1),
                "gg_w_ready":    lock(mt, lock_id=6, init=0),
                "gg_y_done":     lock(mt, lock_id=5, init=1),
                "gg_y_ready":    lock(mt, lock_id=4, init=0),
                "ug_w_dma_done": lock(mt, lock_id=3, init=1),
                "ug_w_ready":    lock(mt, lock_id=2, init=0),
                "ug_y_done":     lock(mt, lock_id=1, init=1),
                "ug_y_ready":    lock(mt, lock_id=0, init=0),
            }

        gg_locks = {}
        ug_locks = {}
        sw_locks = {}
        _w_avail_init = 2 if pingpong_w else 1
        for col in range(N_COLS):
            gt = gg_tiles[col]
            gg_locks[col] = {
                "w_avail": lock(gt, lock_id=5, init=_w_avail_init),
                "w_ready": lock(gt, lock_id=4, init=0),
                "x_avail": lock(gt, lock_id=3, init=1),
                "x_ready": lock(gt, lock_id=2, init=0),
                "y_done":  lock(gt, lock_id=1, init=1),
                "y_full":  lock(gt, lock_id=0, init=0),
            }
            ut = ug_tiles[col]
            ug_locks[col] = {
                "w_avail": lock(ut, lock_id=5, init=_w_avail_init),
                "w_ready": lock(ut, lock_id=4, init=0),
                "x_avail": lock(ut, lock_id=3, init=1),
                "x_ready": lock(ut, lock_id=2, init=0),
                "y_done":  lock(ut, lock_id=1, init=1),
                "y_full":  lock(ut, lock_id=0, init=0),
            }
            st = sw_tiles[col]
            sw_locks[col] = {
                "up_avail":   lock(st, lock_id=5, init=1),
                "up_ready":   lock(st, lock_id=4, init=0),
                "gate_avail": lock(st, lock_id=3, init=1),
                "gate_ready": lock(st, lock_id=2, init=0),
                "out_done":   lock(st, lock_id=1, init=1),
                "out_full":   lock(st, lock_id=0, init=0),
            }

        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.extras import types as T
        from ml_dtypes import bfloat16 as _bf16

        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE, row_bytes, memory_space=2)
        # Fused RMS: activation buffer is packed [res1 | norm_w] (2*EMB_DIM);
        # else the single normed2 vector. The AWQ matvec always reads a bf16
        # [EMB_DIM] activation (normed), so its operand type is unchanged.
        _X_LEN = 2 * EMB_DIM if rms_fused else EMB_DIM
        _X_L1_TY = bf16_memref(_X_LEN, memory_space=2)
        _NORMED_L1_TY = bf16_memref(EMB_DIM, memory_space=2)   # rms scratch
        _RSCR_L1_TY = bf16_memref(16, memory_space=2)          # reduction spill
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _ui8_memref(1, M_TILE, row_bytes, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)
        _SW_TY = bf16_memref(post_chunk, memory_space=2)

        mem_buf_gg_w = {}
        mem_buf_ug_w = {}
        mem_buf_gg_y = {}
        mem_buf_ug_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_gg_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
            mem_buf_ug_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_gg_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)
            mem_buf_ug_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        gg_buf_y = {}
        gg_buf_w = {}
        gg_buf_w1 = {}  # only when pingpong_w
        gg_buf_x = {}
        gg_buf_normed = {}
        gg_buf_rscr = {}
        ug_buf_y = {}
        ug_buf_w = {}
        ug_buf_w1 = {}  # only when pingpong_w
        ug_buf_x = {}
        ug_buf_normed = {}
        ug_buf_rscr = {}
        sw_buf_out = {}
        sw_buf_up = {}
        sw_buf_gate = {}
        for col in reversed(range(N_COLS)):
            gg_buf_y[col] = buffer(gg_tiles[col], datatype=_Y_L1_TY)
            gg_buf_w[col] = buffer(gg_tiles[col], datatype=_W_L1_TY)
            gg_buf_x[col] = buffer(gg_tiles[col], datatype=_X_L1_TY)
            ug_buf_y[col] = buffer(ug_tiles[col], datatype=_Y_L1_TY)
            ug_buf_w[col] = buffer(ug_tiles[col], datatype=_W_L1_TY)
            ug_buf_x[col] = buffer(ug_tiles[col], datatype=_X_L1_TY)
            if rms_fused:
                gg_buf_normed[col] = buffer(gg_tiles[col], datatype=_NORMED_L1_TY)
                gg_buf_rscr[col] = buffer(gg_tiles[col], datatype=_RSCR_L1_TY)
                ug_buf_normed[col] = buffer(ug_tiles[col], datatype=_NORMED_L1_TY)
                ug_buf_rscr[col] = buffer(ug_tiles[col], datatype=_RSCR_L1_TY)
            if pingpong_w:
                gg_buf_w1[col] = buffer(gg_tiles[col], datatype=_W_L1_TY)
                ug_buf_w1[col] = buffer(ug_tiles[col], datatype=_W_L1_TY)
            sw_buf_out[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_up[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_gate[col] = buffer(sw_tiles[col], datatype=_SW_TY)

        _emit_external_buffers(
            ((out_rows, row_bytes), "ui8"),
            ((EMB_DIM,), "bf16"),
            ((out_rows,), "bf16"),
        )

        fill_fn = external_func(
            "awq_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        # For rms_fused the AWQ matvec reads the resident `normed` (bf16,
        # EMB_DIM) computed once per token by rms_fn; same operand type as the
        # non-fused activation, so only the SSA buffer differs.
        _mv_act_ty = _NORMED_L1_TY if rms_fused else _X_L1_TY
        matvec_fn = external_func(
            "awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _mv_act_ty, _Y_L1_TY],
            link_with=KO_AWQ_MV,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if rms_fused:
            rms_fn = external_func(
                "rms_norm_packed_bf16",
                inputs=[_X_L1_TY, _NORMED_L1_TY, _RSCR_L1_TY],
                link_with=KO_MATVEC_RMS,
            )
            rms_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        silu_fn = external_func(
            "silu_and_mul_bf16",
            inputs=[_SW_TY, _SW_TY, _SW_TY, np.int32],
            link_with=KO_SWIGLU,
        )
        silu_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=M_TILE)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_xb, offset=0, len=_X_LEN)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # W L1 ping-pong: 2-BD ring filling wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_mat_mem(
                gg_tiles[col], gg_locks[col], gg_buf_y[col], gg_buf_w[col],
                gg_buf_x[col], gg_buf_w1.get(col)
            )
            _make_mat_mem(
                ug_tiles[col], ug_locks[col], ug_buf_y[col], ug_buf_w[col],
                ug_buf_x[col], ug_buf_w1.get(col)
            )

            # Chunks (M_TILE-row AWQ matvec calls) per tile per token.
            _N_CHUNKS = out_rows // N_COLS // M_TILE

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _wb1, _normed=None, _rscr=None):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_off = arith.constant(0, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())

                    if rms_fused:
                        # air's fold: RMSNorm once per token into resident
                        # `normed`, then AWQ-matvec all chunks over it (only
                        # weights/outputs re-acquired per chunk). pingpong off
                        # for gg/ug AWQ (PINGPONG_W_K2048 = False).
                        for _ in range_(_sys.maxsize):
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            rms_fn(_xb, _normed, _rscr)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            for _c in range_(_N_CHUNKS):
                                use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                                fill_fn(zero_bf16, _yb)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, zero_off, _wb, _normed, _yb)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                                use_lock(_cl["y_full"], LockAction.Release, value=1)
                        return

                    def _group(_w):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE, K_TILE):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _w, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)

                    if _wb1 is None:
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                    else:
                        # W L1 ping-pong: unroll the outer row-group loop by 2.
                        for _ in range_(_sys.maxsize):
                            _group(_wb)
                            _group(_wb1)
            _make_mat_core(
                gg_tiles[col], gg_locks[col], gg_buf_y[col], gg_buf_w[col],
                gg_buf_x[col], gg_buf_w1.get(col),
                gg_buf_normed.get(col), gg_buf_rscr.get(col),
            )
            _make_mat_core(
                ug_tiles[col], ug_locks[col], ug_buf_y[col], ug_buf_w[col],
                ug_buf_x[col], ug_buf_w1.get(col),
                ug_buf_normed.get(col), ug_buf_rscr.get(col),
            )

            def _make_sw_mem(_ct, _cl, _bo, _bup, _bgate):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=post_chunk)
                        use_lock(_cl["out_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["gate_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bgate, offset=0, len=post_chunk)
                        use_lock(_cl["gate_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["up_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bup, offset=0, len=post_chunk)
                        use_lock(_cl["up_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_sw_mem(
                sw_tiles[col], sw_locks[col], sw_buf_out[col], sw_buf_up[col], sw_buf_gate[col]
            )

            def _make_sw_core(_ct, _cl, _bo, _bup, _bgate):
                import sys as _sys

                @core(_ct)
                def _core_body():
                    n_c = arith.constant(post_chunk, T.i32())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["gate_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["up_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        silu_fn(_bgate, _bup, _bo, n_c)
                        use_lock(_cl["gate_avail"], LockAction.Release, value=1)
                        use_lock(_cl["up_avail"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
            _make_sw_core(
                sw_tiles[col], sw_locks[col], sw_buf_out[col], sw_buf_up[col], sw_buf_gate[col]
            )

        for col in range(N_COLS):
            packetflow(
                pkt_id=0,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=1,
                source=shim_tiles[col],
                source_port=WireBundle.DMA,
                source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 2},
            )
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, gg_tiles[col], WireBundle.DMA, 0)
            flow(shim_tiles[1], WireBundle.DMA, 1, ug_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, gg_tiles[col], WireBundle.DMA, 1)
            flow(mem_tiles[col], WireBundle.DMA, 2, ug_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(gg_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
            flow(ug_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 3)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, sw_tiles[col], WireBundle.DMA, 0)
            flow(mem_tiles[col], WireBundle.DMA, 3, sw_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(sw_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        gg_weight_base = gg_chans["weight_base"]
        ug_weight_base = ug_chans["weight_base"]
        gg_input_chan = gg_chans["input"]
        ug_input_chan = ug_chans["input"]
        out_chan = sw_chans["out"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{gg_weight_base}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{ug_weight_base}_{col}",
                shim_tiles[col], DMAChannelDir.MM2S, 0,
            )
        shim_dma_allocation(
            f"air_channel_{gg_input_chan}", shim_tiles[0], DMAChannelDir.MM2S, 1,
        )
        shim_dma_allocation(
            f"air_channel_{ug_input_chan}", shim_tiles[1], DMAChannelDir.MM2S, 1,
        )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col], DMAChannelDir.S2MM, 0,
            )

        def _make_memtile_dma(_col, _ml, _gg_w, _ug_w, _gg_y, _ug_y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["gg_y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_gg_y, offset=0, len=M_TILE)
                    use_lock(_ml["gg_y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                with block[4]:
                    use_lock(_ml["gg_w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_gg_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["gg_w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.MM2S, 2, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["ug_w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["ug_w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.MM2S, 3, dest=block[8], chain=block[9])
                with block[8]:
                    use_lock(_ml["ug_y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_y, offset=0, len=M_TILE)
                    use_lock(_ml["ug_y_done"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[9]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[10], chain=block[11])
                with block[10]:
                    use_lock(_ml["gg_w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_gg_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["gg_w_ready"], LockAction.Release, value=1)
                    next_bd(block[10])
                with block[11]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[12], chain=block[13])
                with block[12]:
                    use_lock(_ml["gg_y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_gg_y, offset=0, len=M_TILE)
                    use_lock(_ml["gg_y_ready"], LockAction.Release, value=1)
                    next_bd(block[12])
                with block[13]:
                    dma_start(DMAChannelDir.S2MM, 2, dest=block[14], chain=block[15])
                with block[14]:
                    use_lock(_ml["ug_w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_w, offset=0, len=M_TILE * row_bytes)
                    use_lock(_ml["ug_w_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[16], chain=block[2])
                with block[16]:
                    use_lock(_ml["ug_y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_y, offset=0, len=M_TILE)
                    use_lock(_ml["ug_y_ready"], LockAction.Release, value=1)
                    next_bd(block[16])
        for col in range(N_COLS):
            _make_memtile_dma(
                col, mem_locks[col],
                mem_buf_gg_w[col], mem_buf_ug_w[col],
                mem_buf_gg_y[col], mem_buf_ug_y[col],
            )

        @runtime_sequence(*_awq_host_arg_types(group_size=group_size),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_gg_w = args[gg_weight_arg_idx]
            arg_ug_w = args[ug_weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_y = args[output_arg_idx]
            arg_res1 = args[res1_arg_idx]
            arg_normw = args[normw_arg_idx]

            def _emit_x_bds(_task):
                with bds(_task) as bd:
                    if rms_fused:
                        with bd[0]:
                            dma_bd(arg_res1, offset=0, len=EMB_DIM,
                                   dimensions=[(4, 512), (512, 1)])
                            next_bd(bd[1])
                        with bd[1]:
                            dma_bd(arg_normw, offset=0, len=EMB_DIM,
                                   dimensions=[(4, 512), (512, 1)])
                            EndOp()
                    else:
                        with bd[0]:
                            dma_bd(arg_x, offset=0, len=EMB_DIM,
                                   dimensions=[(4, 512), (512, 1)])
                            EndOp()

            # Fused RMS: deliver packed [res1|norm_w] once per token (constant
            # for the whole kernel; gate/up compute normed once).
            rms_gg_x_task = rms_ug_x_task = None
            if rms_fused:
                rms_gg_x_task = dma_configure_task_for(
                    f"air_channel_{gg_input_chan}", repeat_count=0)
                _emit_x_bds(rms_gg_x_task)
                dma_start_task(rms_gg_x_task)
                rms_ug_x_task = dma_configure_task_for(
                    f"air_channel_{ug_input_chan}", repeat_count=0)
                _emit_x_bds(rms_ug_x_task)
                dma_start_task(rms_ug_x_task)

            for outer in range(n_outer):
                gg_weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{gg_weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_gg_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len, dimensions=w_dims, packet=(0, 0),
                            )
                            EndOp()
                    dma_start_task(t)
                    gg_weight_tasks.append(t)

                ug_weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{ug_weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_ug_w,
                                offset=outer * weight_outer_stride + col * weight_col_stride,
                                len=w_len, dimensions=w_dims, packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    ug_weight_tasks.append(t)

                gg_x_task = ug_x_task = None
                if not rms_fused:
                    gg_x_task = dma_configure_task_for(
                        f"air_channel_{gg_input_chan}", repeat_count=x_repeat_count,
                    )
                    _emit_x_bds(gg_x_task)
                    dma_start_task(gg_x_task)

                    ug_x_task = dma_configure_task_for(
                        f"air_channel_{ug_input_chan}", repeat_count=x_repeat_count,
                    )
                    _emit_x_bds(ug_x_task)
                    dma_start_task(ug_x_task)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}", issue_token=True,
                    )
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(
                                arg_y,
                                offset=outer * output_outer_stride + col * output_col_stride,
                                len=y_len, dimensions=y_dims,
                            )
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                if not rms_fused:
                    dma_free_task(ug_x_task)
                    dma_free_task(gg_x_task)
                for t in reversed(ug_weight_tasks):
                    dma_free_task(t)
                for t in reversed(gg_weight_tasks):
                    dma_free_task(t)

            if rms_gg_x_task is not None:
                dma_free_task(rms_ug_x_task)
                dma_free_task(rms_gg_x_task)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_o_gemv_ffn_awq_module(emb_dim: int = EMB_DIM,
                                hidden_dim: int = HIDDEN_DIM,
                                *, group_size: int = GROUP_SIZE,
                                pack_mode: str = "none",
                                verbose: bool = False) -> str:
    """Build the ``o_gemv_ffn_awq`` ``aie/aiex``-dialect module.

    Both dimensions are fixed to the Llama-3.2-1B values (the cached AIR
    layout is shape-specialized). ``group_size`` is baked into the AWQ
    kernels at Stage 2 (default 128); changing it requires updating
    ``kernels/awq_mv.py`` first.

    ``pack_mode`` selects experimental device packing (AWQ counterpart of the
    BF16 ``o_gemv_ffn`` packing):
      * ``"none"``    -- 8 standalone segments (baseline).
      * ``"d1"``      -- pack og_matvec + a1_add (7 dispatches).
      * ``"d1d4"``    -- additionally pack dg_matvec + a2_add (6 dispatches).
      * ``"d1d3d4"``  -- additionally pack gg+ug+SwiGLU (4 dispatches).
    """
    if emb_dim != EMB_DIM or hidden_dim != HIDDEN_DIM:
        raise ValueError(
            f"o_gemv_ffn_awq builder is fixed to emb_dim={EMB_DIM}, "
            f"hidden_dim={HIDDEN_DIM}; got emb_dim={emb_dim}, "
            f"hidden_dim={hidden_dim}."
        )
    if group_size != GROUP_SIZE:
        raise ValueError(
            f"o_gemv_ffn_awq builder is fixed to group_size={GROUP_SIZE}; "
            f"got {group_size}. Re-baking requires updating "
            f"kernels/awq_mv.py."
        )
    if pack_mode not in {"none", "d1", "d1d4", "d1d3d4", "d1d3d4_rms",
                         "c2_rms", "c2_merged", "c2_attn"}:
        raise ValueError(f"unsupported o_gemv_ffn_awq pack_mode={pack_mode!r}")
    del verbose  # currently unused

    pack_d1 = pack_mode in {"d1", "d1d4", "d1d3d4", "d1d3d4_rms"}
    pack_d4 = pack_mode in {"d1d4", "d1d3d4", "d1d3d4_rms"}
    pack_d3 = pack_mode in {"d1d3d4", "d1d3d4_rms"}
    # air's 3-device fold: rm_rms eliminated; gate/up tiles compute the
    # RMSNorm once per token from packed res1+ffn_norm_w (see o_gemv_ffn.py).
    _rmsfuse = pack_mode == "d1d3d4_rms"

    # C2 collapse (ported from the BF16 builder): one merged device for call 2.
    # c2_rms keeps the separate D4 (down+add2); c2_merged folds it in too
    # (ONE device / ONE aiex.configure = 1 LoadPDI for the whole of call 2).
    if pack_mode in {"c2_rms", "c2_merged", "c2_attn"}:
        # c2_attn = c2_merged (uint4 O+add1+gate/up+swiglu+down+add2) with GQA
        # decode attention folded in as WAVE 0 on the row-3 add herd (the AWQ
        # counterpart of the BF16 c2_attn).  Weight-free attention (BFP576 BF16
        # attn_pythoc.o kernels) -> structurally identical to c2_merged plus the
        # transplanted wave-0.  Resident (one PDI for all positions, runtime L)
        # is gated by PYTHOC_C2_ATTN_RESIDENT=1 (set by the c2_attn host driver).
        _c2_attn = pack_mode == "c2_attn"
        _c2_down = pack_mode in {"c2_merged", "c2_attn"}
        import os as _os_c2
        # AWQ c2_attn defaults to RESIDENT (one PDI for all positions, runtime
        # L) -- the non-resident single-chunk path is seq<=64 only.  Explicit
        # PYTHOC_C2_ATTN_RESIDENT=0 disables it (for the seq<=64 micro path).
        _attn_resident = (_c2_attn
                          and _os_c2.environ.get("PYTHOC_C2_ATTN_RESIDENT",
                                                 "1") == "1")
        with mlir_mod_ctx() as ctx:
            if not _c2_down:
                _emit_awq_matvec_add_pack_k8192(
                    "d4_dg_a2_pack", "dg_awq_matvec_0", "a2_eltwise_add_seg",
                    weight_arg_idx=12, input_arg_idx=11, residual_arg_idx=4,
                    output_arg_idx=14, group_size=group_size,
                    pingpong_w=PINGPONG_W_DG, pingpong_w_l2=PINGPONG_W_L2_DG)
            _emit_awq_call2_c2(pack_mode, with_down=_c2_down,
                               group_size=group_size,
                               attn_wave0=_c2_attn,
                               attn_resident=_attn_resident)
            # The merged device's sym_name == pack_mode (c2_merged/c2_attn).
            dispatch_sequence = ((pack_mode,) if _c2_down
                                 else ("c2_rms", "d4_dg_a2_pack"))
            _emit_dispatcher_device(group_size=group_size,
                                    dispatch_sequence=dispatch_sequence,
                                    attn_wave0=_c2_attn,
                                    attn_resident=_attn_resident)
            module = ctx.module
            attach_loop_annotation_to_all_scf_for(module)
        return str(module)

    with mlir_mod_ctx() as ctx:
        # AIR emit order is reverse pipeline order:
        # a2, dg, sw, ug, gg, rm, a1, og, then dispatcher.
        if pack_d4:
            _emit_awq_matvec_add_pack_k8192(
                "d4_dg_a2_pack",
                "dg_awq_matvec_0",
                "a2_eltwise_add_seg",
                weight_arg_idx=12,
                input_arg_idx=11,
                residual_arg_idx=4,
                output_arg_idx=14,
                group_size=group_size,
                pingpong_w=PINGPONG_W_DG,
                pingpong_w_l2=PINGPONG_W_L2_DG,
            )
        else:
            _emit_eltwise_add_seg(
                "a2_eltwise_add_seg", in0_arg_idx=13, in1_arg_idx=4, out_arg_idx=14,
                group_size=group_size)
            _emit_awq_matvec_seg_k8192(
                "dg_awq_matvec_0", weight_arg_idx=12, input_arg_idx=11,
                output_arg_idx=13, group_size=group_size,
                pingpong_w=PINGPONG_W_DG, pingpong_w_l2=PINGPONG_W_L2_DG)
        # pingpong_x off: K_TILE_K8192=2 already collapsed the K-loop.
        # pingpong_w_l2 off too: tested, it improves W channel
        # utilization (starv1 22%->12%, dma_in1_eff 30%->44%) but span
        # only drops 0.4% and tok/sec is unchanged within noise -- W
        # wasn't on the critical path here. AWQ dg's remaining ~50%
        # unaccounted cycles are probably memory_stall during the
        # dequant chain, which is outside what DMA optimizations can
        # attack. Both PP infras stay plumbed.
        if pack_d3:
            _emit_awq_gg_ug_swiglu_pack(
                "d3_gg_ug_sw_pack",
                gg_sym="gg_awq_matvec_0",
                ug_sym="ug_awq_matvec_0",
                sw_sym="sw_silu_mul_seg",
                gg_weight_arg_idx=7,
                ug_weight_arg_idx=9,
                input_arg_idx=6,
                output_arg_idx=11,
                out_rows=HIDDEN_DIM,
                group_size=group_size,
                pingpong_w=PINGPONG_W_K2048,
                rms_fused=_rmsfuse,
            )
        else:
            _emit_sw_silu_mul_seg(group_size=group_size)
            _emit_awq_matvec_seg_k2048(
                "ug_awq_matvec_0", weight_arg_idx=9, input_arg_idx=6,
                output_arg_idx=10, out_rows=HIDDEN_DIM, group_size=group_size,
                pingpong_w=PINGPONG_W_K2048)
            _emit_awq_matvec_seg_k2048(
                "gg_awq_matvec_0", weight_arg_idx=7, input_arg_idx=6,
                output_arg_idx=8, out_rows=HIDDEN_DIM, group_size=group_size,
                pingpong_w=PINGPONG_W_K2048)
        # air's fold eliminates the standalone rm_rms device.
        if not _rmsfuse:
            _emit_rm_rms_seg(group_size=group_size)
        if pack_d1:
            _emit_awq_matvec_add_pack_k2048(
                "d1_og_a1_pack",
                "og_awq_matvec_0",
                "a1_eltwise_add_seg",
                weight_arg_idx=0,
                input_arg_idx=1,
                residual_arg_idx=3,
                output_arg_idx=4,
                out_rows=EMB_DIM,
                group_size=group_size,
                pingpong_w=PINGPONG_W_K2048,
            )
        else:
            _emit_eltwise_add_seg(
                "a1_eltwise_add_seg", in0_arg_idx=2, in1_arg_idx=3, out_arg_idx=4,
                group_size=group_size)
            _emit_awq_matvec_seg_k2048(
                "og_awq_matvec_0", weight_arg_idx=0, input_arg_idx=1,
                output_arg_idx=2, out_rows=EMB_DIM, group_size=group_size,
                pingpong_w=PINGPONG_W_K2048)

        if pack_mode == "none":
            dispatch_sequence = None
        else:
            seq = []
            seq += ["d1_og_a1_pack"] if pack_d1 else ["og_awq_matvec_0", "a1_eltwise_add_seg"]
            if not _rmsfuse:
                seq += ["rm_rms_seg"]
            seq += ["d3_gg_ug_sw_pack"] if pack_d3 else ["gg_awq_matvec_0", "ug_awq_matvec_0", "sw_silu_mul_seg"]
            seq += ["d4_dg_a2_pack"] if pack_d4 else ["dg_awq_matvec_0", "a2_eltwise_add_seg"]
            dispatch_sequence = tuple(seq)
        _emit_dispatcher_device(group_size=group_size,
                                dispatch_sequence=dispatch_sequence)
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs cached MLIR).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-mode", default="none")
    parser.add_argument("-o", "--output", help="Output path (default: stdout)",
                        default=None)
    args = parser.parse_args()
    text = build_o_gemv_ffn_awq_module(pack_mode=args.pack_mode)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
