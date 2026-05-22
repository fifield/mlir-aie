# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b o_gemv_ffn decode kernel.

Replaces the cached AIR-stitched ``o_gemv_ffn.npu.air.mlir`` with an
mlir-aie Python program that emits structurally equivalent
``aie/aiex``-dialect text using the dialect Python bindings directly.

The kernel fuses 8 pipeline phases into a single ELF::

    L1: O GEMV          [8,1]  wo x attn_out     -> proj        (M=2048, K=2048)
    L2: Eltwise Add 1   [8,1]  proj + x_residual -> res1        (N=2048)
    L3: RMSNorm         [1,1]  res1 x ffn_norm_w -> normed2     (N=2048)
    L4: Gate GEMV       [8,1]  wgate x normed2   -> gate        (M=8192, K=2048)
    L5: Up GEMV         [8,1]  wup x normed2     -> up          (M=8192, K=2048)
    L6: SiLU * mul      [8,1]  SiLU(gate) * up   -> swiglu      (N=8192)
    L7: Down GEMV       [8,1]  wdown x swiglu    -> down        (M=2048, K=8192)
    L8: Eltwise Add 2   [8,1]  down + res1       -> output      (N=2048)

Module layout (matches the cached reference structurally)::

    module {
      aie.device(npu2) @a2_eltwise_add_seg  { ... }   # 8 compute tiles
      aie.device(npu2) @dg_matvec_bf16_0    { ... }   # K=8192 down GEMV
      aie.device(npu2) @sw_silu_mul_seg     { ... }   # 8 compute tiles
      aie.device(npu2) @ug_matvec_bf16_0    { ... }   # 8 compute tiles
      aie.device(npu2) @gg_matvec_bf16_0    { ... }   # 8 compute tiles
      aie.device(npu2) @rm_rms_seg          { ... }   # 1 compute tile
      aie.device(npu2) @a1_eltwise_add_seg  { ... }   # 8 compute tiles
      aie.device(npu2) @og_matvec_bf16_0    { ... }   # O GEMV (8x[8,1])
      aie.device(npu2) {                              # dispatcher
        aiex.runtime_sequence @o_gemv_ffn(...) {
          # 8 aiex.configure / aiex.run blocks
        }
      }
    }

References:
  * ``reference_mlir/o_gemv_ffn.npu.air.mlir`` -- ground truth
    (7,964 lines, produced by AIR's aircc).
  * ``builders/rms_gemv_rope.py`` -- Phase 4.3 template (similar
    multi-launch decode structure).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

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

from ._emit import bf16_memref, bf16_np, o_gemv_ffn_host_arg_types


# ---------------------------------------------------------------------------
# Constants matching the cached AIR-stitched IR for Llama-3.2-1B.
# ---------------------------------------------------------------------------
EMB_DIM = 2048      # model hidden size
HIDDEN_DIM = 8192   # FFN hidden size
N_COLS = 8          # 8 compute columns in the matvec herd
K_TILE = 4          # inner K tiling factor for the K=2048 matvec
M_TILE = 8          # rows processed per K=2048 matvec call

# Down-projection (K=8192) tiling.
K_TILE_K8192 = 1    # inner K factor for the K=8192 matvec
M_TILE_K8192 = 2    # rows processed per K=8192 matvec call

# Inline-add per-tile chunk size (256 bf16 elements).
ADD_CHUNK = 256

# SwiGLU per-tile buffer size.
SWIGLU_CHUNK = 1024

# Per-segment kernel object filenames.
KO_MATVEC = "mv_pythoc.o"
KO_MATVEC_K8192 = "mv_k8192_pythoc.o"
KO_SWIGLU = "silu_and_mul_bf16.o"
KO_RMS = "rms_norm_2048_bf16.o"


# ---------------------------------------------------------------------------
# Channel-number map (verbatim from the cached IR).
# Captured by reading shim_dma_allocations in each device of the cached
# reference_mlir/o_gemv_ffn.npu.air.mlir.
# ---------------------------------------------------------------------------
_CHANNELS: Dict[str, Dict[str, object]] = {
    # Phase 1: O GEMV (out_rows=2048, mv_pythoc.o)
    "og_matvec_bf16_0": {"weight_base": 36, "out_base": 32, "input": 1},
    # Phase 2: Residual add 1 (inline 256-elt herd)
    "a1_eltwise_add_seg": {"in0": 5, "in1": 6, "out": 7},
    # Phase 3: FFN RMSNorm (single tile)
    "rm_rms_seg": {"in0": 8, "in1": 9, "out": 10},
    # Phase 4: Gate GEMV (out_rows=8192, mv_pythoc.o)
    "gg_matvec_bf16_0": {"weight_base": 39, "out_base": 38, "input": 12},
    # Phase 5: Up GEMV (out_rows=8192, mv_pythoc.o)
    "ug_matvec_bf16_0": {"weight_base": 35, "out_base": 33, "input": 17},
    # Phase 6: SwiGLU (8 tiles, 1024-elt buffers)
    "sw_silu_mul_seg": {"in0": 21, "in1": 22, "out": 23},
    # Phase 7: Down GEMV (out_rows=2048, K=8192, mv_k8192_pythoc.o)
    "dg_matvec_bf16_0": {"weight_base": 34, "out_base": 37, "input": 25},
    # Phase 8: Residual add 2 (inline 256-elt herd)
    "a2_eltwise_add_seg": {"in0": 29, "in1": 30, "out": 31},
}


# ---------------------------------------------------------------------------
# external_buffer triples emitted per device.  AIR uses these as opaque
# metadata; aiecc treats them as references.  We mirror the order/shapes
# the cached MLIR uses so the structural diff stays minimal.
# ---------------------------------------------------------------------------
def _emit_external_buffers(*shapes):
    names = ["__air_external_buffer", "__air_external_buffer_1",
             "__air_external_buffer_2"]
    for nm, shp in zip(names, shapes):
        ty = bf16_np(*shp)
        external_buffer(ty, name=nm)


# ---------------------------------------------------------------------------
# GEMV matvec segment (K=2048, mv_pythoc.o). Shared by og, gg, ug phases.
# Structurally identical to q_matvec_bf16_0 in rms_gemv_rope.py except
# out_rows can be either EMB_DIM (=2048) or HIDDEN_DIM (=8192).
# ---------------------------------------------------------------------------
def _emit_matvec_seg_k2048(sym: str, weight_arg_idx: int, input_arg_idx: int,
                           output_arg_idx: int, out_rows: int) -> None:
    """Emit a [8,1] matvec herd device with K=2048.

    ``out_rows``  -- 2048 (O proj) or 8192 (gate/up projections).
    n_outer = out_rows // 1024.  Each outer iteration delivers 1024 rows
    across the 8 columns (128 per column).
    """
    chans = _CHANNELS[sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    x_repeat_count = 31
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE * EMB_DIM            # 16_384
    weight_outer_stride = 1024 * EMB_DIM            # 2_097_152
    output_col_stride = M_TILE                       # 8
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
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=1),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffer types.
        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
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
        core_buf_x = {}
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)

        # External buffers: weight, input, output (descending shapes).
        _emit_external_buffers(
            (out_rows, EMB_DIM),
            (EMB_DIM,),
            (out_rows,),
        )

        # Declare external_funcs.
        from aie.extras import types as T
        from ml_dtypes import bfloat16 as _bf16
        fill_fn = external_func(
            "linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_MATVEC,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_MATVEC,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # Compute tile mem + core blocks (descending col).
        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            x_buf = core_buf_x[col]

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb):
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
                    with block[6]:
                        use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(ct_op, cl, y_buf, w_buf, x_buf)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE, K_TILE):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf)

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
                    dma_bd(_w, offset=0, len=M_TILE * EMB_DIM)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * EMB_DIM)
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
        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
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
# GEMV matvec segment (K=8192, mv_k8192_pythoc.o) -- Down projection.
# Compute tile: M_TILE_K8192=2 output rows per call, K_TILE_K8192=1 inner k.
# Mem tile: w_buf size = M_TILE_K8192 * HIDDEN_DIM = 16384, y size = 2.
# Input is the full 8192-elt swiglu vector, fanned out to all 8 cores.
# Output rows = EMB_DIM = 2048 across 8 outer iters
#   (each outer covers 256 rows = 8 cols * 32 elts).
# y_dims = [(16, 16), (2, 1)] writes 32 elts per task as 16 stride-16 pairs;
# 8 cols cover 256 elts per outer, output_col_stride = M_TILE_K8192 = 2.
# ---------------------------------------------------------------------------
def _emit_matvec_seg_k8192(sym: str, weight_arg_idx: int, input_arg_idx: int,
                           output_arg_idx: int) -> None:
    """K=8192 down-projection matvec [8,1] herd, mv_k8192_pythoc.o.

    Output rows: 2048 across 8 outer iters (each outer covers 256 rows =
    8 cols * 32 elts each).  Weight has same access pattern as K=2048
    case in elements (w_dims=[(16,131072),(32,512),(512,1)], len=262144),
    but offsets stride by 2*HIDDEN_DIM = 16384 per output row band.
    """
    chans = _CHANNELS[sym]
    out_rows = EMB_DIM
    # Each outer iter writes 256 rows (8 cols * 32 elts).
    n_outer = out_rows // 256  # 8

    y_dims = [(16, 16), (2, 1)]
    y_len = 32
    # Input: 8192 elements broadcast, 16 chunks of 512 per slice;
    # repeat_count = 31 = 32 deliveries.
    x_repeat_count = 31
    x_dims = [(16, 512), (512, 1)]
    x_len = HIDDEN_DIM
    # Weight: 2048x8192; outer covers 256 output rows; col stride = 32 *
    # 8192 / N_COLS? Actually the cached IR shows weight tasks per outer
    # with offsets {0, 16384, 32768, 49152, 65536, 81920, 98304, 114688}
    # (col stride 16384 = M_TILE_K8192 * HIDDEN_DIM = 2*8192). Each task
    # delivers len=262144 with dims=[(16,131072),(32,512),(512,1)] which
    # is 16 mini-rows of 32x512. 16 mini-rows * 2 rows each = 32... no.
    # The compute consumes len=16384 = M_TILE_K8192 * HIDDEN_DIM.
    # The 262144 = 16384 * 16 because the mem tile only holds 1x2x8192,
    # and the mem-tile DMA cycles 16 times per outer iter (to feed 16
    # core invocations).
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE_K8192 * HIDDEN_DIM    # 16_384
    # Outer stride = 256 rows * 8192 cols = 2_097_152
    weight_outer_stride = 256 * HIDDEN_DIM            # 2_097_152
    output_col_stride = M_TILE_K8192                  # 2
    output_outer_stride = 256

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=1),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffer types (K=8192 variant).
        _W_L1_TY = bf16_memref(K_TILE_K8192, HIDDEN_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE_K8192, HIDDEN_DIM, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE_K8192, memory_space=1)

        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        core_buf_y = {}
        core_buf_w = {}
        core_buf_x = {}
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)

        # External buffers: weight (2048x8192), input (8192), output (2048).
        _emit_external_buffers(
            (EMB_DIM, HIDDEN_DIM),
            (HIDDEN_DIM,),
            (EMB_DIM,),
        )

        from aie.extras import types as T
        from ml_dtypes import bfloat16 as _bf16
        fill_fn = external_func(
            "dg_linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_MATVEC_K8192,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "dg_matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KO_MATVEC_K8192,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            x_buf = core_buf_x[col]

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb):
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
                        dma_bd(_wb, offset=0, len=K_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(ct_op, cl, y_buf, w_buf, x_buf)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(HIDDEN_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE_K8192, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf)

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

        def _make_memtile_dma(_col, _ml, _w, _y):
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
                with block[4]:
                    use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
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
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col], mem_buf_y[col])

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

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
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
# 8 compute tiles, each holds 256-elt bf16 buffers (3 per tile).
# No mem tiles -- shim<->compute direct flows.
# ---------------------------------------------------------------------------
def _emit_eltwise_add_seg(sym: str, in0_arg_idx: int, in1_arg_idx: int,
                          out_arg_idx: int) -> None:
    """Emit an inline-add [8,1] herd device.

    Each compute tile processes ADD_CHUNK (256) elements per iteration:
    in0 + in1 -> out, vectorized as 16x16-elt addf.
    """
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # 6 locks per compute tile (ids 5..0, ascending col).
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

        # Buffers in AIR's descending col order: bufC (out), bufB (in2), bufA (in1).
        core_buf_out = {}
        core_buf_in2 = {}
        core_buf_in1 = {}
        for col in reversed(range(N_COLS)):
            core_buf_out[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in2[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in1[col] = buffer(compute_tiles[col], datatype=_BUF_TY)

        _emit_external_buffers((EMB_DIM,), (EMB_DIM,), (EMB_DIM,))

        # mem block + core block per tile (descending col).
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
                    # MM2S 0: out -> shim
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

        # Flows: shim_c DMA0 -> compute_c DMA0 (in1); shim_c DMA1 -> compute_c DMA1 (in2);
        # compute_c DMA0 -> shim_c DMA0 (out).  Per-column, no mem tile.
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

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
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
# RMSNorm segment (single compute tile, inline scalar/vector RMSNorm).
# Reads ffn_norm_w (arg5) and res1 (arg4), writes normed2 (arg6).
# Channel layout: in0=8 (weight), in1=9 (x), out=10 (y).
# Buffers in AIR order: buf67 (x, on S2MM 1), buf66 (out, on MM2S 0),
#                       buf65 (w, on S2MM 0), buf64 (scratch 16xbf16).
# ---------------------------------------------------------------------------
def _emit_rm_rms_seg() -> None:
    sym = "rm_rms_seg"
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim = tile(0, 0)
        ct = tile(0, 2)

        lk5 = lock(ct, lock_id=5, init=1)  # in2 avail (x in)
        lk4 = lock(ct, lock_id=4, init=0)  # in2 ready
        lk3 = lock(ct, lock_id=3, init=1)  # in1 avail (w in)
        lk2 = lock(ct, lock_id=2, init=0)  # in1 ready
        lk1 = lock(ct, lock_id=1, init=1)  # out done
        lk0 = lock(ct, lock_id=0, init=0)  # out full

        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        _BF16_16_L1 = bf16_memref(16, memory_space=2)
        # AIR emit order: buf67 (x), buf66 (out), buf65 (w), buf64 (scratch).
        buf_x = buffer(ct, datatype=_BF16_2048_L1)
        buf_y = buffer(ct, datatype=_BF16_2048_L1)
        buf_w = buffer(ct, datatype=_BF16_2048_L1)
        buf_s = buffer(ct, datatype=_BF16_16_L1)

        _emit_external_buffers((EMB_DIM,), (EMB_DIM,), (EMB_DIM,))

        @mem(ct)
        def _core_mem(block):
            # MM2S 0: out -> shim
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

        # Call external rms_norm_2048_bf16(weight, x, y, scratch).
        # NOTE: the cached AIR-stitched IR has the body inlined as direct
        # vector ops.  We delegate to the same `rms_norm_2048_bf16.o` that
        # the rms_gemv_rope builder uses; the inline vs out-of-line bodies
        # are semantically identical so the HF answer gate still passes,
        # at the cost of a couple of additional `func.call`s in the diff.
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
                # Core acquires complements of mem DMA blocks:
                #   mem MM2S 0 (out drain) -> id=0 acquired, id=1 released.
                #     core acquires id=1 (out_done = lk1).
                #   mem S2MM 0 (w in)      -> id=3 acquired, id=2 released.
                #     core acquires id=2 (w_ready = lk2).
                #   mem S2MM 1 (x in)      -> id=5 acquired, id=4 released.
                #     core acquires id=4 (x_ready = lk4).
                use_lock(lk1, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk2, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk4, LockAction.AcquireGreaterEqual, value=1)
                # PythoC kernel signature is (x, w, y, scratch); pass x first.
                rms_fn(buf_x, buf_w, buf_y, buf_s)
                # Releases: id=5 (x_avail), id=0 (y_full), id=3 (w_avail).
                use_lock(lk5, LockAction.Release, value=1)
                use_lock(lk0, LockAction.Release, value=1)
                use_lock(lk3, LockAction.Release, value=1)

        # Flows.
        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[5]  # ffn_norm weight
            arg_x = args[4]  # res1 input
            arg_y = args[6]  # normed2 output

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
# SwiGLU segment: [8,1] herd, link_with silu_and_mul_bf16.o.
# Each tile processes SWIGLU_CHUNK (1024) elements: silu_and_mul(in0, in1, out, 1024).
# Direct shim<->compute flows (no mem tile).
# Args: arg8 (gate), arg10 (up), arg11 (output).
# ---------------------------------------------------------------------------
def _emit_sw_silu_mul_seg() -> None:
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

        # Buffers in AIR descending order: out, in2, in1.
        core_buf_out = {}
        core_buf_in2 = {}
        core_buf_in1 = {}
        for col in reversed(range(N_COLS)):
            core_buf_out[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in2[col] = buffer(compute_tiles[col], datatype=_BUF_TY)
            core_buf_in1[col] = buffer(compute_tiles[col], datatype=_BUF_TY)

        _emit_external_buffers((HIDDEN_DIM,), (HIDDEN_DIM,), (HIDDEN_DIM,))

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

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
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
def _emit_dispatcher_device() -> None:
    """Emit the unnamed top-level dispatcher device.

    Fires the 8 segments in pipeline order:
        og -> a1 -> rm -> gg -> ug -> sw -> dg -> a2.
    All segments share the 15-arg host signature.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *o_gemv_ffn_host_arg_types(),
            sym_name="o_gemv_ffn",
        )
        def _outer(*args):
            for sym in ("og_matvec_bf16_0", "a1_eltwise_add_seg",
                        "rm_rms_seg", "gg_matvec_bf16_0",
                        "ug_matvec_bf16_0", "sw_silu_mul_seg",
                        "dg_matvec_bf16_0", "a2_eltwise_add_seg"):
                cfg = ConfigureOp(symbol=sym)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(
                        runtime_sequence_symbol=f"{sym}_sequence",
                        args=list(args),
                    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_o_gemv_ffn_module(emb_dim: int = EMB_DIM,
                            hidden_dim: int = HIDDEN_DIM) -> str:
    """Build the o_gemv_ffn ``aie/aiex``-dialect module.

    Both dimensions are fixed to the Llama-3.2-1B values (the cached AIR
    layout is shape-specialized).
    """
    if emb_dim != EMB_DIM or hidden_dim != HIDDEN_DIM:
        raise ValueError(
            f"o_gemv_ffn builder is fixed to emb_dim={EMB_DIM}, "
            f"hidden_dim={HIDDEN_DIM}; got emb_dim={emb_dim}, "
            f"hidden_dim={hidden_dim}."
        )

    with mlir_mod_ctx() as ctx:
        # AIR emit order is reverse pipeline order:
        # a2, dg, sw, ug, gg, rm, a1, og, then dispatcher.
        _emit_eltwise_add_seg(
            "a2_eltwise_add_seg", in0_arg_idx=13, in1_arg_idx=4, out_arg_idx=14)
        _emit_matvec_seg_k8192(
            "dg_matvec_bf16_0", weight_arg_idx=12, input_arg_idx=11,
            output_arg_idx=13)
        _emit_sw_silu_mul_seg()
        _emit_matvec_seg_k2048(
            "ug_matvec_bf16_0", weight_arg_idx=9, input_arg_idx=6,
            output_arg_idx=10, out_rows=HIDDEN_DIM)
        _emit_matvec_seg_k2048(
            "gg_matvec_bf16_0", weight_arg_idx=7, input_arg_idx=6,
            output_arg_idx=8, out_rows=HIDDEN_DIM)
        _emit_rm_rms_seg()
        _emit_eltwise_add_seg(
            "a1_eltwise_add_seg", in0_arg_idx=2, in1_arg_idx=3, out_arg_idx=4)
        _emit_matvec_seg_k2048(
            "og_matvec_bf16_0", weight_arg_idx=0, input_arg_idx=1,
            output_arg_idx=2, out_rows=EMB_DIM)
        _emit_dispatcher_device()
        module = ctx.module

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
    text = build_o_gemv_ffn_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
