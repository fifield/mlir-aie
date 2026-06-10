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

from typing import Dict, Sequence

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
    o_gemv_ffn_host_arg_types,
)


# ---------------------------------------------------------------------------
# Constants matching the cached AIR-stitched IR for Llama-3.2-1B.
# ---------------------------------------------------------------------------
EMB_DIM = 2048      # model hidden size
HIDDEN_DIM = 8192   # FFN hidden size
N_COLS = 8          # 8 compute columns in the matvec herd
K_TILE = 8          # inner K tiling factor for the K=2048 matvec
M_TILE = 8          # rows processed per K=2048 matvec call
# K_TILE = M_TILE => K-loop is single iter. See rms_gemv_rope.py.

# Down-projection (K=8192) tiling.
K_TILE_K8192 = 2    # inner K factor for the K=8192 matvec
M_TILE_K8192 = 2    # rows processed per K=8192 matvec call
# K_TILE_K8192 = M_TILE_K8192 => K-loop is single iter.

# Inline-add per-tile chunk size (256 bf16 elements).
ADD_CHUNK = 256

# normed2 L2-chaining (pack_mode "d1d3d4_n2l2"): pinned col-0 mem-tile address
# for the resident normed2 vector. Chosen high (256 KB) to clear D3's per-column
# footprint (~64 KB); see L2_CROSS_COLUMN_ROUTING_SCOPE.md.
_NORMED2_L2_ADDR = 0x40000

# SwiGLU per-tile buffer size.
SWIGLU_CHUNK = 1024

DEFAULT_DISPATCH_SEQUENCE = (
    "og_matvec_bf16_0",
    "a1_eltwise_add_seg",
    "rm_rms_seg",
    "gg_matvec_bf16_0",
    "ug_matvec_bf16_0",
    "sw_silu_mul_seg",
    "dg_matvec_bf16_0",
    "a2_eltwise_add_seg",
)

# Per-segment kernel object filenames.
KO_MATVEC = "mv_pythoc.ll"  # inlined (alwaysinline IR-merge)
KO_MATVEC_RMS = "matvec_rms_pythoc.ll"  # inlined (alwaysinline IR-merge)  # fused RMSNorm+matvec (air 3-device fold)
KO_MATVEC_K8192 = "mv_k8192_pythoc.o"
KO_MATVEC_FUSED = "matvec_fused_pythoc.o"  # mode-switched K2048+K8192 (proj-engine probe)
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
                           output_arg_idx: int, out_rows: int,
                           pingpong_w: bool = False) -> None:
    """Emit a [8,1] matvec herd device with K=2048.

    ``out_rows``  -- 2048 (O proj) or 8192 (gate/up projections).
    n_outer = out_rows // 1024.  Each outer iteration delivers 1024 rows
    across the 8 columns (128 per column).

    ``pingpong_w=True`` doubles the L1 W buffer (wb0+wb1) and runs the
    W DMA BD chain as a 2-BD ring; ``w_avail`` starts at init=2 so the
    L1 producer can stage two tiles ahead. K_TILE-loop is unrolled to
    2 iters (M_TILE/K_TILE=2). See rms_gemv_rope.py::_emit_matvec_seg
    for the rationale.
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
        core_buf_w1 = {}
        core_buf_x = {}
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            if pingpong_w:
                core_buf_w1[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
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
            w_buf1 = core_buf_w1.get(col)  # None unless pingpong_w
            x_buf = core_buf_x[col]

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
                            dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # Ping-pong: 2-BD ring writing wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE * EMB_DIM)
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
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        if _wb1 is None:
                            for k_idx in range_(0, M_TILE, K_TILE):
                                k_i32 = index_cast(k_idx, to=T.i32())
                                use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                                use_lock(_cl["x_avail"], LockAction.Release, value=1)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        else:
                            # Ping-pong: M_TILE/K_TILE must be 2.
                            assert M_TILE // K_TILE == 2, (
                                f"pingpong unroll assumes M_TILE/K_TILE==2, "
                                f"got {M_TILE}/{K_TILE}"
                            )
                            k_i32_0 = arith.constant(0, T.i32())
                            k_i32_1 = arith.constant(K_TILE, T.i32())
                            # K-iter 0: wb0
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            # K-iter 1: wb1
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_1, _wb1, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
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
                           output_arg_idx: int, pingpong_w: bool = False,
                           pingpong_w_l2: bool = False) -> None:
    """K=8192 down-projection matvec [8,1] herd, mv_k8192_pythoc.o.

    Output rows: 2048 across 8 outer iters (each outer covers 256 rows =
    8 cols * 32 elts each).  Weight has same access pattern as K=2048
    case in elements (w_dims=[(16,131072),(32,512),(512,1)], len=262144),
    but offsets stride by 2*HIDDEN_DIM = 16384 per output row band.

    ``pingpong_w=True`` doubles the L1 W buffer (wb0+wb1), turns the W
    DMA BD chain into a 2-BD ring (wb0/wb1 alternating), and raises
    ``w_avail`` to init=2 so the memtile->L1 producer can stage two
    tiles before any compute drains. The inner K-loop is unrolled
    (M_TILE_K8192/K_TILE_K8192=2, so 2 iters) -- iter 0 reads wb0,
    iter 1 reads wb1.

    ``pingpong_w_l2=True`` does the same one level up: doubles the
    memtile W buffer (L2 has 512 KB per tile vs L1's 64 KB, so capacity
    is never the constraint here), splits both memtile BD chains
    (S2MM ch 0 shim->L2 fill, MM2S ch 1 L2->L1 drain) into 2-BD rings,
    and raises ``w_dma_done`` to init=2. This is the natural follow-on
    to ``pingpong_w`` -- without it the L1 ping-pong saturates against
    a single-slot L2 (visible as starv1 rising on the L1 trace).
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

        # Buffer types (K=8192 variant).
        _W_L1_TY = bf16_memref(K_TILE_K8192, HIDDEN_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE_K8192, HIDDEN_DIM, memory_space=1)
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
        core_buf_w1 = {}
        core_buf_x = {}
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            if pingpong_w:
                core_buf_w1[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
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
            w_buf1 = core_buf_w1.get(col)  # None unless pingpong_w
            x_buf = core_buf_x[col]

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
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
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * HIDDEN_DIM)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # Ping-pong: 2-BD ring writing wb0 then wb1.
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * HIDDEN_DIM)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=K_TILE_K8192 * HIDDEN_DIM)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_core_mem(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb, _wb1):
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
                        if _wb1 is None:
                            for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                                k_i32 = index_cast(k_idx, to=T.i32())
                                use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                                use_lock(_cl["x_avail"], LockAction.Release, value=1)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        else:
                            # Ping-pong: M_TILE_K8192/K_TILE_K8192 must be 2.
                            assert M_TILE_K8192 // K_TILE_K8192 == 2, (
                                f"pingpong unroll assumes M/K_TILE==2, got "
                                f"{M_TILE_K8192}/{K_TILE_K8192}"
                            )
                            k_i32_0 = arith.constant(0, T.i32())
                            k_i32_1 = arith.constant(K_TILE_K8192, T.i32())
                            # K-iter 0: wb0
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            # K-iter 1: wb1
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_1, _wb1, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

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
                    # MM2S ch 1 (L2 -> L1 stream): single BD looping on itself.
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    # S2MM ch 0 (shim -> L2 fill): single BD looping on itself.
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                else:
                    # L2 ping-pong: 2-BD rings on both MM2S ch 1 (L2->L1) and
                    # S2MM ch 0 (shim->L2), alternating w0 / w1. With
                    # w_dma_done init=2, the shim can stage two L2 tiles
                    # before any are drained to L1.
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[9])
                    with block[9]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[10])
                    with block[10]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
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
# Packed matvec -> add devices (D1 and D4).
# ---------------------------------------------------------------------------
def _emit_matvec_add_pack_k2048(
    sym: str,
    matvec_sym: str,
    add_sym: str,
    *,
    weight_arg_idx: int,
    input_arg_idx: int,
    residual_arg_idx: int,
    output_arg_idx: int,
    out_rows: int = EMB_DIM,
    fused_mv: bool = False,
) -> None:
    """Pack one K=2048 matvec with its following residual add.

    The add consumes the matvec's per-column strided output partition in L2
    and writes the global output with the matvec output BD dimensions. This is
    the D1 shape: og(r2) -> a1_add(r3).

    ``fused_mv`` (proj-engine probe): run the O matvec on the SAME
    `matvec_fused_bf16` kernel as the K=8192 down matvec, selecting the
    K=2048 / loop_range-32 arm via a per-tile `mode` RTP the runtime sequence
    hard-codes to 0. This proves one ELF (matvec_fused_pythoc.o) serves both
    K-roles; bit-exact (runs the same 32-iter body as the dedicated kernel).
    """
    mat_chans = _CHANNELS[matvec_sym]
    add_chans = _CHANNELS[add_sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    post_chunk = y_len
    x_repeat_count = 31
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE * EMB_DIM
    weight_outer_stride = 1024 * EMB_DIM
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
        for col in range(N_COLS):
            mt = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(mt, lock_id=5, init=1),
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

        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
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
        mat_buf_x = {}
        add_buf_out = {}
        add_buf_res = {}
        add_buf_proj = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_res[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_proj[col] = buffer(add_tiles[col], datatype=_ADD_TY)

        _emit_external_buffers((out_rows, EMB_DIM), (EMB_DIM,), (out_rows,))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        # Proj-engine probe: same mode-switched matvec + per-tile mode RTP as
        # the K=8192 down pack, but hard-coded to 0 (K=2048 / loop_range-32).
        mat_mode_rtp = {}
        if fused_mv:
            fill_fn = external_func(
                "mvf_linalg_fill_bf16",
                inputs=[_bf16, _Y_L1_TY],
                link_with=KO_MATVEC_FUSED,
            )
            fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            matvec_fused_fn = external_func(
                "matvec_fused_bf16",
                inputs=[np.int32, np.int32, np.int32, np.int32,
                        _W_L1_TY, _X_L1_TY, _Y_L1_TY],
                link_with=KO_MATVEC_FUSED,
            )
            matvec_fused_fn.operation.attributes["llvm.emit_c_interface"] = \
                UnitAttr.get()
            for col in range(N_COLS):
                mat_mode_rtp[col] = buffer(
                    mat_tiles[col],
                    np.ndarray[(1,), np.dtype[np.int32]],
                    f"{sym}_mvmode_{col}",
                    use_write_rtp=True,
                )
        else:
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

        for col in reversed(range(N_COLS)):
            ct_op = mat_tiles[col]
            cl = mat_locks[col]
            y_buf = mat_buf_y[col]
            w_buf = mat_buf_w[col]
            x_buf = mat_buf_x[col]

            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb):
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
            _make_mat_mem(ct_op, cl, y_buf, w_buf, x_buf)

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _mode=None):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    # Read the mode RTP once (loop-invariant); the fused matvec
                    # branches on it (0 -> K=2048 / loop_range 32).
                    mode_v = _mode[0] if _mode is not None else None
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE, K_TILE):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if _mode is not None:
                                matvec_fused_fn(mode_v, k_tile_c, k_total, k_i32,
                                                _wb, _xb, _yb)
                            else:
                                matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_mat_core(ct_op, cl, y_buf, w_buf, x_buf,
                           mat_mode_rtp.get(col) if fused_mv else None)

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

        # Packet-route shim MM2S0: pkt 0 feeds matvec weights, pkt 1 feeds
        # the residual add operand. The matvec input broadcast uses shim0
        # MM2S1 as in the standalone matvec.
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
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{residual_chan}_{col}",
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
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
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

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_res = args[residual_arg_idx]
            arg_y = args[output_arg_idx]
            # Proj-engine probe: hard-code the matvec mode RTP to 0 (K=2048)
            # before any data movement, so each mat core reads mode=0 at start.
            if fused_mv:
                for col in range(N_COLS):
                    mat_mode_rtp[col][0] = 0
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
                            arg_x,
                            offset=0,
                            len=EMB_DIM,
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
                                len=y_len,
                                dimensions=y_dims,
                                packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    res_tasks.append(t)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}",
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
                for t in reversed(res_tasks):
                    dma_free_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


def _emit_matvec_add_pack_k8192(
    sym: str,
    matvec_sym: str,
    add_sym: str,
    *,
    weight_arg_idx: int,
    input_arg_idx: int,
    residual_arg_idx: int,
    output_arg_idx: int,
    fused_mv: bool = False,
) -> None:
    """Pack the K=8192 down matvec with a2_add (D4).

    ``fused_mv`` (proj-engine probe): replace the dedicated K=8192 matvec
    (`dg_matvec_vectorized_bf16_bf16`) with the mode-switched
    `matvec_fused_bf16` kernel, whose mode is read from a per-tile RTP that
    the runtime sequence hard-codes to 1 (the K=8192 / loop_range-128 arm).
    Functionally identical (runs the same 128-iter body) -- the point is to
    measure the cost of carrying both core bodies behind one mode RTP, the
    primitive the single packet-fed proj-engine needs.
    """
    mat_chans = _CHANNELS[matvec_sym]
    add_chans = _CHANNELS[add_sym]
    n_outer = EMB_DIM // 256

    y_dims = [(16, 16), (2, 1)]
    y_len = 32
    post_chunk = y_len
    x_repeat_count = 31
    x_dims = [(16, 512), (512, 1)]
    x_len = HIDDEN_DIM
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE_K8192 * HIDDEN_DIM
    weight_outer_stride = 256 * HIDDEN_DIM
    output_col_stride = M_TILE_K8192
    output_outer_stride = 256

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
        for col in range(N_COLS):
            mt = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(mt, lock_id=5, init=1),
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

        _W_L1_TY = bf16_memref(K_TILE_K8192, HIDDEN_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE_K8192, HIDDEN_DIM, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE_K8192, memory_space=1)
        _ADD_TY = bf16_memref(post_chunk, memory_space=2)

        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        mat_buf_y = {}
        mat_buf_w = {}
        mat_buf_x = {}
        add_buf_out = {}
        add_buf_res = {}
        add_buf_down = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_res[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_down[col] = buffer(add_tiles[col], datatype=_ADD_TY)

        _emit_external_buffers((EMB_DIM, HIDDEN_DIM), (HIDDEN_DIM,), (EMB_DIM,))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        # Proj-engine probe: mode-switched matvec + a per-tile `mode` RTP that
        # the runtime sequence hard-codes to 1 (K=8192 / loop_range-128 arm).
        # In fused mode the core links ONLY matvec_fused_pythoc.o (which also
        # carries its own fill), so its program-memory size is clean.
        mat_mode_rtp = {}
        if fused_mv:
            fill_fn = external_func(
                "mvf_linalg_fill_bf16",
                inputs=[_bf16, _Y_L1_TY],
                link_with=KO_MATVEC_FUSED,
            )
            fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            matvec_fused_fn = external_func(
                "matvec_fused_bf16",
                inputs=[np.int32, np.int32, np.int32, np.int32,
                        _W_L1_TY, _X_L1_TY, _Y_L1_TY],
                link_with=KO_MATVEC_FUSED,
            )
            matvec_fused_fn.operation.attributes["llvm.emit_c_interface"] = \
                UnitAttr.get()
            for col in range(N_COLS):
                mat_mode_rtp[col] = buffer(
                    mat_tiles[col],
                    np.ndarray[(1,), np.dtype[np.int32]],
                    f"{sym}_mvmode_{col}",
                    use_write_rtp=True,
                )
        else:
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
            ct_op = mat_tiles[col]
            cl = mat_locks[col]
            y_buf = mat_buf_y[col]
            w_buf = mat_buf_w[col]
            x_buf = mat_buf_x[col]

            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb):
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
            _make_mat_mem(ct_op, cl, y_buf, w_buf, x_buf)

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _mode=None):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(HIDDEN_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE_K8192, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    # Read the mode RTP once (loop-invariant); the fused matvec
                    # branches on it (1 -> K=8192 / loop_range 128).
                    mode_v = _mode[0] if _mode is not None else None
                    for _ in range_(_sys.maxsize):
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        fill_fn(zero_bf16, _yb)
                        for k_idx in range_(0, M_TILE_K8192, K_TILE_K8192):
                            k_i32 = index_cast(k_idx, to=T.i32())
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if _mode is not None:
                                matvec_fused_fn(mode_v, k_tile_c, k_total, k_i32,
                                                _wb, _xb, _yb)
                            else:
                                matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_mat_core(ct_op, cl, y_buf, w_buf, x_buf,
                           mat_mode_rtp.get(col) if fused_mv else None)

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
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{residual_chan}_{col}",
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
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )

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

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_w = args[weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_res = args[residual_arg_idx]
            arg_y = args[output_arg_idx]
            # Proj-engine probe: hard-code the matvec mode RTP to 1 (K=8192)
            # before any data movement, so each mat core reads mode=1 at start.
            if fused_mv:
                for col in range(N_COLS):
                    mat_mode_rtp[col][0] = 1
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
                            arg_x,
                            offset=0,
                            len=x_len,
                            dimensions=x_dims,
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
                                len=y_len,
                                dimensions=y_dims,
                                packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    res_tasks.append(t)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}",
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
                for t in reversed(res_tasks):
                    dma_free_task(t)
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
def _emit_rm_rms_seg(normed2_l2: bool = False) -> None:
    sym = "rm_rms_seg"
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim = tile(0, 0)
        ct = tile(0, 2)
        # normed2 L2 capture (Step A): route the rms output through col-0
        # mem-tile into a pinned resident buffer, then on to the shim/DDR. The
        # DDR copy is unchanged (D3 still reads it); the L2 copy persists for a
        # future Step B that re-sources the D3 broadcast from it.
        mt = tile(0, 1) if normed2_l2 else None
        if normed2_l2:
            n2_empty = lock(mt, lock_id=0, init=1)
            n2_full = lock(mt, lock_id=1, init=0)

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
        if normed2_l2:
            normed2_l2_buf = buffer(
                mt, datatype=bf16_memref(EMB_DIM, memory_space=1),
                address=_NORMED2_L2_ADDR)

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
        if normed2_l2:
            # normed2: ct -> mem-tile (capture into normed2_l2_buf) -> shim/DDR.
            flow(ct, WireBundle.DMA, 0, mt, WireBundle.DMA, 0)
            flow(mt, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

            @memtile_dma(mt)
            def _n2_mt(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(n2_empty, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(normed2_l2_buf, offset=0, len=EMB_DIM)
                    use_lock(n2_full, LockAction.Release, value=1)
                    next_bd(block[4])
                with block[2]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(n2_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(normed2_l2_buf, offset=0, len=EMB_DIM)
                    use_lock(n2_empty, LockAction.Release, value=1)
                    next_bd(block[4])
                with block[4]:
                    EndOp()
        else:
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
# Packed gate/up matvec -> SwiGLU device (D3).
# ---------------------------------------------------------------------------
def _emit_gg_ug_swiglu_pack(
    sym: str = "d3_gg_ug_sw_pack",
    *,
    gg_sym: str = "gg_matvec_bf16_0",
    ug_sym: str = "ug_matvec_bf16_0",
    sw_sym: str = "sw_silu_mul_seg",
    gg_weight_arg_idx: int = 7,
    ug_weight_arg_idx: int = 9,
    input_arg_idx: int = 6,
    output_arg_idx: int = 11,
    out_rows: int = HIDDEN_DIM,
    normed2_l2: bool = False,
    rms_fused: bool = False,
    result_pkt: bool = False,
    fused_mv: bool = False,
    res1_arg_idx: int = 4,
    normw_arg_idx: int = 5,
) -> None:
    """Pack gate/up K=2048 matvecs with the following SwiGLU.

    Gate runs on row 2, up runs on row 3, and SwiGLU runs on row 4. The
    post-op consumes the matvec per-column output stream directly in 128-elt
    chunks and writes the global hidden vector using the matvec output layout.

    ``result_pkt`` (proj-engine step 1a) routes the SwiGLU result stream
    (sw row 4 -> shim) over a single-ID ``packetflow`` instead of a
    circuit-switched ``flow``, with the SAME destination. This is a pure
    structural convergence: the packet header is stripped at the destination
    port, so lengths and data are unchanged and the HF gate stays bit-exact.
    It proves the compute-tile-sourced packet-result primitive that the
    collapsed engine needs to demux O/gate/up/down outputs by packet ID.
    """
    # Single result packet ID for step 1a (same dest, bit-exact). Later
    # sub-steps vary this ID per projection role to demux the engine output.
    _RESULT_PKT_ID = 0
    gg_chans = _CHANNELS[gg_sym]
    ug_chans = _CHANNELS[ug_sym]
    sw_chans = _CHANNELS[sw_sym]
    assert out_rows % 1024 == 0, "out_rows must be multiple of 1024"
    n_outer = out_rows // 1024

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    x_repeat_count = 31
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE * EMB_DIM
    weight_outer_stride = 1024 * EMB_DIM
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
        for col in range(N_COLS):
            gt = gg_tiles[col]
            gg_locks[col] = {
                "w_avail": lock(gt, lock_id=5, init=1),
                "w_ready": lock(gt, lock_id=4, init=0),
                "x_avail": lock(gt, lock_id=3, init=1),
                "x_ready": lock(gt, lock_id=2, init=0),
                "y_done":  lock(gt, lock_id=1, init=1),
                "y_full":  lock(gt, lock_id=0, init=0),
            }
            ut = ug_tiles[col]
            ug_locks[col] = {
                "w_avail": lock(ut, lock_id=5, init=1),
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

        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        # Fused RMS: the gate/up activation buffer is packed [res1 | norm_w]
        # (2*EMB_DIM); else it's the single normed2 vector (EMB_DIM).
        _X_LEN = 2 * EMB_DIM if rms_fused else EMB_DIM
        _X_L1_TY = bf16_memref(_X_LEN, memory_space=2)
        _NORMED_L1_TY = bf16_memref(EMB_DIM, memory_space=2)   # rms scratch
        _RSCR_L1_TY = bf16_memref(16, memory_space=2)          # reduction spill
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
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

        # normed2 L2 chaining (Step B): col-0 mem-tile holds the resident
        # normed2 vector (written by rm; persists across the swap) and
        # broadcasts it to gg/ug matvec inputs, replacing the shim DDR read.
        # Repeat the broadcast to match the shim's total deliveries
        # (n_outer per-outer tasks x (x_repeat_count+1) each).
        normed2_l2_buf = None
        # Empirical (2026-05-29): the matvec consumes ~256 x-deliveries/col
        # (matching the shim's n_outer*(x_repeat_count+1)=8*32). repeat=255 runs
        # (does NOT deadlock) but races (non-deterministic garbage) because this
        # broadcast is lock-free; repeat=127 (128) DEADLOCKS (starved). So 255 is
        # the correct COUNT -- the open problem is SYNCHRONIZATION (the lock-free
        # broadcast races the per-group consumption; the shim avoided this via
        # its per-outer await/free pacing). TODO: lock-synchronize the broadcast.
        n2_bcast_repeat = n_outer * (x_repeat_count + 1) - 1  # 255
        if normed2_l2:
            normed2_l2_buf = buffer(
                mem_tiles[0], datatype=bf16_memref(EMB_DIM, memory_space=1),
                address=_NORMED2_L2_ADDR)

        gg_buf_y = {}
        gg_buf_w = {}
        gg_buf_x = {}
        gg_buf_normed = {}
        gg_buf_rscr = {}
        ug_buf_y = {}
        ug_buf_w = {}
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
            sw_buf_out[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_up[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_gate[col] = buffer(sw_tiles[col], datatype=_SW_TY)

        _emit_external_buffers((out_rows, EMB_DIM), (EMB_DIM,), (out_rows,))

        from aie.extras import types as T
        from ml_dtypes import bfloat16 as _bf16

        # Plain fill + matvec (bit-exact baseline kernels). For rms_fused the
        # matvec reads the resident `normed` vector (computed once per token by
        # rms_fn below) instead of the raw activation -- so its activation
        # operand type is _NORMED_L1_TY (== EMB_DIM, same as the non-fused X).
        _mv_act_ty = _NORMED_L1_TY if rms_fused else _X_L1_TY
        fill_fn = external_func(
            "linalg_fill_bf16",
            inputs=[_bf16, _Y_L1_TY],
            link_with=KO_MATVEC,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _mv_act_ty, _Y_L1_TY],
            link_with=KO_MATVEC,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        # Proj-engine: route the gate/up matvec through the SAME mode-switched
        # matvec_fused_bf16 as O (D1) and down (D4), mode 0 (K=2048 / lr 32),
        # so all four projections run on one kernel. RMS prologue is unchanged.
        gg_mode_rtp = {}
        ug_mode_rtp = {}
        if fused_mv:
            matvec_fused_fn = external_func(
                "matvec_fused_bf16",
                inputs=[np.int32, np.int32, np.int32, np.int32,
                        _W_L1_TY, _mv_act_ty, _Y_L1_TY],
                link_with=KO_MATVEC_FUSED,
            )
            matvec_fused_fn.operation.attributes["llvm.emit_c_interface"] = \
                UnitAttr.get()
            for col in range(N_COLS):
                gg_mode_rtp[col] = buffer(
                    gg_tiles[col],
                    np.ndarray[(1,), np.dtype[np.int32]],
                    f"{sym}_gg_mvmode_{col}",
                    use_write_rtp=True,
                )
                ug_mode_rtp[col] = buffer(
                    ug_tiles[col],
                    np.ndarray[(1,), np.dtype[np.int32]],
                    f"{sym}_ug_mvmode_{col}",
                    use_write_rtp=True,
                )
        if rms_fused:
            # Packed [2,K] RMSNorm prologue, run ONCE per token per tile.
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
            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb):
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
                    with block[6]:
                        use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_mat_mem(
                gg_tiles[col], gg_locks[col], gg_buf_y[col], gg_buf_w[col], gg_buf_x[col]
            )
            _make_mat_mem(
                ug_tiles[col], ug_locks[col], ug_buf_y[col], ug_buf_w[col], ug_buf_x[col]
            )

            # Chunks (8-row matvec calls) per tile per token.
            _N_CHUNKS = out_rows // N_COLS // M_TILE

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb, _normed=None, _rscr=None,
                               _mode=None):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(EMB_DIM, T.i32())
                    k_tile_c = arith.constant(K_TILE, T.i32())
                    zero_off = arith.constant(0, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    # Proj-engine: read the mode RTP once (0 -> K=2048 / lr 32).
                    mode_v = _mode[0] if _mode is not None else None
                    if rms_fused:
                        # air's fold: compute the RMSNorm ONCE per token into the
                        # resident `normed` buffer, then matvec all output chunks
                        # over it (re-acquiring only weights/outputs per chunk).
                        for _ in range_(_sys.maxsize):
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            rms_fn(_xb, _normed, _rscr)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            for _c in range_(_N_CHUNKS):
                                use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                                fill_fn(zero_bf16, _yb)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                if _mode is not None:
                                    matvec_fused_fn(mode_v, k_tile_c, k_total,
                                                    zero_off, _wb, _normed, _yb)
                                else:
                                    matvec_fn(k_tile_c, k_total, zero_off, _wb, _normed, _yb)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                                use_lock(_cl["y_full"], LockAction.Release, value=1)
                    else:
                        for _ in range_(_sys.maxsize):
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            for k_idx in range_(0, M_TILE, K_TILE):
                                k_i32 = index_cast(k_idx, to=T.i32())
                                use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                                use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                                if _mode is not None:
                                    matvec_fused_fn(mode_v, k_tile_c, k_total,
                                                    k_i32, _wb, _xb, _yb)
                                else:
                                    matvec_fn(k_tile_c, k_total, k_i32, _wb, _xb, _yb)
                                use_lock(_cl["x_avail"], LockAction.Release, value=1)
                                use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_mat_core(
                gg_tiles[col], gg_locks[col], gg_buf_y[col], gg_buf_w[col], gg_buf_x[col],
                gg_buf_normed.get(col), gg_buf_rscr.get(col),
                gg_mode_rtp.get(col) if fused_mv else None,
            )
            _make_mat_core(
                ug_tiles[col], ug_locks[col], ug_buf_y[col], ug_buf_w[col], ug_buf_x[col],
                ug_buf_normed.get(col), ug_buf_rscr.get(col),
                ug_mode_rtp.get(col) if fused_mv else None,
            )

            def _make_sw_mem(_ct, _cl, _bo, _bup, _bgate):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        if result_pkt:
                            # Step 1a: emit the result over a packet route.
                            dma_bd(_bo, offset=0, len=post_chunk,
                                   packet=(0, _RESULT_PKT_ID))
                        else:
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
            if normed2_l2:
                # gg/ug input broadcast from col-0 L2 (mem-tile lateral) instead
                # of the shim DDR read.
                flow(mem_tiles[0], WireBundle.DMA, 4, gg_tiles[col], WireBundle.DMA, 0)
                flow(mem_tiles[0], WireBundle.DMA, 5, ug_tiles[col], WireBundle.DMA, 0)
            else:
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
            if result_pkt:
                # Step 1a: result path sw(row4) -> shim over a single-ID
                # packetflow (same dest as the circuit-switched flow).
                packetflow(
                    pkt_id=_RESULT_PKT_ID,
                    source=sw_tiles[col],
                    source_port=WireBundle.DMA,
                    source_channel=0,
                    dests={"dest": shim_tiles[col], "port": WireBundle.DMA,
                           "channel": 0},
                )
            else:
                flow(sw_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        gg_weight_base = gg_chans["weight_base"]
        ug_weight_base = ug_chans["weight_base"]
        gg_input_chan = gg_chans["input"]
        ug_input_chan = ug_chans["input"]
        out_chan = sw_chans["out"]
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{gg_weight_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{ug_weight_base}_{col}",
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        if not normed2_l2:
            shim_dma_allocation(
                f"air_channel_{gg_input_chan}",
                shim_tiles[0],
                DMAChannelDir.MM2S,
                1,
            )
            shim_dma_allocation(
                f"air_channel_{ug_input_chan}",
                shim_tiles[1],
                DMAChannelDir.MM2S,
                1,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{out_chan}_{col}",
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )

        def _make_memtile_dma(_col, _ml, _gg_w, _ug_w, _gg_y, _ug_y, _bcast=None):
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
                    dma_bd(_gg_w, offset=0, len=M_TILE * EMB_DIM)
                    use_lock(_ml["gg_w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.MM2S, 2, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["ug_w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_w, offset=0, len=M_TILE * EMB_DIM)
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
                    dma_bd(_gg_w, offset=0, len=M_TILE * EMB_DIM)
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
                    dma_bd(_ug_w, offset=0, len=M_TILE * EMB_DIM)
                    use_lock(_ml["ug_w_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[16],
                              chain=(block[17] if _bcast is not None else block[2]))
                with block[16]:
                    use_lock(_ml["ug_y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_ug_y, offset=0, len=M_TILE)
                    use_lock(_ml["ug_y_ready"], LockAction.Release, value=1)
                    next_bd(block[16])
                if _bcast is not None:
                    # MM2S 4/5: lock-free broadcast of the resident normed2_l2
                    # to gg/ug matvec inputs, repeated to match the shim total.
                    # No lock: the data is resident from rm (prior PDI) and not
                    # written in this device; the matvec paces via its own x
                    # locks (stream backpressure).
                    # AIE2 mem-tile BD/channel rule (xaie_dma_aieml.c): even
                    # channels use bd_id 0..23, odd channels 24..47. Pin
                    # top-of-pool ids so MM2S 4 (even) / 5 (odd) are valid and
                    # don't collide with the gg/ug chains' auto-assigned low ids.
                    with block[17]:
                        dma_start(DMAChannelDir.MM2S, 4, dest=block[18],
                                  chain=block[19], repeat_count=n2_bcast_repeat)
                    with block[18]:
                        dma_bd(_bcast, offset=0, len=EMB_DIM,
                               dimensions=[(4, 512), (512, 1)], bd_id=23)
                        next_bd(block[2])
                    with block[19]:
                        dma_start(DMAChannelDir.MM2S, 5, dest=block[20],
                                  chain=block[2], repeat_count=n2_bcast_repeat)
                    with block[20]:
                        dma_bd(_bcast, offset=0, len=EMB_DIM,
                               dimensions=[(4, 512), (512, 1)], bd_id=47)
                        next_bd(block[2])
        for col in range(N_COLS):
            _make_memtile_dma(
                col,
                mem_locks[col],
                mem_buf_gg_w[col],
                mem_buf_ug_w[col],
                mem_buf_gg_y[col],
                mem_buf_ug_y[col],
                normed2_l2_buf if (normed2_l2 and col == 0) else None,
            )

        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_gg_w = args[gg_weight_arg_idx]
            arg_ug_w = args[ug_weight_arg_idx]
            arg_x = args[input_arg_idx]
            arg_y = args[output_arg_idx]
            # Fused RMS: broadcast [res1 | ffn_norm_w] (args 4,5) as a 2-BD
            # chain on the existing input channel instead of the single normed2.
            arg_res1 = args[res1_arg_idx]
            arg_normw = args[normw_arg_idx]
            # Proj-engine: gate/up matvec mode RTP = 0 (K=2048) before any DMA.
            if fused_mv:
                for col in range(N_COLS):
                    gg_mode_rtp[col][0] = 0
                    ug_mode_rtp[col][0] = 0

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
            # Fused RMS: res1+ffn_norm_w are constant for the whole kernel, and
            # the gate/up core computes normed ONCE per token -- so deliver the
            # packed [res1|norm_w] a single time (no per-outer repeat).
            rms_gg_x_task = rms_ug_x_task = None
            if rms_fused and not normed2_l2:
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
                                len=w_len,
                                dimensions=w_dims,
                                packet=(0, 0),
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
                                len=w_len,
                                dimensions=w_dims,
                                packet=(0, 1),
                            )
                            EndOp()
                    dma_start_task(t)
                    ug_weight_tasks.append(t)

                gg_x_task = ug_x_task = None
                if not normed2_l2 and not rms_fused:
                    gg_x_task = dma_configure_task_for(
                        f"air_channel_{gg_input_chan}",
                        repeat_count=x_repeat_count,
                    )
                    _emit_x_bds(gg_x_task)
                    dma_start_task(gg_x_task)

                    ug_x_task = dma_configure_task_for(
                        f"air_channel_{ug_input_chan}",
                        repeat_count=x_repeat_count,
                    )
                    _emit_x_bds(ug_x_task)
                    dma_start_task(ug_x_task)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(
                        f"air_channel_{out_chan}_{col}",
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
                if not normed2_l2 and not rms_fused:
                    dma_free_task(ug_x_task)
                    dma_free_task(gg_x_task)
                for t in reversed(ug_weight_tasks):
                    dma_free_task(t)
                for t in reversed(gg_weight_tasks):
                    dma_free_task(t)

            # Free the once-per-token fused input tasks after all outer iters.
            if rms_gg_x_task is not None:
                dma_free_task(rms_ug_x_task)
                dma_free_task(rms_gg_x_task)


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
# C1 merged device: D1+D3 (+rms) in ONE aie.device, sequential waves.
# ---------------------------------------------------------------------------
def _emit_call2_merged(sym: str = "c1_merged") -> None:
    """C1 (RESIDENT_DEVICE_EVOLUTION.md collapse plan): merge D1+D3 -> one device.

    One ``aie.device``/configure runs O, add1, rms, gate, up, swiglu as six
    SEQUENTIAL runtime stages; D4 (down+add2) stays a separate device. All
    intermediates keep the DDR handoff (safety net), so D4 and the host args
    are untouched (3 -> 2 LoadPDIs for call 2).

    Tile map (4 compute rows -- balanced, not tile-starved):

        row 2: matvec herd (8 col)  reused O -> gate -> up (3 waves, K=2048)
        row 3: add herd (8 col)     add1 (once)
        row 4: swiglu herd (8 col)  swiglu (once)
        row 5: rms (col 0)          ffn RMSNorm (once)

    Stages are sequential so the SAME shim channels are reused, demuxed by
    packet ID (per col: MM2S0 / MM2S1 / S2MM0):

        MM2S0[c]: pkt1 -> mem[c]   (matvec W)   pkt8 -> add[c] in0
                  pkt12 -> sw[c] gate           col0: pkt13 -> rms w
        MM2S1[c]: pkt1 -> mat[*] x (col0 broadcast)  pkt8 -> add[c] in1
                  pkt12 -> sw[c] up              col0: pkt13 -> rms x
        S2MM0[c]: pkt1 <- everything (no demux needed at the shim)

    ID scheme is mask-constrained: ids passing the same slave port get merged
    into one masked rule, so chained pass-throughs (rows 3/4/5) must mask
    EXACTLY at every hop without capturing the matvec id 1. {8,12,13} group
    as 8..15 at row1, {12,13} masks exactly at row3, 13 exact at row4. All
    OUTPUT streams converge to shim S2MM0 with one id (no demux), so they
    share id 1 from the producer BDs.
    Input pkt IDs are set per stage on the shim BDs;
    input pkt IDs are set per stage on the shim BDs. The matvec mem<->core
    hops stay circuit flows. Core/mem/lock bodies are copied verbatim from
    the standalone segment emitters; only the tile row changes.
    """
    # Device-local shim channel names (fresh number range, no _CHANNELS entry).
    W_CH, A0_CH, SG_CH, RW_CH = 60, 61, 62, 63          # MM2S 0 demux
    X_CH, A1_CH, SU_CH, RX_CH = 64, 65, 66, 67          # MM2S 1 demux
    YO_CH, AO_CH, SO_CH, RO_CH = 68, 69, 70, 71         # S2MM 0 demux

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    # EXACT x delivery: 16 chunks/col/outer = 16 broadcasts per outer
    # (the standalone seg overdelivers 32; harmless there, but leftovers jam
    # the shared MM2S1 queue here, deadlocking stage 2+).
    x_repeat_count = 15
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE * EMB_DIM
    weight_outer_stride = 1024 * EMB_DIM
    output_col_stride = M_TILE
    output_outer_stride = 1024

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        mat_tiles = [tile(c, 2) for c in range(N_COLS)]
        add_tiles = [tile(c, 3) for c in range(N_COLS)]
        sw_tiles = [tile(c, 4) for c in range(N_COLS)]
        rms_tile = tile(0, 5)

        # --- locks (same ids as the standalone emitters, per tile) ---
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
        sw_locks = {}
        for col in range(N_COLS):
            t = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(t, lock_id=5, init=1),
                "w_ready": lock(t, lock_id=4, init=0),
                "x_avail": lock(t, lock_id=3, init=1),
                "x_ready": lock(t, lock_id=2, init=0),
                "y_done":  lock(t, lock_id=1, init=1),
                "y_full":  lock(t, lock_id=0, init=0),
            }
            t = add_tiles[col]
            add_locks[col] = {
                "in2_avail": lock(t, lock_id=5, init=1),
                "in2_ready": lock(t, lock_id=4, init=0),
                "in1_avail": lock(t, lock_id=3, init=1),
                "in1_ready": lock(t, lock_id=2, init=0),
                "out_done":  lock(t, lock_id=1, init=1),
                "out_full":  lock(t, lock_id=0, init=0),
            }
            t = sw_tiles[col]
            sw_locks[col] = {
                "in2_avail": lock(t, lock_id=5, init=1),
                "in2_ready": lock(t, lock_id=4, init=0),
                "in1_avail": lock(t, lock_id=3, init=1),
                "in1_ready": lock(t, lock_id=2, init=0),
                "out_done":  lock(t, lock_id=1, init=1),
                "out_full":  lock(t, lock_id=0, init=0),
            }
        rms_lk5 = lock(rms_tile, lock_id=5, init=1)  # x avail
        rms_lk4 = lock(rms_tile, lock_id=4, init=0)  # x ready
        rms_lk3 = lock(rms_tile, lock_id=3, init=1)  # w avail
        rms_lk2 = lock(rms_tile, lock_id=2, init=0)  # w ready
        rms_lk1 = lock(rms_tile, lock_id=1, init=1)  # out done
        rms_lk0 = lock(rms_tile, lock_id=0, init=0)  # out full

        # --- buffers ---
        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)
        _ADD_TY = bf16_memref(ADD_CHUNK, memory_space=2)
        _SW_TY = bf16_memref(SWIGLU_CHUNK, memory_space=2)
        _RMS_TY = bf16_memref(EMB_DIM, memory_space=2)
        _RMS_SCR_TY = bf16_memref(16, memory_space=2)

        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)
        mat_buf_y = {}
        mat_buf_w = {}
        mat_buf_x = {}
        add_buf_out = {}
        add_buf_in2 = {}
        add_buf_in1 = {}
        sw_buf_out = {}
        sw_buf_in2 = {}
        sw_buf_in1 = {}
        for col in reversed(range(N_COLS)):
            mat_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            mat_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            mat_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)
            add_buf_out[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_in2[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            add_buf_in1[col] = buffer(add_tiles[col], datatype=_ADD_TY)
            sw_buf_out[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_in2[col] = buffer(sw_tiles[col], datatype=_SW_TY)
            sw_buf_in1[col] = buffer(sw_tiles[col], datatype=_SW_TY)
        rms_buf_x = buffer(rms_tile, datatype=_RMS_TY)
        rms_buf_y = buffer(rms_tile, datatype=_RMS_TY)
        rms_buf_w = buffer(rms_tile, datatype=_RMS_TY)
        rms_buf_s = buffer(rms_tile, datatype=_RMS_SCR_TY)

        _emit_external_buffers((HIDDEN_DIM, EMB_DIM), (EMB_DIM,), (HIDDEN_DIM,))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
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
        silu_fn = external_func(
            "silu_and_mul_bf16",
            inputs=[_SW_TY, _SW_TY, _SW_TY, np.int32],
            link_with=KO_SWIGLU,
        )
        silu_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        rms_fn = external_func(
            "rms_norm_2048_bf16",
            inputs=[_RMS_TY, _RMS_TY, _RMS_TY, _RMS_SCR_TY],
            link_with=KO_RMS,
        )
        rms_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # --- matvec row 2: mem + core verbatim from _emit_matvec_seg_k2048 ---
        for col in reversed(range(N_COLS)):
            def _make_mat_mem(_ct, _cl, _yb, _wb, _xb):
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
            _make_mat_mem(mat_tiles[col], mat_locks[col], mat_buf_y[col],
                          mat_buf_w[col], mat_buf_x[col])

            def _make_mat_core(_ct, _cl, _yb, _wb, _xb):
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
            _make_mat_core(mat_tiles[col], mat_locks[col], mat_buf_y[col],
                           mat_buf_w[col], mat_buf_x[col])

        # --- add row 3: verbatim from _emit_eltwise_add_seg (row moved) ---
        for col in reversed(range(N_COLS)):
            def _make_add_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=ADD_CHUNK, packet=(0, 1))
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

            def _make_add_core(_ct, _cl, _bo, _b2, _b1):
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
            _make_add_core(add_tiles[col], add_locks[col], add_buf_out[col],
                           add_buf_in2[col], add_buf_in1[col])

        # --- swiglu row 4: verbatim from _emit_sw_silu_mul_seg (row moved) ---
        for col in reversed(range(N_COLS)):
            def _make_sw_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_bo, offset=0, len=SWIGLU_CHUNK, packet=(0, 1))
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

        # --- rms row 5 col 0: verbatim from _emit_rm_rms_seg (row moved) ---
        @mem(rms_tile)
        def _rms_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(rms_lk0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(rms_buf_y, offset=0, len=EMB_DIM, packet=(0, 1))
                use_lock(rms_lk1, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                EndOp()
            with block[3]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
            with block[4]:
                use_lock(rms_lk3, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(rms_buf_w, offset=0, len=EMB_DIM)
                use_lock(rms_lk2, LockAction.Release, value=1)
                next_bd(block[4])
            with block[5]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
            with block[6]:
                use_lock(rms_lk5, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(rms_buf_x, offset=0, len=EMB_DIM)
                use_lock(rms_lk4, LockAction.Release, value=1)
                next_bd(block[6])

        import sys as _sys

        @core(rms_tile)
        def _rms_core():
            for _ in range_(_sys.maxsize):
                use_lock(rms_lk1, LockAction.AcquireGreaterEqual, value=1)
                use_lock(rms_lk2, LockAction.AcquireGreaterEqual, value=1)
                use_lock(rms_lk4, LockAction.AcquireGreaterEqual, value=1)
                rms_fn(rms_buf_x, rms_buf_w, rms_buf_y, rms_buf_s)
                use_lock(rms_lk5, LockAction.Release, value=1)
                use_lock(rms_lk0, LockAction.Release, value=1)
                use_lock(rms_lk3, LockAction.Release, value=1)

        # --- routing ---
        # MM2S0 demux (weights / add in0 / sw gate / rms w).
        for col in range(N_COLS):
            packetflow(
                pkt_id=1,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=8,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=12,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": sw_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
        packetflow(
            pkt_id=13,
            source=shim_tiles[0], source_port=WireBundle.DMA, source_channel=0,
            dests={"dest": rms_tile, "port": WireBundle.DMA, "channel": 0},
        )
        # MM2S1 demux (x broadcast / add in1 / sw up / rms x).
        packetflow(
            pkt_id=1,
            source=shim_tiles[0], source_port=WireBundle.DMA, source_channel=1,
            dests=[{"dest": mat_tiles[c], "port": WireBundle.DMA, "channel": 0}
                   for c in range(N_COLS)],
        )
        for col in range(N_COLS):
            packetflow(
                pkt_id=8,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=1,
                dests={"dest": add_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
            packetflow(
                pkt_id=12,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=1,
                dests={"dest": sw_tiles[col], "port": WireBundle.DMA, "channel": 1},
            )
        packetflow(
            pkt_id=13,
            source=shim_tiles[0], source_port=WireBundle.DMA, source_channel=1,
            dests={"dest": rms_tile, "port": WireBundle.DMA, "channel": 1},
        )
        # matvec internal hops stay circuit-switched.
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
        # S2MM0 mux (results back to shim, pkt id from producer BD).
        for col in range(N_COLS):
            packetflow(
                pkt_id=1,
                source=mem_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=1,
                source=add_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
            packetflow(
                pkt_id=1,
                source=sw_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": shim_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
        packetflow(
            pkt_id=1,
            source=rms_tile, source_port=WireBundle.DMA, source_channel=0,
            dests={"dest": shim_tiles[0], "port": WireBundle.DMA, "channel": 0},
        )

        # --- shim DMA allocations (multiple names share a physical channel) ---
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
        shim_dma_allocation(f"air_channel_{X_CH}",
                            shim_tiles[0], DMAChannelDir.MM2S, 1)
        shim_dma_allocation(f"air_channel_{RW_CH}",
                            shim_tiles[0], DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{RX_CH}",
                            shim_tiles[0], DMAChannelDir.MM2S, 1)
        shim_dma_allocation(f"air_channel_{RO_CH}",
                            shim_tiles[0], DMAChannelDir.S2MM, 0)

        # --- mem tile DMAs: verbatim, plus pkt0 on the y -> shim BD ---
        def _make_memtile_dma(_col, _ml, _w, _y):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
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

        # --- runtime sequence: six sequential waves, DDR between stages ---
        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            def _mat_wave(arg_w, arg_x, arg_y, out_rows):
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
                                    len=w_len,
                                    dimensions=w_dims,
                                    packet=(0, 1),
                                )
                                EndOp()
                        dma_start_task(t)
                        weight_tasks.append(t)
                    x_task = dma_configure_task_for(
                        f"air_channel_{X_CH}", repeat_count=x_repeat_count)
                    with bds(x_task) as bd:
                        with bd[0]:
                            dma_bd(arg_x, offset=0, len=EMB_DIM,
                                   dimensions=[(4, 512), (512, 1)],
                                   packet=(0, 1))
                            EndOp()
                    dma_start_task(x_task)
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

            # Debug knob (wedge bisect): number of stages to emit, 1..6.
            import os as _os
            _n_stages = int(_os.environ.get("PYTHOC_C1_STAGES", "6"))
            # 1: O proj    wo x attn_out -> proj
            _mat_wave(args[0], args[1], args[2], EMB_DIM)
            if _n_stages < 2:
                return
            # 2: add1      proj + x_resid -> res1
            _eltwise_wave(A0_CH, A1_CH, AO_CH, args[2], args[3], args[4],
                          ADD_CHUNK, [(ADD_CHUNK, 1)], 8)
            if _n_stages < 3:
                return
            # 3: rms       res1 x ffn_norm_w -> normed2
            t_w = dma_configure_task_for(f"air_channel_{RW_CH}")
            with bds(t_w) as bd:
                with bd[0]:
                    dma_bd(args[5], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)], packet=(0, 13))
                    EndOp()
            dma_start_task(t_w)
            t_x = dma_configure_task_for(f"air_channel_{RX_CH}")
            with bds(t_x) as bd:
                with bd[0]:
                    dma_bd(args[4], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)], packet=(0, 13))
                    EndOp()
            dma_start_task(t_x)
            t_y = dma_configure_task_for(f"air_channel_{RO_CH}", issue_token=True)
            with bds(t_y) as bd:
                with bd[0]:
                    dma_bd(args[6], offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_y)
            dma_await_task(t_y)
            dma_free_task(t_x)
            dma_free_task(t_w)
            if _n_stages < 4:
                return
            # 4/5: gate, up   wgate/wup x normed2 -> gate/up
            _mat_wave(args[7], args[6], args[8], HIDDEN_DIM)
            if _n_stages < 5:
                return
            _mat_wave(args[9], args[6], args[10], HIDDEN_DIM)
            if _n_stages < 6:
                return
            # 6: swiglu    SiLU(gate) * up -> swiglu
            _eltwise_wave(SG_CH, SU_CH, SO_CH, args[8], args[10], args[11],
                          SWIGLU_CHUNK, [(2, 512), (512, 1)], 12)


# ---------------------------------------------------------------------------
# C2 merged device: C1 + RMS folded into gate/up waves (+ optional down/add2).
# ---------------------------------------------------------------------------
def _emit_call2_c2(sym: str, with_down: bool) -> None:
    """C2 (collapse plan): the C1 merged device, evolved per the C2 row map.

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
    W_CH, A0_CH, SG_CH = 60, 61, 62                     # MM2S 0 demux
    X_CH, A1_CH, SU_CH = 64, 65, 66                     # MM2S 1 demux
    YO_CH, AO_CH, SO_CH = 68, 69, 70                    # S2MM 0 mux
    DW_CH, DX_CH, DO_CH = 72, 73, 74                    # down (with_down)

    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    weight_col_stride = M_TILE * EMB_DIM
    weight_outer_stride = 1024 * EMB_DIM
    output_col_stride = M_TILE
    output_outer_stride = 1024
    # down (K=8192) geometry, verbatim from _emit_matvec_add_pack_k8192
    d_n_outer = EMB_DIM // 256
    d_y_dims = [(16, 16), (2, 1)]
    d_y_len = 32
    d_w_col_stride = M_TILE_K8192 * HIDDEN_DIM
    d_w_outer_stride = 256 * HIDDEN_DIM

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

        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }
            if with_down:
                mem_locks[col].update({
                    "dw_dma_done": lock(mt, lock_id=7, init=1),
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

        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _XP_L1_TY = bf16_memref(2 * EMB_DIM, memory_space=2)   # [res1|norm_w]
        _NORMED_TY = bf16_memref(EMB_DIM, memory_space=2)
        _RSCR_TY = bf16_memref(16, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)
        _ADD_TY = bf16_memref(ADD_CHUNK, memory_space=2)
        _SW_TY = bf16_memref(SWIGLU_CHUNK, memory_space=2)
        _DW_L1_TY = bf16_memref(K_TILE_K8192, HIDDEN_DIM, memory_space=2)
        _DX_L1_TY = bf16_memref(HIDDEN_DIM, memory_space=2)
        _DY_L1_TY = bf16_memref(M_TILE_K8192, memory_space=2)
        _DW_L2_TY = bf16_memref(1, M_TILE_K8192, HIDDEN_DIM, memory_space=1)
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
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)
        if with_down:
            for col in reversed(range(N_COLS)):
                mem_buf_dw[col] = buffer(mem_tiles[col], datatype=_DW_L2_TY)
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

        _emit_external_buffers((HIDDEN_DIM, EMB_DIM), (EMB_DIM,), (HIDDEN_DIM,))

        from aie.dialects import memref, vector
        from aie.extras import types as T
        from aie.ir import AffineDimExpr, AffineMap
        from ml_dtypes import bfloat16 as _bf16

        fill_fn = external_func(
            "linalg_fill_bf16", inputs=[_bf16, _Y_L1_TY], link_with=KO_MATVEC)
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _NORMED_TY, _Y_L1_TY],
            link_with=KO_MATVEC)
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
                "dg_linalg_fill_bf16", inputs=[_bf16, _DY_L1_TY],
                link_with=KO_MATVEC_K8192)
            dn_fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            dn_matvec_fn = external_func(
                "dg_matvec_vectorized_bf16_bf16",
                inputs=[np.int32, np.int32, np.int32, _DW_L1_TY, _DX_L1_TY,
                        _DY_L1_TY],
                link_with=KO_MATVEC_K8192)
            dn_matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

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
                        dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
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

        # --- add row 3 (verbatim C1 add herd; runs add1 then add2 waves) ---
        for col in reversed(range(N_COLS)):
            def _make_add_mem(_ct, _cl, _bo, _b2, _b1):
                @mem(_ct)
                def _core_mem(block):
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

            def _make_add_core(_ct, _cl, _bo, _b2, _b1):
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
                            dma_bd(_wb, offset=0, len=K_TILE_K8192 * HIDDEN_DIM)
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

        # --- mem tile DMAs: matvec W/y chains + (with_down) down W/y chains ---
        def _make_memtile_dma(_col, _ml, _w, _y, _dw, _dy, _mxb=None):
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
                    with block[10]:
                        use_lock(_ml["dw_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_dw, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
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
                    with block[14]:
                        use_lock(_ml["dw_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_dw, offset=0, len=M_TILE_K8192 * HIDDEN_DIM)
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
                              mem_buf_x.get(col))

        # --- runtime sequence ---
        @runtime_sequence(*o_gemv_ffn_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
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


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device(
    dispatch_sequence: Sequence[str] = DEFAULT_DISPATCH_SEQUENCE,
) -> None:
    """Emit the unnamed top-level dispatcher device.

    By default this fires the 8 segments in pipeline order:
        og -> a1 -> rm -> gg -> ug -> sw -> dg -> a2.
    ``dispatch_sequence`` is exposed for dispatch-overhead microbenchmarks
    that need reduced or repeated inner ``aiex.run`` sequences without
    changing the production kernel.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *o_gemv_ffn_host_arg_types(),
            sym_name="o_gemv_ffn",
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
# Public entry point.
# ---------------------------------------------------------------------------
def build_o_gemv_ffn_module(
    emb_dim: int = EMB_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dispatch_sequence: Sequence[str] | None = None,
    pack_mode: str = "none",
) -> str:
    """Build the o_gemv_ffn ``aie/aiex``-dialect module.

    Both dimensions are fixed to the Llama-3.2-1B values (the cached AIR
    layout is shape-specialized). ``dispatch_sequence`` is only for
    measurement variants; leave it unset for production IR.
    ``pack_mode="d1d4"`` emits the experimental D1/D4 packed devices;
    ``pack_mode="d1d3d4"`` also packs gate/up/SwiGLU into D3.
    """
    if emb_dim != EMB_DIM or hidden_dim != HIDDEN_DIM:
        raise ValueError(
            f"o_gemv_ffn builder is fixed to emb_dim={EMB_DIM}, "
            f"hidden_dim={HIDDEN_DIM}; got emb_dim={emb_dim}, "
            f"hidden_dim={hidden_dim}."
        )

    if pack_mode not in {"none", "d1d4", "d1d3d4", "d1d3d4_n2l2", "d1d3d4_rms",
                         "d1d3d4_rms_pkt", "d1d3d4_rms_fmv", "c1_merged",
                         "c2_rms", "c2_merged"}:
        raise ValueError(f"unsupported o_gemv_ffn pack_mode={pack_mode!r}")
    # "d1d3d4_n2l2" == d1d3d4 plus normed2 chained rm->D3 through col-0 L2.
    # Step A (current): rm ALSO writes normed2 to L2 (DDR write kept, D3 still
    # reads DDR -> hf-gate neutral). Step B will re-source the D3 broadcast.
    _n2l2 = pack_mode == "d1d3d4_n2l2"
    # "d1d3d4_rms" == air's 3-device fold: the separate rm_rms (D2) device is
    # ELIMINATED; each gate/up tile receives the pre-norm res1 + ffn_norm_w
    # (packed [2,K] on the existing input channel) and computes the RMSNorm
    # itself via the fused matvec_rms kernel. normed2 (arg6) goes unused. This
    # reuses the WORKING DDR broadcast (not the racy n2l2 L2 path) and drops a
    # per-token device dispatch (3 devices: d1 / gg+ug+sw+rms / d4).
    # "d1d3d4_rms_pkt" == d1d3d4_rms (RMS-fused, 3 devices) plus proj-engine
    # step 1a: the D3 result path (SwiGLU -> shim) is carried over a single-ID
    # packetflow instead of a circuit-switched flow, same destination. Pure
    # convergence toward the packet-fed proj-engine; bit-exact, still 3 devices,
    # still host-dispatched.
    # "d1d3d4_rms_fmv" == d1d3d4_rms plus proj-engine probe: the K=8192 down
    # matvec (D4) runs the mode-switched matvec_fused kernel (both K=2048 and
    # K=8192 bodies compiled in) selected by a per-tile mode RTP hard-coded to
    # 1. Runs the same 128-iter body -> bit-exact; measures the perf/size cost
    # of carrying both core bodies behind one mode RTP (the proj-engine
    # primitive). Still 3 devices, still host-dispatched.
    _rmsfuse = pack_mode in {"d1d3d4_rms", "d1d3d4_rms_pkt", "d1d3d4_rms_fmv"}
    _result_pkt = pack_mode == "d1d3d4_rms_pkt"
    _fused_mv = pack_mode == "d1d3d4_rms_fmv"
    # "c1_merged" == C1 of the collapse plan (RESIDENT_DEVICE_EVOLUTION.md):
    # ONE merged device runs O/add1/rms/gate/up/swiglu as sequential waves on
    # a reused row-2 matvec herd (rows 3/4/5 hold add/swiglu/rms); D4 (down +
    # add2) stays separate. DDR handoffs kept -> host args unchanged; call 2
    # goes 3 -> 2 configures.
    _c1 = pack_mode == "c1_merged"
    # "c2_rms" == C2a: c1_merged with the RMS stage folded into the gate/up
    # waves (d1d3d4_rms fold on the reused row-2 core); rms tile gone, D4 kept.
    # "c2_merged" == C2b: c2_rms + down/add2 folded in (row-5 K=8192 herd,
    # add herd runs 2 waves); call 2 = ONE configure.
    _c2 = pack_mode in {"c2_rms", "c2_merged"}
    _c2_down = pack_mode == "c2_merged"

    with mlir_mod_ctx() as ctx:
        # AIR emit order is reverse pipeline order.  In d1d4 mode, D4 replaces
        # {dg,a2} and D1 replaces {og,a1}; the middle {rm,gg,ug,sw} remains
        # unchanged until the D3 pack lands.
        if _c2_down:
            pass  # down+add2 live inside c2_merged
        elif pack_mode in {"d1d4", "d1d3d4", "d1d3d4_n2l2", "d1d3d4_rms",
                           "d1d3d4_rms_pkt", "d1d3d4_rms_fmv", "c1_merged",
                           "c2_rms"}:
            _emit_matvec_add_pack_k8192(
                "d4_dg_a2_pack",
                "dg_matvec_bf16_0",
                "a2_eltwise_add_seg",
                weight_arg_idx=12,
                input_arg_idx=11,
                residual_arg_idx=4,
                output_arg_idx=14,
                fused_mv=_fused_mv,
            )
        else:
            _emit_eltwise_add_seg(
                "a2_eltwise_add_seg", in0_arg_idx=13, in1_arg_idx=4, out_arg_idx=14)
            _emit_matvec_seg_k8192(
                "dg_matvec_bf16_0", weight_arg_idx=12, input_arg_idx=11,
                output_arg_idx=13, pingpong_w_l2=True)
        # pingpong_w (L1) intentionally off: dg's L1 is already at the
        # 64 KB cap with one W buffer (yb 1 KB + wb 16 KB + xb 16 KB +
        # slack), so doubling W has nowhere to go. We ping-pong at L2
        # instead -- memtile L2 has 512 KB to spare for the second W
        # slot (32 KB).
        if _c1:
            _emit_call2_merged()
        elif _c2:
            _emit_call2_c2(pack_mode, with_down=_c2_down)
        elif pack_mode in {"d1d3d4", "d1d3d4_n2l2", "d1d3d4_rms",
                           "d1d3d4_rms_pkt", "d1d3d4_rms_fmv"}:
            _emit_gg_ug_swiglu_pack(normed2_l2=_n2l2, rms_fused=_rmsfuse,
                                    result_pkt=_result_pkt, fused_mv=_fused_mv)
        else:
            _emit_sw_silu_mul_seg()
            _emit_matvec_seg_k2048(
                "ug_matvec_bf16_0", weight_arg_idx=9, input_arg_idx=6,
                output_arg_idx=10, out_rows=HIDDEN_DIM)
            _emit_matvec_seg_k2048(
                "gg_matvec_bf16_0", weight_arg_idx=7, input_arg_idx=6,
                output_arg_idx=8, out_rows=HIDDEN_DIM)
        # air's fold eliminates the standalone rm_rms device entirely; the C1
        # merge folds rm AND D1 (og+a1) into the merged device too.
        if not _rmsfuse and not _c1 and not _c2:
            _emit_rm_rms_seg(normed2_l2=_n2l2)
        if _c1 or _c2:
            pass  # og+a1 live inside the merged device
        elif pack_mode in {"d1d4", "d1d3d4", "d1d3d4_n2l2", "d1d3d4_rms",
                           "d1d3d4_rms_pkt", "d1d3d4_rms_fmv"}:
            _emit_matvec_add_pack_k2048(
                "d1_og_a1_pack",
                "og_matvec_bf16_0",
                "a1_eltwise_add_seg",
                weight_arg_idx=0,
                input_arg_idx=1,
                residual_arg_idx=3,
                output_arg_idx=4,
                out_rows=EMB_DIM,
                fused_mv=_fused_mv,
            )
        else:
            _emit_eltwise_add_seg(
                "a1_eltwise_add_seg", in0_arg_idx=2, in1_arg_idx=3, out_arg_idx=4)
            _emit_matvec_seg_k2048(
                "og_matvec_bf16_0", weight_arg_idx=0, input_arg_idx=1,
                output_arg_idx=2, out_rows=EMB_DIM)
        if dispatch_sequence is None:
            if _c1:
                # C1: one merged device + the separate down+add2 pack.
                dispatch_sequence = ("c1_merged", "d4_dg_a2_pack")
            elif pack_mode == "c2_rms":
                import os as _os
                dispatch_sequence = (
                    ("c2_rms",) if _os.environ.get("PYTHOC_C2_NO_D4")
                    else ("c2_rms", "d4_dg_a2_pack"))
            elif pack_mode == "c2_merged":
                # C2: everything in ONE device / ONE configure.
                dispatch_sequence = ("c2_merged",)
            elif pack_mode in {"d1d3d4_rms", "d1d3d4_rms_pkt", "d1d3d4_rms_fmv"}:
                # 3 devices: rm_rms folded into the gate/up stage.
                dispatch_sequence = (
                    "d1_og_a1_pack",
                    "d3_gg_ug_sw_pack",
                    "d4_dg_a2_pack",
                )
            elif pack_mode in {"d1d3d4", "d1d3d4_n2l2"}:
                dispatch_sequence = (
                    "d1_og_a1_pack",
                    "rm_rms_seg",
                    "d3_gg_ug_sw_pack",
                    "d4_dg_a2_pack",
                )
            elif pack_mode == "d1d4":
                dispatch_sequence = (
                    "d1_og_a1_pack",
                    "rm_rms_seg",
                    "gg_matvec_bf16_0",
                    "ug_matvec_bf16_0",
                    "sw_silu_mul_seg",
                    "d4_dg_a2_pack",
                )
            else:
                dispatch_sequence = DEFAULT_DISPATCH_SEQUENCE
        _emit_dispatcher_device(dispatch_sequence)
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs cached MLIR).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-mode",
                        choices=["none", "d1d4", "d1d3d4", "d1d3d4_n2l2",
                                 "d1d3d4_rms", "d1d3d4_rms_pkt",
                                 "d1d3d4_rms_fmv", "c1_merged",
                                 "c2_rms", "c2_merged"],
                        default="none", help="Experimental device packing mode")
    parser.add_argument("-o", "--output", help="Output path (default: stdout)",
                        default=None)
    args = parser.parse_args()
    text = build_o_gemv_ffn_module(pack_mode=args.pack_mode)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
