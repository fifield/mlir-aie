# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b RMS+GEMV+RoPE decode kernel -- AWQ.

This is the Stage-3 AWQ counterpart of ``builders/rms_gemv_rope.py``. The
6-segment topology (1 RMSNorm + 3 GEMVs + 2 RoPEs) is identical; the 3
GEMV segments swap their BF16 weight DMAs for packed-uint4 + groupwise
parameter (AWQ) weight DMAs and call AWQ external kernels:

    q/k/v matvec:  awq_matvec_vectorized_u4_bf16 + awq_linalg_fill_bf16
                   link_with "awq_mv_pythoc.o"
                   weight memref: ui8[M, K/2 + 4*(K/group_size)]   (K=2048,
                                                                  row=1088)

The RMSNorm and RoPE segments are reused verbatim (modulo the dispatcher
host signature swap to the AWQ 13-arg version, since args 3/5/7 are now
ui8 packed weights instead of bf16).

Module layout (matches the BF16 sibling structurally)::

    module {
      aie.device(npu2) @rk_rope_awq_seg     { ... }   # 1 compute tile
      aie.device(npu2) @rq_rope_awq_seg     { ... }   # 1 compute tile
      aie.device(npu2) @v_matvec_awq_bf16_0 { ... }   # 8 compute tiles
      aie.device(npu2) @k_matvec_awq_bf16_0 { ... }   # 8 compute tiles
      aie.device(npu2) @q_matvec_awq_bf16_0 { ... }   # 8 compute tiles
      aie.device(npu2) @r_rms_awq_seg       { ... }   # 1 compute tile
      aie.device(npu2) {                                # dispatcher
        aiex.runtime_sequence @rms_gemv_rope_awq(...) {
          aiex.configure @r_rms_awq_seg       { aiex.run ... }
          aiex.configure @q_matvec_awq_bf16_0 { aiex.run ... }
          aiex.configure @k_matvec_awq_bf16_0 { aiex.run ... }
          aiex.configure @v_matvec_awq_bf16_0 { aiex.run ... }
          aiex.configure @rq_rope_awq_seg     { aiex.run ... }
          aiex.configure @rk_rope_awq_seg     { aiex.run ... }
        }
      }
    }

References:
  * ``builders/rms_gemv_rope.py`` -- BF16 sibling template.
  * ``builders/o_gemv_ffn_awq.py`` -- K=2048 AWQ matvec emit pattern;
    see ``_emit_awq_matvec_seg_k2048`` for the AWQ strides + ui8 buffer
    types this module reuses.
  * ``builders/lm_head_gemv_awq.py`` -- prior AWQ port that extends a
    BF16 builder by swapping weight memrefs.
  * ``kernels/awq_mv.py`` -- external AWQ K=2048 kernel ABI.
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
    rms_gemv_rope_awq_host_arg_types,
)


# ---------------------------------------------------------------------------
# Constants matching the cached AIR-stitched IR for Llama-3.2-1B.
# ---------------------------------------------------------------------------
EMB_DIM = 2048      # model hidden size
KV_DIM = 512        # n_kv_heads * head_dim = 8 * 64
HEAD_DIM = 64       # per-head dimension (RoPE chunk size)
N_COLS = 8          # 8 compute columns in the matvec herd
K_TILE = 8          # inner K tiling factor for the matvec kernel
M_TILE = 8          # rows processed per matvec call
# K_TILE = M_TILE => K-loop is a single iter. Doubling K_TILE from the
# original 4 grows the W L1 tile from 4.25 KB to 8.5 KB but halves the
# per-K-iter lock acquire/release cycle count, the BD setup overhead,
# and the K-loop prolog. L1 still has plenty of headroom (~12 KB used
# total vs 64 KB cap). Mirrors the K_TILE_K8192=2 change on dg AWQ.

# AWQ row layout: K/2 packed uint4 bytes + 4 bytes per group of params.
GROUP_SIZE = 128


def _combined_row_bytes(k: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes per AWQ row: K/2 packed uint4 + 4*(K/group_size) param bytes."""
    return k // 2 + 4 * (k // group_size)


ROW_BYTES = _combined_row_bytes(EMB_DIM)  # 1088

# Per-segment kernel object filenames. RMSNorm + RoPE stay BF16 (unchanged
# external kernels); the 3 GEMVs swap to the AWQ kernel object.
KO_AWQ_MV = "awq_mv_pythoc.o"
KO_ROPE = "rope_pythoc.o"
KO_RMS = "rms_norm_2048_bf16.o"

DEFAULT_DISPATCH_SEQUENCE = (
    "r_rms_awq_seg",
    "q_matvec_awq_bf16_0",
    "k_matvec_awq_bf16_0",
    "v_matvec_awq_bf16_0",
    "rq_rope_awq_seg",
    "rk_rope_awq_seg",
)

RGR2_PACK_SYM = "rgr2_qkv_rope_pack"


# ---------------------------------------------------------------------------
# Channel-number map. We reuse the exact BF16 IDs because shim_dma_allocation
# symbols (``air_channel_<num>_<col>``) live in per-device scope; AWQ-tree
# orchestration in ``llama32_1b_awq_runtime.py`` references the same numbers.
# ---------------------------------------------------------------------------
_CHANNELS: Dict[str, Dict[str, object]] = {
    "rk_rope_awq_seg":     {"in0": 21, "in1": 22, "out": 23},
    "rq_rope_awq_seg":     {"in0": 18, "in1": 19, "out": 20},
    "r_rms_awq_seg":       {"in0": 0,  "in1": 1,  "out": 2},
    "v_matvec_awq_bf16_0": {"weight_base": 24, "out_base": 29, "input": 14},
    "k_matvec_awq_bf16_0": {"weight_base": 28, "out_base": 25, "input": 9},
    "q_matvec_awq_bf16_0": {"weight_base": 26, "out_base": 27, "input": 4},
}


# ---------------------------------------------------------------------------
# external_buffer triples emitted per device. AIR uses these as opaque
# metadata; aiecc treats them as references. Each entry is a tuple
# ``(shape_tuple, dtype)`` where dtype is ``"bf16"`` or ``"ui8"``.
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
# RMSNorm segment (@r_rms_awq_seg). Identical structurally to the BF16
# sibling -- only the dispatcher host signature swap (rms_gemv_rope_awq_
# host_arg_types) is required.
# ---------------------------------------------------------------------------
def _emit_r_rms_seg() -> None:
    sym = "r_rms_awq_seg"
    chans = _CHANNELS[sym]

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim = tile(0, 0)
        ct = tile(0, 2)

        # Locks in AIR's emit order (5..0).
        lk5 = lock(ct, lock_id=5, init=1)  # weight avail
        lk4 = lock(ct, lock_id=4, init=0)  # weight ready
        lk3 = lock(ct, lock_id=3, init=1)  # x avail
        lk2 = lock(ct, lock_id=2, init=0)  # x ready
        lk1 = lock(ct, lock_id=1, init=1)  # y done
        lk0 = lock(ct, lock_id=0, init=0)  # y full

        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        _BF16_16_L1 = bf16_memref(16, memory_space=2)
        buf_w = buffer(ct, datatype=_BF16_2048_L1)   # weight
        buf_y = buffer(ct, datatype=_BF16_2048_L1)   # output
        buf_x = buffer(ct, datatype=_BF16_2048_L1)   # input
        buf_s = buffer(ct, datatype=_BF16_16_L1)     # scratch

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
                dma_bd(buf_x, offset=0, len=EMB_DIM)
                use_lock(lk2, LockAction.Release, value=1)
                next_bd(block[4])
            with block[5]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
            with block[6]:
                use_lock(lk5, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_w, offset=0, len=EMB_DIM)
                use_lock(lk4, LockAction.Release, value=1)
                next_bd(block[6])

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
                rms_fn(buf_w, buf_x, buf_y, buf_s)
                use_lock(lk5, LockAction.Release, value=1)
                use_lock(lk0, LockAction.Release, value=1)
                use_lock(lk3, LockAction.Release, value=1)

        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        @runtime_sequence(*rms_gemv_rope_awq_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12):
            t0 = dma_configure_task_for(f"air_channel_{chans['in0']}")
            with bds(t0) as bd:
                with bd[0]:
                    dma_bd(arg1, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t0)
            t1 = dma_configure_task_for(f"air_channel_{chans['in1']}")
            with bds(t1) as bd:
                with bd[0]:
                    dma_bd(arg0, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t1)
            t2 = dma_configure_task_for(f"air_channel_{chans['out']}",
                                         issue_token=True)
            with bds(t2) as bd:
                with bd[0]:
                    dma_bd(arg2, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t2)
            dma_await_task(t2)
            dma_free_task(t0)
            dma_free_task(t1)


# ---------------------------------------------------------------------------
# RoPE segments. Identical to BF16 sibling apart from the AWQ 13-arg
# dispatcher signature.
# ---------------------------------------------------------------------------
def _emit_rope_seg(sym: str, x_arg_idx: int, freqs_arg_idx: int,
                   out_arg_idx: int, vec_size: int) -> None:
    chans = _CHANNELS[sym]
    n_iters = vec_size // HEAD_DIM

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

        _BF16_64_L1 = bf16_memref(HEAD_DIM, memory_space=2)
        buf_y = buffer(ct, datatype=_BF16_64_L1)
        buf_f = buffer(ct, datatype=_BF16_64_L1)
        buf_x = buffer(ct, datatype=_BF16_64_L1)

        _emit_external_buffers_bf16((vec_size,), (vec_size,), (vec_size,))

        @mem(ct)
        def _core_mem(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(lk0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_y, offset=0, len=HEAD_DIM)
                use_lock(lk1, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                EndOp()
            with block[3]:
                dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
            with block[4]:
                use_lock(lk3, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_x, offset=0, len=HEAD_DIM)
                use_lock(lk2, LockAction.Release, value=1)
                next_bd(block[4])
            with block[5]:
                dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
            with block[6]:
                use_lock(lk5, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf_f, offset=0, len=HEAD_DIM)
                use_lock(lk4, LockAction.Release, value=1)
                next_bd(block[6])

        rope_fn = external_func(
            "rope",
            inputs=[_BF16_64_L1, _BF16_64_L1, _BF16_64_L1, np.int32],
            link_with=KO_ROPE,
        )
        rope_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        from aie.extras import types as T

        import sys as _sys

        @core(ct)
        def _core_body():
            head_dim_c = arith.constant(HEAD_DIM, T.i32())
            for _outer in range_(_sys.maxsize):
                for _ in range_(0, n_iters, 1):
                    use_lock(lk1, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(lk2, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(lk4, LockAction.AcquireGreaterEqual, value=1)
                    rope_fn(buf_x, buf_f, buf_y, head_dim_c)
                    use_lock(lk3, LockAction.Release, value=1)
                    use_lock(lk5, LockAction.Release, value=1)
                    use_lock(lk0, LockAction.Release, value=1)

        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        @runtime_sequence(*rms_gemv_rope_awq_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_x = args[x_arg_idx]
            arg_f = args[freqs_arg_idx]
            arg_y = args[out_arg_idx]

            if vec_size == EMB_DIM:
                dims = [(4, 512), (512, 1)]
            else:
                dims = [(512, 1)]

            t0 = dma_configure_task_for(f"air_channel_{chans['in0']}")
            with bds(t0) as bd:
                with bd[0]:
                    dma_bd(arg_x, offset=0, len=vec_size, dimensions=dims)
                    EndOp()
            dma_start_task(t0)
            t1 = dma_configure_task_for(f"air_channel_{chans['in1']}")
            with bds(t1) as bd:
                with bd[0]:
                    dma_bd(arg_f, offset=0, len=vec_size, dimensions=dims)
                    EndOp()
            dma_start_task(t1)
            t2 = dma_configure_task_for(f"air_channel_{chans['out']}",
                                         issue_token=True)
            with bds(t2) as bd:
                with bd[0]:
                    dma_bd(arg_y, offset=0, len=vec_size, dimensions=dims)
                    EndOp()
            dma_start_task(t2)
            dma_await_task(t2)
            dma_free_task(t0)
            dma_free_task(t1)


# ---------------------------------------------------------------------------
# AWQ GEMV matvec segment (K=2048, awq_mv_pythoc.o). Shared by q/k/v.
#
# Structurally a shape-shrunk version of one partition from
# ``o_gemv_ffn_awq.py::_emit_awq_matvec_seg_k2048``. The only knobs that
# vary across q vs k/v:
#   - ``out_rows`` (EMB_DIM=2048 for Q; KV_DIM=512 for K/V)
#   - host arg indices (passed in)
#   - n_outer iters (2 for Q; 1 for K/V) and per-outer dim shapes
#   - per-col output stride and input repeat count
# ---------------------------------------------------------------------------
def _emit_awq_matvec_seg(sym: str, weight_arg_idx: int, output_arg_idx: int,
                          out_rows: int, pingpong_w: bool = False,
                          pingpong_w_l2: bool = False) -> None:
    """Emit one AWQ q/k/v matvec segment device.

    Q variant (out_rows=EMB_DIM=2048): n_outer=2, output 128 elts/col/outer
    using the (16,64),(8,1) pattern; weight uses 16 mini-rows of 16x544 byte
    chunks (139264 bytes per task).

    K/V variant (out_rows=KV_DIM=512): n_outer=1, output 64 elts/col using
    (8,64),(8,1) pattern; weight uses 8 mini-rows of 16x544 byte chunks
    (69632 bytes per task). Mini-row outer stride is 69632 = 64 source rows
    of 1088 bytes -- the same striped row pattern as the BF16 K/V variant,
    just AWQ-bytes-per-element instead of bf16 elements.

    ``pingpong_w=True`` doubles the L1 W buffer (~4.25 KB each for the AWQ
    ui8-packed slab, vs 16 KB bf16) and turns the W DMA into a 2-BD ring;
    ``w_avail`` becomes init=2. The K_TILE-step inner loop is unrolled
    (M_TILE/K_TILE=2). See builders/rms_gemv_rope.py for the rationale.

    ``pingpong_w_l2=True`` does the same one hop upstream: doubles the
    memtile L2 W buffer (~8.5 KB each), splits both memtile chains into
    2-BD rings, and raises ``w_dma_done`` to init=2. The two flags are
    independent; combining both gives a 2-deep pipeline shim->L2->L1.
    """
    chans = _CHANNELS[sym]
    row_bytes = ROW_BYTES  # 1088

    if out_rows == KV_DIM:
        n_outer = 1
        y_dims = [(8, 64), (8, 1)]
        y_len = 64
        x_repeat_count = 15
        # 8 mini-rows of 16 chunks of 544 bytes = 69632 bytes per col.
        # mini-row outer stride 69632 = 64 src rows * 1088 bytes/row.
        w_dims = [(8, 69632), (16, 544), (544, 1)]
        w_len = 8 * 16 * 544                                  # 69632
        weight_col_stride = M_TILE * row_bytes                # 8 * 1088 = 8704
        weight_outer_stride = 0                                # unused
        output_col_stride = M_TILE                             # 8
        output_outer_stride = 0                                # unused
    else:
        assert out_rows == EMB_DIM
        n_outer = 2
        y_dims = [(16, 64), (8, 1)]
        y_len = 128
        x_repeat_count = 31
        w_dims = [(16, 69632), (16, 544), (544, 1)]
        w_len = 16 * 16 * 544                                 # 139264
        weight_col_stride = M_TILE * row_bytes                # 8 * 1088 = 8704
        weight_outer_stride = 1024 * row_bytes                # 1_114_112
        output_col_stride = M_TILE                             # 8
        output_outer_stride = 1024

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Mem tile locks (descending col).
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

        # Compute tile locks (ascending col).
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

        # ui8 weight buffer types (the AWQ change vs BF16 sibling).
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

        # Mem tile buffers (descending col).
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
        for col in reversed(range(N_COLS)):
            core_buf_y[col] = buffer(compute_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            if pingpong_w:
                core_buf_w1[col] = buffer(compute_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(compute_tiles[col], datatype=_X_L1_TY)

        # External buffers: weight (out_rows, row_bytes) ui8, input (K,) bf16,
        # output (out_rows,) bf16.
        _emit_external_buffers(
            ((out_rows, row_bytes), "ui8"),
            ((EMB_DIM,), "bf16"),
            ((out_rows,), "bf16"),
        )

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
                            dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        # L1 ping-pong: 2-BD ring writing wb0 then wb1.
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
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            matvec_fn(k_tile_c, k_total, k_i32_1, _wb1, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct_op, cl, y_buf, w_buf, x_buf, w_buf1)

        # Flows.
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
        def _make_memtile_dma(_col, _ml, _w, _w1, _y):
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
                    # L2 ping-pong: 2-BD rings on both MM2S ch 1 (L2->L1)
                    # and S2MM ch 0 (shim->L2).
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=M_TILE * row_bytes)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[9])
                    with block[9]:
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
                        next_bd(block[10])
                    with block[10]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=M_TILE * row_bytes)
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
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col],
                              mem_buf_w1.get(col), mem_buf_y[col])

        # Shim allocations.
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
        @runtime_sequence(*rms_gemv_rope_awq_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_x = args[2]
            arg_w = args[weight_arg_idx]
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
# Experimental RGR2 6->2 packing device (AWQ).
# ---------------------------------------------------------------------------
def _emit_awq_qkv_rope_pack(sym: str = RGR2_PACK_SYM) -> None:
    """Emit a single RGR2 device for AWQ Q/K/V matvecs plus Q/K RoPE.

    AWQ counterpart of ``builders/rms_gemv_rope.py::_emit_qkv_rope_pack``.
    Structurally identical; the only differences are weight-side: ui8 packed
    L1/L2 buffers, AWQ kernel symbols/link object, AWQ row-byte DMA strides.

    The device reuses one 8-core matvec herd for Q, K, and V sequentially in a
    single runtime_sequence, then runs the existing one-core RoPE kernels for Q
    and K before returning to the host dispatcher.  The outer dispatcher thus
    has only two device runs: r_rms_awq_seg and this packed RGR2 sequence.
    """
    mat_chans = _CHANNELS["q_matvec_awq_bf16_0"]
    rq_chans = _CHANNELS["rq_rope_awq_seg"]
    rk_chans = _CHANNELS["rk_rope_awq_seg"]
    row_bytes = ROW_BYTES  # 1088

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        mat_tiles = [tile(c, 2) for c in range(N_COLS)]
        rq_tile = tile(2, 3)
        rk_tile = tile(5, 3)

        # Matvec locks and buffers: one reusable Q/K/V herd.
        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready": lock(mt, lock_id=2, init=0),
                "y_done": lock(mt, lock_id=1, init=1),
                "y_ready": lock(mt, lock_id=0, init=0),
            }

        core_locks = {}
        for col in range(N_COLS):
            ct = mat_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=1),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done": lock(ct, lock_id=1, init=1),
                "y_full": lock(ct, lock_id=0, init=0),
            }

        # RoPE locks.  Each tile uses the standalone RoPE lock numbering.
        def _rope_locks(_ct):
            return {
                "freqs_avail": lock(_ct, lock_id=5, init=1),
                "freqs_ready": lock(_ct, lock_id=4, init=0),
                "x_avail": lock(_ct, lock_id=3, init=1),
                "x_ready": lock(_ct, lock_id=2, init=0),
                "y_done": lock(_ct, lock_id=1, init=1),
                "y_full": lock(_ct, lock_id=0, init=0),
            }

        rq_locks = _rope_locks(rq_tile)
        rk_locks = _rope_locks(rk_tile)

        # ui8 weight buffer types (the AWQ change vs BF16 sibling).
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
        _BF16_64_L1 = bf16_memref(HEAD_DIM, memory_space=2)

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
            core_buf_y[col] = buffer(mat_tiles[col], datatype=_Y_L1_TY)
            core_buf_w[col] = buffer(mat_tiles[col], datatype=_W_L1_TY)
            core_buf_x[col] = buffer(mat_tiles[col], datatype=_X_L1_TY)

        def _rope_buffers(_ct):
            return {
                "y": buffer(_ct, datatype=_BF16_64_L1),
                "f": buffer(_ct, datatype=_BF16_64_L1),
                "x": buffer(_ct, datatype=_BF16_64_L1),
            }

        rq_bufs = _rope_buffers(rq_tile)
        rk_bufs = _rope_buffers(rk_tile)

        _emit_external_buffers(
            ((EMB_DIM, row_bytes), "ui8"),
            ((EMB_DIM,), "bf16"),
            ((EMB_DIM,), "bf16"),
        )

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
        rope_fn = external_func(
            "rope",
            inputs=[_BF16_64_L1, _BF16_64_L1, _BF16_64_L1, np.int32],
            link_with=KO_ROPE,
        )
        rope_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in reversed(range(N_COLS)):
            ct_op = mat_tiles[col]
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
                        dma_bd(_wb, offset=0, len=K_TILE * row_bytes)
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

        def _make_rope_tile(_ct, _locks, _bufs, _n_iters):
            @mem(_ct)
            def _rope_mem(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_locks["y_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["y"], offset=0, len=HEAD_DIM)
                    use_lock(_locks["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                with block[4]:
                    use_lock(_locks["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["x"], offset=0, len=HEAD_DIM)
                    use_lock(_locks["x_ready"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                with block[6]:
                    use_lock(_locks["freqs_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["f"], offset=0, len=HEAD_DIM)
                    use_lock(_locks["freqs_ready"], LockAction.Release, value=1)
                    next_bd(block[6])

            @core(_ct)
            def _rope_core():
                import sys as _sys
                head_dim_c = arith.constant(HEAD_DIM, T.i32())
                for _outer in range_(_sys.maxsize):
                    for _ in range_(0, _n_iters, 1):
                        use_lock(_locks["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_locks["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_locks["freqs_ready"], LockAction.AcquireGreaterEqual, value=1)
                        rope_fn(_bufs["x"], _bufs["f"], _bufs["y"], head_dim_c)
                        use_lock(_locks["x_avail"], LockAction.Release, value=1)
                        use_lock(_locks["freqs_avail"], LockAction.Release, value=1)
                        use_lock(_locks["y_full"], LockAction.Release, value=1)

        _make_rope_tile(rq_tile, rq_locks, rq_bufs, EMB_DIM // HEAD_DIM)
        _make_rope_tile(rk_tile, rk_locks, rk_bufs, KV_DIM // HEAD_DIM)

        # Matvec flows.
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)

        # RoPE flows.  The matvec herd already uses shim MM2S0 on every
        # column, shim0 MM2S1 for the normed input multicast, and shim S2MM0
        # on every column.  Route RoPE through otherwise-unused physical shim
        # channels to avoid duplicate static connects while preserving the
        # original logical air_channel ids.
        flow(shim_tiles[2], WireBundle.DMA, 1, rq_tile, WireBundle.DMA, 0)
        flow(shim_tiles[3], WireBundle.DMA, 1, rq_tile, WireBundle.DMA, 1)
        flow(rq_tile, WireBundle.DMA, 0, shim_tiles[4], WireBundle.DMA, 1)
        flow(shim_tiles[5], WireBundle.DMA, 1, rk_tile, WireBundle.DMA, 0)
        flow(shim_tiles[6], WireBundle.DMA, 1, rk_tile, WireBundle.DMA, 1)
        flow(rk_tile, WireBundle.DMA, 0, shim_tiles[7], WireBundle.DMA, 1)

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

        out_base = mat_chans["out_base"]
        weight_base = mat_chans["weight_base"]
        input_chan = mat_chans["input"]
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
        shim_dma_allocation(f"air_channel_{rq_chans['out']}", shim_tiles[4], DMAChannelDir.S2MM, 1)
        shim_dma_allocation(f"air_channel_{rq_chans['in0']}", shim_tiles[2], DMAChannelDir.MM2S, 1)
        shim_dma_allocation(f"air_channel_{rq_chans['in1']}", shim_tiles[3], DMAChannelDir.MM2S, 1)
        shim_dma_allocation(f"air_channel_{rk_chans['out']}", shim_tiles[7], DMAChannelDir.S2MM, 1)
        shim_dma_allocation(f"air_channel_{rk_chans['in0']}", shim_tiles[5], DMAChannelDir.MM2S, 1)
        shim_dma_allocation(f"air_channel_{rk_chans['in1']}", shim_tiles[6], DMAChannelDir.MM2S, 1)

        @runtime_sequence(*rms_gemv_rope_awq_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_x = args[2]
            weight_col_stride = M_TILE * row_bytes
            output_col_stride = M_TILE

            def _run_matvec(arg_w, arg_y, n_outer, y_dims, y_len,
                            x_repeat_count, w_dims, w_len,
                            weight_outer_stride, output_outer_stride):
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

            def _run_rope(chans, arg_x_rope, arg_f, arg_y_rope, vec_size):
                if vec_size == EMB_DIM:
                    dims = [(4, 512), (512, 1)]
                else:
                    dims = [(512, 1)]

                t0 = dma_configure_task_for(f"air_channel_{chans['in0']}")
                with bds(t0) as bd:
                    with bd[0]:
                        dma_bd(arg_x_rope, offset=0, len=vec_size, dimensions=dims)
                        EndOp()
                dma_start_task(t0)
                t1 = dma_configure_task_for(f"air_channel_{chans['in1']}")
                with bds(t1) as bd:
                    with bd[0]:
                        dma_bd(arg_f, offset=0, len=vec_size, dimensions=dims)
                        EndOp()
                dma_start_task(t1)
                t2 = dma_configure_task_for(f"air_channel_{chans['out']}", issue_token=True)
                with bds(t2) as bd:
                    with bd[0]:
                        dma_bd(arg_y_rope, offset=0, len=vec_size, dimensions=dims)
                        EndOp()
                dma_start_task(t2)
                dma_await_task(t2)
                dma_free_task(t0)
                dma_free_task(t1)

            _run_matvec(
                args[3],
                args[4],
                2,
                [(16, 64), (8, 1)],
                128,
                31,
                [(16, 69632), (16, 544), (544, 1)],
                16 * 16 * 544,
                1024 * row_bytes,
                1024,
            )
            for weight_arg, output_arg in ((args[5], args[6]), (args[7], args[8])):
                _run_matvec(
                    weight_arg,
                    output_arg,
                    1,
                    [(8, 64), (8, 1)],
                    64,
                    15,
                    [(8, 69632), (16, 544), (544, 1)],
                    8 * 16 * 544,
                    0,
                    0,
                )
            _run_rope(rq_chans, args[4], args[9], args[11], EMB_DIM)
            _run_rope(rk_chans, args[6], args[10], args[12], KV_DIM)


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device(dispatch_sequence: Sequence[str] = None) -> None:
    """Emit the unnamed top-level dispatcher device.

    ``dispatch_sequence`` lists the device runtime sequences that form the
    outer ``aie.runtime_sequence @rms_gemv_rope_awq``.  The default path keeps
    AIR's 6-device sequence; experimental packing uses a 2-device sequence.
    All segments share the same AWQ 13-arg signature.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    if dispatch_sequence is None:
        dispatch_sequence = DEFAULT_DISPATCH_SEQUENCE

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *rms_gemv_rope_awq_host_arg_types(),
            sym_name="rms_gemv_rope_awq",
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
def build_rms_gemv_rope_awq_module(emb_dim: int = EMB_DIM,
                                    kv_dim: int = KV_DIM,
                                    n_heads: int = 32,
                                    n_kv_heads: int = 8,
                                    head_dim: int = HEAD_DIM,
                                    *,
                                    group_size: int = 128,
                                    pack_mode: str = "none") -> str:
    """Build the RMS+GEMV+RoPE AWQ ``aie/aiex``-dialect module.

    All dimension args must match the Llama-3.2-1B values; ``group_size``
    is accepted for API future-proofing but currently must equal 128 (the
    value baked into ``awq_mv_pythoc.o``).  ``pack_mode="rgr2_ddr"`` emits the
    experimental 2-device dispatcher: standalone RMS followed by one packed
    Q/K/V+RoPE runtime sequence.
    """
    if emb_dim != EMB_DIM or kv_dim != KV_DIM or head_dim != HEAD_DIM:
        raise ValueError(
            f"rms_gemv_rope_awq builder is fixed to emb_dim={EMB_DIM}, "
            f"kv_dim={KV_DIM}, head_dim={HEAD_DIM}; got "
            f"emb_dim={emb_dim}, kv_dim={kv_dim}, head_dim={head_dim}."
        )
    if group_size != 128:
        raise ValueError(
            f"rms_gemv_rope_awq builder is fixed to group_size=128; "
            f"got group_size={group_size}."
        )
    del n_heads, n_kv_heads

    pack_mode = (pack_mode or "none").strip()
    valid_pack_modes = {"none", "rgr2_ddr"}
    if pack_mode not in valid_pack_modes:
        raise ValueError(
            f"unknown rms_gemv_rope_awq pack_mode={pack_mode!r}; "
            f"expected one of {sorted(valid_pack_modes)}"
        )

    with mlir_mod_ctx() as ctx:
        if pack_mode == "rgr2_ddr":
            _emit_awq_qkv_rope_pack()
        else:
            _emit_rope_seg("rk_rope_awq_seg",
                           x_arg_idx=6, freqs_arg_idx=10, out_arg_idx=12,
                           vec_size=KV_DIM)
            _emit_rope_seg("rq_rope_awq_seg",
                           x_arg_idx=4, freqs_arg_idx=9, out_arg_idx=11,
                           vec_size=EMB_DIM)
            _emit_awq_matvec_seg("v_matvec_awq_bf16_0",
                                  weight_arg_idx=7, output_arg_idx=8,
                                  out_rows=KV_DIM)
            # pingpong_w / pingpong_w_l2 plumbed but off: prior trace showed
            # AWQ V kernel is not lock-stall-bound enough for W PP to claim
            # back cycles (lock_stall ~23%, span +2% when PP'd). K_TILE=8 is
            # the bigger-tile alternative; with M_TILE/K_TILE=1, PP can't be
            # enabled here anyway (unroll requires M/K==2).
            _emit_awq_matvec_seg("k_matvec_awq_bf16_0",
                                  weight_arg_idx=5, output_arg_idx=6,
                                  out_rows=KV_DIM)
            _emit_awq_matvec_seg("q_matvec_awq_bf16_0",
                                  weight_arg_idx=3, output_arg_idx=4,
                                  out_rows=EMB_DIM)
        _emit_r_rms_seg()
        dispatch_sequence = (
            ("r_rms_awq_seg", RGR2_PACK_SYM)
            if pack_mode == "rgr2_ddr"
            else DEFAULT_DISPATCH_SEQUENCE
        )
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
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: stdout)",
        default=None,
    )
    parser.add_argument("--pack-mode", choices=("none", "rgr2_ddr"),
                        default="none",
                        help="Experimental device packing mode")
    args = parser.parse_args()
    text = build_rms_gemv_rope_awq_module(pack_mode=args.pack_mode)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
