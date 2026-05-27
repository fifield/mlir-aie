# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b RMS+GEMV+RoPE decode kernel.

Replaces the cached AIR-stitched ``rms_gemv_rope.npu.air.mlir`` with an
mlir-aie Python program that emits structurally equivalent
``aie/aiex``-dialect text using the dialect Python bindings directly.

Module layout (matches the cached reference structurally)::

    module {
      aie.device(npu2) @rk_rope_seg   { ... }   # 1 compute tile
      aie.device(npu2) @rq_rope_seg   { ... }   # 1 compute tile
      aie.device(npu2) @v_matvec_bf16_0 { ... } # 8 compute tiles (herd_size [8,1])
      aie.device(npu2) @k_matvec_bf16_0 { ... } # 8 compute tiles
      aie.device(npu2) @q_matvec_bf16_0 { ... } # 8 compute tiles, 2 outer iters
      aie.device(npu2) @r_rms_seg     { ... }   # 1 compute tile
      aie.device(npu2) {                         # dispatcher
        aiex.runtime_sequence @rms_gemv_rope(...) {
          aiex.configure @r_rms_seg     { aiex.run ... }
          aiex.configure @q_matvec_bf16_0 { aiex.run ... }
          aiex.configure @k_matvec_bf16_0 { aiex.run ... }
          aiex.configure @v_matvec_bf16_0 { aiex.run ... }
          aiex.configure @rq_rope_seg   { aiex.run ... }
          aiex.configure @rk_rope_seg   { aiex.run ... }
        }
      }
    }

This is the *decode* fused-launch RMS+GEMV+RoPE kernel for Llama-3.2-1B.
Reads x (post-residual), produces rmsnorm-normalized x, then three GEMVs
(Q, K, V), then RoPE-Q and RoPE-K.

References:
  * ``reference_mlir/rms_gemv_rope.npu.air.mlir`` -- ground truth
    (3,113 lines, produced by AIR's aircc).
  * ``builders/lm_head_gemv.py`` -- the Phase 4.1 placed-IRON
    template. The matvec segments here are essentially shape-shrunk
    versions of one partition from ``lm_head_gemv``.
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

from ._emit import (
    attach_loop_annotation_to_all_scf_for,
    bf16_memref,
    bf16_np,
    rms_gemv_rope_host_arg_types,
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
# K_TILE = M_TILE => K-loop is a single iter. Mirrors the same change
# on AWQ K=2048 builders. With M_TILE/K_TILE==1, pingpong_w cannot be
# enabled here (the unroll assertion requires M_TILE/K_TILE==2).

# Per-segment kernel object filenames (must match what the kernel
# builder cached on disk under ``build_peano/``). aie-assign-core-link-files
# reads ``link_with`` from each ``external_func`` decl and attaches it
# to the core op -- same mechanism AIR uses.
KO_MATVEC = "mv_pythoc.o"
KO_ROPE = "rope_pythoc.o"
KO_RMS = "rms_norm_2048_bf16.o"


# ---------------------------------------------------------------------------
# Channel-number map (verbatim from the cached IR).
#
# AIR assigns numeric ids to each air_channel during stitching; we reuse
# those exact ids so the resulting ``aie.shim_dma_allocation`` symbols
# match what the orchestration in ``llama32_1b_decode.py`` references
# via the dispatcher's ``aiex.run``.
# ---------------------------------------------------------------------------
_CHANNELS: Dict[str, Dict[str, object]] = {
    "rk_rope_seg":   {"in0": 21, "in1": 22, "out": 23},
    "rq_rope_seg":   {"in0": 18, "in1": 19, "out": 20},
    "r_rms_seg":     {"in0": 0,  "in1": 1,  "out": 2},
    "v_matvec_bf16_0": {"weight_base": 24, "out_base": 29, "input": 14},
    "k_matvec_bf16_0": {"weight_base": 28, "out_base": 25, "input": 9},
    "q_matvec_bf16_0": {"weight_base": 26, "out_base": 27, "input": 4},
}


# ---------------------------------------------------------------------------
# Helper: declare the three ``__air_external_buffer`` symbols inside a
# device.  AIR emits these as opaque metadata; aiecc treats them as
# references and the symbols are otherwise unused.  We mirror AIR's
# usage to keep the structural diff minimal.
# ---------------------------------------------------------------------------
def _emit_external_buffers(*shapes_with_dtype):
    """Emit 3 ``aie.external_buffer`` decls with auto-suffixed names.

    Each entry is a tuple ``(shape...,)`` describing a bf16 memref.
    Order matters: AIR emits ``__air_external_buffer`` first, then
    ``__air_external_buffer_1``, then ``__air_external_buffer_2``.
    """
    names = ["__air_external_buffer", "__air_external_buffer_1", "__air_external_buffer_2"]
    for nm, shp in zip(names, shapes_with_dtype):
        ty = bf16_np(*shp)
        external_buffer(ty, name=nm)


# ---------------------------------------------------------------------------
# RMSNorm segment (@r_rms_seg).
# ---------------------------------------------------------------------------
# Layout: 1 shim tile (col 0), 1 compute tile (0,2).
# Locks (per tile, 6 ids 5..0):
#   id=5 init=1: weight_l1_avail  (matches lock_0_2)
#   id=4 init=0: weight_l1_ready  (matches lock_0_2_0)
#   id=3 init=1: x_l1_avail       (matches lock_0_2_1)
#   id=2 init=0: x_l1_ready       (matches lock_0_2_2)
#   id=1 init=1: y_l1_done        (matches lock_0_2_3)
#   id=0 init=0: y_l1_full        (matches lock_0_2_4)
# Buffers (in cached emit order, top of device): buf3,buf2,buf1,buf0
#   buf3 = weight (2048xbf16, L1)
#   buf2 = y (output 2048xbf16, L1)
#   buf1 = x (input 2048xbf16, L1)
#   buf0 = scratch (16xbf16, L1)
# Channel routing:
#   air_channel_0 = MM2S 0 (shim DMA0) -> compute DMA0 = buf1 (x), from arg1
#   air_channel_1 = MM2S 1 (shim DMA1) -> compute DMA1 = buf3 (weight), from arg0
#   air_channel_2 = S2MM 0 (shim DMA0) <- compute DMA0 (out) = buf2 (y), to arg2
# Note: compute DMA0 is bidirectional between MM2S (output) and S2MM (input).
# core body: rms_norm_2048_bf16(weight, x, y, scratch).
def _emit_r_rms_seg() -> None:
    sym = "r_rms_seg"
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

        # Buffers, AIR emit order: buf3, buf2, buf1, buf0.
        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        _BF16_16_L1 = bf16_memref(16, memory_space=2)
        buf_w = buffer(ct, datatype=_BF16_2048_L1)   # weight
        buf_y = buffer(ct, datatype=_BF16_2048_L1)   # output
        buf_x = buffer(ct, datatype=_BF16_2048_L1)   # input
        buf_s = buffer(ct, datatype=_BF16_16_L1)     # scratch

        _emit_external_buffers((EMB_DIM,), (EMB_DIM,), (EMB_DIM,))

        # aie.mem block.
        @mem(ct)
        def _core_mem(block):
            # bb0: dma_start MM2S 0 (output y -> shim) -> bb1, fallthrough bb3
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

        # external_func decl + core body.
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
                # Acquire: x_ready (id=1), weight_ready (id=2), y_done (id=4).
                # Wait -- complement of mem DMA blocks:
                #   mem MM2S 0 (buf_y out): acquires id=0 (y_full),
                #                            releases id=1 (y_done)
                #   mem S2MM 0 (buf_x in):  acquires id=3 (x_avail),
                #                            releases id=2 (x_ready)
                #   mem S2MM 1 (buf_w in):  acquires id=5 (w_avail),
                #                            releases id=4 (w_ready)
                # So core acquires {y_done=id=1, x_ready=id=2, w_ready=id=4}
                # and releases {w_avail=id=5, y_full=id=0, x_avail=id=3}.
                use_lock(lk1, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk2, LockAction.AcquireGreaterEqual, value=1)
                use_lock(lk4, LockAction.AcquireGreaterEqual, value=1)
                rms_fn(buf_w, buf_x, buf_y, buf_s)
                use_lock(lk5, LockAction.Release, value=1)
                use_lock(lk0, LockAction.Release, value=1)
                use_lock(lk3, LockAction.Release, value=1)

        # Flows: shim<->compute (no mem tile).
        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        # Shim allocations (channel ids match cached IR).
        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        # Runtime sequence.
        @runtime_sequence(*rms_gemv_rope_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12):
            # in0 (MM2S 0) <- arg1 (x), 4 chunks of 512
            t0 = dma_configure_task_for(f"air_channel_{chans['in0']}")
            with bds(t0) as bd:
                with bd[0]:
                    dma_bd(arg1, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t0)
            # in1 (MM2S 1) <- arg0 (weight), 4 chunks of 512
            t1 = dma_configure_task_for(f"air_channel_{chans['in1']}")
            with bds(t1) as bd:
                with bd[0]:
                    dma_bd(arg0, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t1)
            # out (S2MM 0) -> arg2 (y), 4 chunks of 512
            t2 = dma_configure_task_for(f"air_channel_{chans['out']}", issue_token=True)
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
# RoPE segments (@rq_rope_seg, @rk_rope_seg).
# ---------------------------------------------------------------------------
# Both have identical structure -- single 1x1 herd:
#   1 shim tile (0,0), 1 compute tile (0,2)
#   6 locks (ids 5..0)
#   3 buffers of HEAD_DIM (64) bf16 in L1:
#     bufC = output       (rope result for this head, 64 elements)
#     bufB = freqs input  (sin/cos table chunk, 64 elements)
#     bufA = x input      (Q or K slice, 64 elements)
# Core loops scf.for(0..n_iters): rope(bufA, bufB, bufC, head_dim_i32)
#   n_iters = emb_dim/head_dim for rq (32), kv_dim/head_dim for rk (8)
# Channel routing per segment (cached numbers in _CHANNELS):
#   in0  MM2S 0 -> compute DMA 0  (S2MM 0 in compute, bufA = x slice)
#   in1  MM2S 1 -> compute DMA 1  (S2MM 1 in compute, bufB = freqs)
#   out  S2MM 0 <- compute DMA 0  (MM2S 0 in compute, bufC = y)
def _emit_rope_seg(sym: str, x_arg_idx: int, freqs_arg_idx: int,
                   out_arg_idx: int, vec_size: int) -> None:
    """Emit one rope segment device.

    ``vec_size`` is the host-side memref size for x/freqs/y in this
    segment -- ``emb_dim`` (=2048) for the Q stream, ``kv_dim`` (=512)
    for the K stream.  ``n_iters = vec_size // HEAD_DIM`` is the core
    loop trip count.
    """
    chans = _CHANNELS[sym]
    n_iters = vec_size // HEAD_DIM

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim = tile(0, 0)
        ct = tile(0, 2)

        # Locks (AIR emit order, ids 5..0).
        lk5 = lock(ct, lock_id=5, init=1)
        lk4 = lock(ct, lock_id=4, init=0)
        lk3 = lock(ct, lock_id=3, init=1)
        lk2 = lock(ct, lock_id=2, init=0)
        lk1 = lock(ct, lock_id=1, init=1)
        lk0 = lock(ct, lock_id=0, init=0)

        _BF16_64_L1 = bf16_memref(HEAD_DIM, memory_space=2)
        # AIR emits buffers as bufC (out), bufB (freqs), bufA (x).
        buf_y = buffer(ct, datatype=_BF16_64_L1)
        buf_f = buffer(ct, datatype=_BF16_64_L1)
        buf_x = buffer(ct, datatype=_BF16_64_L1)

        _emit_external_buffers((vec_size,), (vec_size,), (vec_size,))

        @mem(ct)
        def _core_mem(block):
            # MM2S 0: y -> shim
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

        # external_func + core body.
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
            # Acquires complement the mem DMA blocks (see rms_seg comment):
            #   id=1 y_done (output free), id=2 x_ready (input1 full),
            #   id=4 freqs_ready (input2 full).
            # Releases:
            #   id=3 x_avail, id=5 freqs_avail, id=0 y_full.
            for _outer in range_(_sys.maxsize):
                for _ in range_(0, n_iters, 1):
                    use_lock(lk1, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(lk2, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(lk4, LockAction.AcquireGreaterEqual, value=1)
                    rope_fn(buf_x, buf_f, buf_y, head_dim_c)
                    use_lock(lk3, LockAction.Release, value=1)
                    use_lock(lk5, LockAction.Release, value=1)
                    use_lock(lk0, LockAction.Release, value=1)

        # Flows.
        flow(shim, WireBundle.DMA, 0, ct, WireBundle.DMA, 0)
        flow(shim, WireBundle.DMA, 1, ct, WireBundle.DMA, 1)
        flow(ct, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

        shim_dma_allocation(f"air_channel_{chans['out']}", shim, DMAChannelDir.S2MM, 0)
        shim_dma_allocation(f"air_channel_{chans['in0']}", shim, DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{chans['in1']}", shim, DMAChannelDir.MM2S, 1)

        @runtime_sequence(*rms_gemv_rope_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_x = args[x_arg_idx]
            arg_f = args[freqs_arg_idx]
            arg_y = args[out_arg_idx]

            # rq variant: 4 chunks of 512 dims (for emb_dim=2048).
            # rk variant: single 512-element contig (for kv_dim=512).
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
            t2 = dma_configure_task_for(f"air_channel_{chans['out']}", issue_token=True)
            with bds(t2) as bd:
                with bd[0]:
                    dma_bd(arg_y, offset=0, len=vec_size, dimensions=dims)
                    EndOp()
            dma_start_task(t2)
            dma_await_task(t2)
            dma_free_task(t0)
            dma_free_task(t1)


# ---------------------------------------------------------------------------
# GEMV matvec segments (@q_matvec_bf16_0, @k_matvec_bf16_0, @v_matvec_bf16_0).
# ---------------------------------------------------------------------------
# Each matvec segment is structurally a shape-shrunk version of one
# partition of ``lm_head_gemv``. The only knobs that vary across q/k/v:
#   - weight host shape: 2048x2048 (Q) vs 512x2048 (K/V)
#   - output host shape: 2048 (Q) vs 512 (K/V)
#   - host arg indices into the 13-arg signature
#   - number of outer iters: 2 (Q, 2048 rows / 1024 rows-per-outer)
#     vs 1 (K, V; 512 rows / 1024 rows-per-outer rounded up = 1)
#   - input repeat_count per outer iter (15 for K/V, 31 for Q -- see notes)
#   - channel numbers (passed in via ``chan_numbers``)
def _emit_matvec_seg(sym: str, weight_arg_idx: int, output_arg_idx: int,
                     out_rows: int, pingpong_w: bool = False) -> None:
    """Emit one matvec_bf16_0 segment device.

    ``out_rows`` is the host-side row count for this segment's output
    (= ``EMB_DIM`` for Q, ``KV_DIM`` for K/V). This determines how many
    "outer" iterations the runtime_sequence loops over.

    Per-outer iteration:
      * 8 weight MM2S tasks (one per shim column)
      * 1 input MM2S task on shim 0, ``repeat_count = 8 * outer_inner_iters - 1``
      * 8 output S2MM tasks (one per shim column, issue_token = true)

    ``pingpong_w=True`` doubles the L1 W buffer (wb0 + wb1) and changes the
    W DMA BD chain into a 2-BD ring that alternates wb0/wb1. ``w_avail`` is
    initialised to 2 so the L1 DMA can stay one tile ahead of compute. The
    inner K_TILE-loop is unrolled (2 iters when M_TILE/K_TILE = 2) so iter 0
    consumes wb0 and iter 1 consumes wb1. Other roles (X / Y) unchanged.
    """
    chans = _CHANNELS[sym]
    # Per-partition row-band:
    #   8 cols * 8 rows-per-mini-row * 16 mini-rows (K=128 inner chunks * 2048)
    # The weight DMA stride pattern is fixed by AIR: 8 mini-rows of 32x512
    # contig (size=8 stride=131072 for "outer cols" of 8 row banks; in the
    # AIR-stitched view this is just one shape).
    # outputs:
    #   8 cols * 8 rows-each, output col_stride = M_TILE = 8.
    #   outer_stride for outputs is the per-outer row count delivered to
    #   the host buffer.  For K/V: each col writes [size=8, stride=64,
    #   size=8, stride=1] = 64 elements per task; 8 cols * 64 = 512 total
    #   (= KV_DIM) -> only one outer iter.  For Q: each col writes
    #   [size=16, stride=64, size=8, stride=1] = 128 elements per task;
    #   8 cols * 128 = 1024 per outer iter, with 2 outer iters -> 2048
    #   (= EMB_DIM).

    if out_rows == KV_DIM:
        # K/V: 512-element output, 1 outer iter.
        n_outer = 1
        y_dims = [(8, 64), (8, 1)]
        y_len = 64
        x_repeat_count = 15
        # Weight stride pattern: 8 mini-rows of 32x512 contig.
        w_dims = [(8, 131072), (32, 512), (512, 1)]
        w_len = 131072
        weight_col_stride = M_TILE * EMB_DIM  # 16_384
        weight_outer_stride = 0  # unused (only 1 outer)
        output_col_stride = M_TILE  # 8
        output_outer_stride = 0  # unused
    else:
        # Q: 2048-element output, 2 outer iters.
        assert out_rows == EMB_DIM
        n_outer = 2
        y_dims = [(16, 64), (8, 1)]
        y_len = 128
        x_repeat_count = 31
        # Weight stride pattern: 16 mini-rows of 32x512 contig.
        w_dims = [(16, 131072), (32, 512), (512, 1)]
        w_len = 262144
        weight_col_stride = M_TILE * EMB_DIM  # 16_384
        # Outer stride spans the row band one core covers per outer iter.
        # AIR: 8 cores * 8 mini-rows per outer * 2048 emb = 131072 per col
        # multiplied across cols gives 8 * 16384 = 131072 row-band height
        # in bf16 elements... actually the dma_bd offset jumps by 2_097_152
        # = ROWS_PER_OUTER * EMB_DIM = 1024 * 2048.
        weight_outer_stride = 1024 * EMB_DIM
        output_col_stride = M_TILE  # 8
        output_outer_stride = 1024

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Mem tile locks (4 ids 3..0, in AIR's descending-col order).
        # ids match AIR: 3 = w_dma_done(1), 2 = w_ready(0),
        # 1 = y_done(1), 0 = y_ready(0).
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
        # When pingpong_w, w_avail is a 2-counting semaphore so the producer
        # (memtile->L1 W DMA) can stage two tiles before any compute drains.
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

        # Buffers (memrefs must be built inside the device ctx).
        _W_L1_TY = bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
        _Y_L2_TY = bf16_memref(1, M_TILE, memory_space=1)

        # Mem tile buffers, in AIR's descending-col order for weight then output.
        mem_buf_w = {}
        mem_buf_y = {}
        for col in reversed(range(N_COLS)):
            mem_buf_w[col] = buffer(mem_tiles[col], datatype=_W_L2_TY)
        for col in reversed(range(N_COLS)):
            mem_buf_y[col] = buffer(mem_tiles[col], datatype=_Y_L2_TY)

        # Compute tile buffers (output y, weight w, input x) -- descending col.
        # When pingpong_w, allocate a second W L1 slot per core (wb0/wb1).
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

        # External buffers.
        _emit_external_buffers((out_rows, EMB_DIM), (EMB_DIM,), (out_rows,))

        # Compute mem + core blocks. AIR emits in descending col order.
        # Declare external_funcs once; aie-assign-core-link-files routes
        # the link_with attribute onto every core that calls them.
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
                            # Ping-pong: M_TILE/K_TILE must be 2. Unroll the
                            # K-loop so iter 0 reads wb0, iter 1 reads wb1.
                            # DMA ring fills (wb0, wb1, wb0, wb1, ...) and the
                            # init=2 w_avail keeps producer one tile ahead.
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

        # Shim allocations. AIR emits 8x S2MM then 8x MM2S(weight) then 1 MM2S(input).
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
        @runtime_sequence(*rms_gemv_rope_host_arg_types(), sym_name=f"{sym}_sequence")
        def _seq(*args):
            arg_x = args[2]  # broadcast input (post-rmsnorm) is always arg2
            arg_w = args[weight_arg_idx]
            arg_y = args[output_arg_idx]

            for outer in range(n_outer):
                # 8 weight MM2S tasks.
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

                # 1 input MM2S task.
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

                # 8 output S2MM tasks.
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

                # Awaits in reverse order (matches AIR).
                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device() -> None:
    """Emit the unnamed top-level dispatcher device.

    The dispatcher carries the outer ``aie.runtime_sequence @rms_gemv_rope``
    that fires the 6 segment sequences in topological order:
        r_rms_seg -> q_matvec -> k_matvec -> v_matvec -> rq_rope -> rk_rope.
    All segments share the same 13-arg signature.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *rms_gemv_rope_host_arg_types(),
            sym_name="rms_gemv_rope",
        )
        def _outer(*args):
            for sym in ("r_rms_seg", "q_matvec_bf16_0", "k_matvec_bf16_0",
                        "v_matvec_bf16_0", "rq_rope_seg", "rk_rope_seg"):
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
def build_rms_gemv_rope_module(emb_dim: int = EMB_DIM,
                               kv_dim: int = KV_DIM,
                               n_heads: int = 32,
                               n_kv_heads: int = 8,
                               head_dim: int = HEAD_DIM) -> str:
    """Build the RMS+GEMV+RoPE ``aie/aiex``-dialect module.

    All dimension args must match the Llama-3.2-1B values; the cached
    AIR layout is shape-specialized for these. Other values raise
    ``ValueError``.
    """
    if emb_dim != EMB_DIM or kv_dim != KV_DIM or head_dim != HEAD_DIM:
        raise ValueError(
            f"rms_gemv_rope builder is fixed to emb_dim={EMB_DIM}, "
            f"kv_dim={KV_DIM}, head_dim={HEAD_DIM}; got "
            f"emb_dim={emb_dim}, kv_dim={kv_dim}, head_dim={head_dim}."
        )
    del n_heads, n_kv_heads

    with mlir_mod_ctx() as ctx:
        # AIR emits devices in this order (rope-K first; r_rms last).
        _emit_rope_seg("rk_rope_seg",
                       x_arg_idx=6, freqs_arg_idx=10, out_arg_idx=12,
                       vec_size=KV_DIM)
        _emit_rope_seg("rq_rope_seg",
                       x_arg_idx=4, freqs_arg_idx=9, out_arg_idx=11,
                       vec_size=EMB_DIM)
        _emit_matvec_seg("v_matvec_bf16_0",
                         weight_arg_idx=7, output_arg_idx=8,
                         out_rows=KV_DIM)
        # pingpong_w off for K_TILE=8 experiment; M_TILE/K_TILE==1 makes
        # the L1 PP unroll assertion fire anyway.
        _emit_matvec_seg("k_matvec_bf16_0",
                         weight_arg_idx=5, output_arg_idx=6,
                         out_rows=KV_DIM)
        _emit_matvec_seg("q_matvec_bf16_0",
                         weight_arg_idx=3, output_arg_idx=4,
                         out_rows=EMB_DIM)
        _emit_r_rms_seg()
        _emit_dispatcher_device()
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
    text = build_rms_gemv_rope_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
