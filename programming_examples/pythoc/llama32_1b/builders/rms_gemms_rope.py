# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b prefill RMS+GEMMS+RoPE kernel.

Phase 4.5e scope: all 7 devices are emitted from placed IRON --
``@r_weighted_rms_norm_seg``, ``@rk_rope_seg``, ``@rq_rope_seg``,
``@v_matmul_seg``, ``@k_matmul_seg``, ``@q_matmul_seg``, and the outer
unnamed dispatcher ``aie.device(npu2)`` that hosts
``aiex.runtime_sequence @rms_gemms_rope``.  Only the leading
``module { ... }`` wrapper text and inter-device whitespace still come
from the cached ``rms_gemms_rope.npu.air.mlir`` text via splice.

Splice mechanism::

    cached_text =
      aie.device @rk_rope_seg { ... }              <-- REPLACED
      aie.device @rq_rope_seg { ... }              <-- REPLACED
      aie.device @v_matmul_seg { ... }             <-- REPLACED
      aie.device @k_matmul_seg { ... }             <-- REPLACED
      aie.device @q_matmul_seg { ... }             <-- REPLACED
      aie.device @r_weighted_rms_norm_seg { ... }  <-- REPLACED
      aie.device (dispatcher) { ... }              <-- REPLACED

The cached file's structure for ``r_weighted_rms_norm_seg`` (1230 lines)
contains a 1x8 herd of compute tiles (col 0..7, row 2) executing inline
RMSNorm math (``arith.addf`` sum-of-squares -> ``math.rsqrt`` ->
``arith.mulf`` rescale) with NO call to an external kernel object --
unlike decode's ``@r_rms_seg`` which calls ``rms_norm_2048_bf16.o``.

Per-tile structure (8 cores all identical, per-col c=0..7):
  * 6 locks at lock_ids {5,4,3,2,1,0} with init {2,0,1,0,2,0}
  * 7 buffers in L1 (memory_space=2):
      weight (2048xbf16), x_pong (2048), scratch_pong (16xbf16),
      y_pong (2048), x_ping (2048), y_ping (2048), scratch_ping (16)
  * aie.mem block with 3 DMA channels:
      MM2S 0 (y out): ping-pong y_ping, y_pong, lock pair (id=0,1)
      S2MM 0 (weight): single buf weight, lock pair (id=3,2)
      S2MM 1 (x in):  ping-pong x_ping, x_pong, lock pair (id=5,4)
  * aie.core body: cf.br ^bb1 with infinite loop. Acquires id=2 once,
    then for 128 outer iters (256 step 2), inside each:
      acquire id=1 twice (y_done x2), id=4 once (x_ready), compute iter A
      release id=5 (x_avail);
      acquire id=4 once, compute iter B
      release id=5, id=0 twice (y_full x2).
    After outer loop release id=3 (w_avail), branch back to bb1.

Inline math per inner iter (within outer step=2 loop):
  1. Zero scratch (16xbf16).
  2. For i in 0..2048 step 16: read x[i:i+16], square -> intermediate
     (uses y buf as temp), add to scratch.
  3. vector.reduction <add> -> scalar bf16 sum.
  4. divf sum / 2048.0, addf eps (1.001360e-05), extf to f32,
     math.rsqrt, truncf back to bf16, broadcast to vector<16xbf16>.
  5. For i in 0..2048 step 16: x[i:i+16] * rsqrt_broadcast * weight[i:i+16]
     -> y[i:i+16].

Flows + shim allocations (matches cached exactly):
  * 8 flows shim_0_0 DMA 0 -> tile_X_2 DMA 0  (gamma weight broadcast)
  * 1 flow shim_0_0 DMA 1 -> tile_0_2 DMA 1
  * 7 flows shim_C_0 DMA 0 -> tile_C_2 DMA 1 for C in 1..7  (x input per col)
  * 8 flows tile_C_2 DMA 0 -> shim_C_0 DMA 0  (y output per col)

  * 17 shim_dma_allocations (8 S2MM outputs, 1 MM2S gamma broadcast,
    8 MM2S X per col; col 0 uses MM2S 1, cols 1-7 use MM2S 0).
"""

from __future__ import annotations

from pathlib import Path
import sys as _sys

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
from aie.ir import AffineDimExpr, AffineMap, InsertionPoint, UnitAttr

from ._emit import attach_loop_annotation_to_all_scf_for, bf16_memref, bf16_np


# ---------------------------------------------------------------------------
# Constants matching the cached AIR-stitched IR for Llama-3.2-1B prefill.
# ---------------------------------------------------------------------------
EMB_DIM = 2048             # model hidden size
SEQ_LEN = 2048             # prefill sequence length
KV_DIM = 512               # n_kv_heads * head_dim
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
N_COLS = 8                 # 8 compute tile columns
VEC_LANES = 16             # bf16 vector lane width
SCRATCH_LEN = 16           # scratch buffer length (= VEC_LANES)
N_OUTER_STEPS = 256        # outer scf.for upper bound
OUTER_STEP = 2             # step (so 128 inner iters)

# Host arg layout for r_weighted_rms_norm_seg_sequence (13 args).
# Matches the cached IR's runtime_sequence signature:
#   arg0 :  2048x2048 bf16  Q weight (unused by this device)
#   arg1 :  2048      bf16  RMSNorm gamma (broadcast)
#   arg2 :  2048x2048 bf16  X input (seq_len x emb_dim, sliced per col)
#   arg3 :  2048x2048 bf16  K weight stack
#   arg4 :  2048x2048 bf16  V weight stack
#   arg5..arg8 : 2048x512 bf16  Various intermediate kv buffers
#   arg9 : 4194304  bf16   Q output
#   arg10: 1048576  bf16   KV output
#   arg11: 2048x2048 bf16  Output (normed X delivered here by this device)
#   arg12: 2048x512 bf16
#
# Note: the cached uses arg2 for both input and output of this device --
# the device writes normed X back to arg2 in-place.  (host-side: arg2 is
# the X buffer that the GEMM devices read as input.)
def _rms_gemms_rope_host_arg_types():
    return [
        bf16_np(EMB_DIM, EMB_DIM),       # arg0
        bf16_np(EMB_DIM),                # arg1 (gamma)
        bf16_np(EMB_DIM, EMB_DIM),       # arg2 (X + normed_x output)
        bf16_np(EMB_DIM, EMB_DIM),       # arg3
        bf16_np(EMB_DIM, EMB_DIM),       # arg4
        bf16_np(EMB_DIM, KV_DIM),        # arg5
        bf16_np(EMB_DIM, KV_DIM),        # arg6
        bf16_np(EMB_DIM, KV_DIM),        # arg7
        bf16_np(EMB_DIM, KV_DIM),        # arg8
        bf16_np(4194304,),               # arg9
        bf16_np(1048576,),               # arg10
        bf16_np(EMB_DIM, EMB_DIM),       # arg11
        bf16_np(EMB_DIM, KV_DIM),        # arg12
    ]


# ---------------------------------------------------------------------------
# Emit the @r_weighted_rms_norm_seg device.
# ---------------------------------------------------------------------------
def _emit_r_weighted_rms_norm_seg() -> None:
    """Emit the placed-IRON r_weighted_rms_norm_seg device.

    Must be called inside an mlir_mod_ctx(); registers one
    ``aie.device(npu2) @r_weighted_rms_norm_seg`` op.
    """
    from aie.dialects import memref as memref_dialect
    from aie.dialects import vector as vector_dialect
    from aie.dialects._vector_enum_gen import CombiningKind
    from aie.dialects import math as math_dialect
    from aie.extras import types as T

    @device(AIEDevice.npu2, sym_name="r_weighted_rms_norm_seg")
    def _dev():
        # 8 shim tiles + 8 compute tiles.
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Locks per compute tile -- 6 ids descending (5..0), in AIR's order.
        # Init values per cached IR:
        #   id=5 (x_avail)   init=2
        #   id=4 (x_ready)   init=0
        #   id=3 (w_avail)   init=1
        #   id=2 (w_ready)   init=0
        #   id=1 (y_done)    init=2
        #   id=0 (y_full)    init=0
        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "x_avail": lock(ct, lock_id=5, init=2),
                "x_ready": lock(ct, lock_id=4, init=0),
                "w_avail": lock(ct, lock_id=3, init=1),
                "w_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=2),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffers per compute tile.  AIR-emit order top->bottom:
        #   weight, x_pong, scratch_pong, y_pong, x_ping, y_ping, scratch_ping.
        # Buffers are declared in descending column order in the cached IR
        # (col 7 first, then col 6, ..., col 0).
        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        _BF16_16_L1 = bf16_memref(SCRATCH_LEN, memory_space=2)

        core_buf = {col: {} for col in range(N_COLS)}
        for col in reversed(range(N_COLS)):
            ct = compute_tiles[col]
            core_buf[col]["weight"]       = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["x_pong"]       = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["scratch_pong"] = buffer(ct, datatype=_BF16_16_L1)
            core_buf[col]["y_pong"]       = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["x_ping"]       = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["y_ping"]       = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["scratch_ping"] = buffer(ct, datatype=_BF16_16_L1)

        # External buffers (opaque AIR metadata, kept for structural diff).
        external_buffer(bf16_np(EMB_DIM,), name="__air_external_buffer")
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_1")
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_2")

        # aie.mem block per compute tile.  Structure mirrors cached:
        #   MM2S 0 (y out): bb1=y_ping, bb2=y_pong; lock id=0 acq, id=1 rel
        #   S2MM 0 (weight in): bb5=weight, self-loop; lock id=3 acq, id=2 rel
        #   S2MM 1 (x in): bb7=x_ping, bb8=x_pong; lock id=5 acq, id=4 rel
        def _make_core_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                # MM2S 0: y out (ping-pong)
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[4])
                with block[1]:
                    use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["y_ping"], offset=0, len=EMB_DIM)
                    use_lock(_cl["y_done"], LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["y_pong"], offset=0, len=EMB_DIM)
                    use_lock(_cl["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    EndOp()
                with block[4]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[6])
                with block[5]:
                    use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["weight"], offset=0, len=EMB_DIM)
                    use_lock(_cl["w_ready"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[6]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[3])
                with block[7]:
                    use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["x_ping"], offset=0, len=EMB_DIM)
                    use_lock(_cl["x_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["x_pong"], offset=0, len=EMB_DIM)
                    use_lock(_cl["x_ready"], LockAction.Release, value=1)
                    next_bd(block[7])

        for col in reversed(range(N_COLS)):
            _make_core_mem(compute_tiles[col], core_locks[col], core_buf[col])

        # aie.core body per compute tile.  See top-of-file docstring for the
        # high-level structure.  Per-iteration RMSNorm math is inline.
        def _make_core_body(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                # Constants.
                cst_zero_bf16 = arith.constant(0.0, T.bf16())
                cst_norm_div = arith.constant(2048.0, T.bf16())
                cst_eps = arith.constant(1.001360e-05, T.bf16())
                # Vector zero constant (dense<0.0> : vector<16xbf16>).
                vec16_ty = T.vector(VEC_LANES, T.bf16())
                np_zero = np.zeros((VEC_LANES,), dtype=bfloat16)
                cst_vec_zero = arith.constant(np_zero, vec16_ty)
                # Identity affine map for vector.transfer_read/write.
                perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                c0 = arith.constant(0, T.index())

                def _rms_norm_pass(x_buf, w_buf, y_buf, s_buf):
                    """Inline RMSNorm: y = x * rsqrt(sum(x*x)/N + eps) * w."""
                    # Zero the scratch (16xbf16).
                    vector_dialect.transfer_write(
                        None, cst_vec_zero, s_buf, [c0],
                        permutation_map=perm, in_bounds=[True])
                    # Sum-of-squares loop.
                    for i in range_(0, EMB_DIM, VEC_LANES):
                        sub_x = memref_dialect.subview(x_buf, [i], [VEC_LANES], [1])
                        sub_y = memref_dialect.subview(y_buf, [i], [VEC_LANES], [1])
                        v_x = vector_dialect.transfer_read(
                            vec16_ty, sub_x, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_sq = arith.mulf(v_x, v_x)
                        vector_dialect.transfer_write(
                            None, v_sq, sub_y, [c0],
                            permutation_map=perm, in_bounds=[True])
                        v_sq2 = vector_dialect.transfer_read(
                            vec16_ty, sub_y, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_acc = vector_dialect.transfer_read(
                            vec16_ty, s_buf, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_sum = arith.addf(v_acc, v_sq2)
                        vector_dialect.transfer_write(
                            None, v_sum, s_buf, [c0],
                            permutation_map=perm, in_bounds=[True])
                    # Scalar reduction + rsqrt.
                    v_final = vector_dialect.transfer_read(
                        vec16_ty, s_buf, [c0],
                        permutation_map=perm, padding=cst_zero_bf16,
                        in_bounds=[True])
                    s_sum = vector_dialect.reduction(
                        T.bf16(), CombiningKind.ADD, v_final)
                    s_mean = arith.divf(s_sum, cst_norm_div)
                    s_meps = arith.addf(s_mean, cst_eps)
                    s_f32 = arith.extf(T.f32(), s_meps)
                    s_rsq = math_dialect.rsqrt(s_f32)
                    s_rsq_bf = arith.truncf(T.bf16(), s_rsq)
                    v_rsq = vector_dialect.broadcast(vec16_ty, s_rsq_bf)
                    # Rescale loop: y[i] = x[i] * rsqrt * w[i].
                    for i in range_(0, EMB_DIM, VEC_LANES):
                        sub_x = memref_dialect.subview(x_buf, [i], [VEC_LANES], [1])
                        sub_w = memref_dialect.subview(w_buf, [i], [VEC_LANES], [1])
                        sub_y = memref_dialect.subview(y_buf, [i], [VEC_LANES], [1])
                        v_x = vector_dialect.transfer_read(
                            vec16_ty, sub_x, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_w = vector_dialect.transfer_read(
                            vec16_ty, sub_w, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_scaled = arith.mulf(v_x, v_rsq)
                        v_out = arith.mulf(v_scaled, v_w)
                        vector_dialect.transfer_write(
                            None, v_out, sub_y, [c0],
                            permutation_map=perm, in_bounds=[True])

                # Infinite outer loop (cf.br ^bb1 in cached IR).
                for _outer in range_(_sys.maxsize):
                    # Acquire weight once at top of outer.
                    use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                    # 128 inner iters (256 step 2).
                    for _inner in range_(0, N_OUTER_STEPS, OUTER_STEP):
                        # Iter A: y_done x2, x_ready x1, compute, x_avail x1.
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        _rms_norm_pass(
                            _bufs["x_ping"], _bufs["weight"],
                            _bufs["y_ping"], _bufs["scratch_ping"])
                        use_lock(_cl["x_avail"], LockAction.Release, value=1)
                        # Iter B: x_ready x1, compute, x_avail x1, y_full x2.
                        use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        _rms_norm_pass(
                            _bufs["x_pong"], _bufs["weight"],
                            _bufs["y_pong"], _bufs["scratch_pong"])
                        use_lock(_cl["x_avail"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
                    # Release weight slot after outer-iter inner loop.
                    use_lock(_cl["w_avail"], LockAction.Release, value=1)

        for col in reversed(range(N_COLS)):
            _make_core_body(compute_tiles[col], core_locks[col], core_buf[col])

        # Flows.  See top-of-file docstring for the layout.
        #   shim_0_0 DMA 0 -> tile_X_2 DMA 0  (gamma broadcast, 8 flows)
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 0)
        #   shim_0_0 DMA 1 -> tile_0_2 DMA 1
        flow(shim_tiles[0], WireBundle.DMA, 1,
             compute_tiles[0], WireBundle.DMA, 1)
        #   shim_C_0 DMA 0 -> tile_C_2 DMA 1 for C in 1..7
        for col in range(1, N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 1)
        #   tile_C_2 DMA 0 -> shim_C_0 DMA 0  (y output per col)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)

        # Shim allocations.  Cached order:
        #   8 S2MM (y out) air_channel_2_C
        #   1 MM2S 0 on shim_0_0  air_channel_0  (gamma broadcast)
        #   1 MM2S 1 on shim_0_0  air_channel_1_0
        #   7 MM2S 0 on shim_C_0  air_channel_1_C for C in 1..7
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_2_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        shim_dma_allocation(
            "air_channel_0", shim_tiles[0], DMAChannelDir.MM2S, 0)
        shim_dma_allocation(
            "air_channel_1_0", shim_tiles[0], DMAChannelDir.MM2S, 1)
        for col in range(1, N_COLS):
            shim_dma_allocation(
                f"air_channel_1_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)

        # Runtime sequence.  Layout (per cached IR):
        #   t_gamma  = MM2S 0 on shim_0_0:  arg1 -> all 8 cores (broadcast)
        #             dims = [(4, 512), (512, 1)]  len=2048
        #   t_x_C    = MM2S X on shim_C_0:  arg2 (X input slice)
        #             offset = C * (2 * 262144)  len=524288
        #             dims = [(2, 262144), (512, 512), (512, 1)]
        #   t_y_C    = S2MM 0 on shim_C_0:  arg2 (Y output, written in-place
        #             at the same offset as the X slice -- cached IR uses arg2)
        #             offset = C * (2 * 262144)  len=524288
        #             dims = [(2, 262144), (512, 512), (512, 1)]
        # Then 9 dma_free_task (gamma + 8 inputs), 8 dma_await_task (outputs).
        @runtime_sequence(*_rms_gemms_rope_host_arg_types(),
                          sym_name="r_weighted_rms_norm_seg_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12):
            del arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12

            # Gamma broadcast (single task, shim_0_0 MM2S 0). Reads arg1.
            t_gamma = dma_configure_task_for("air_channel_0")
            with bds(t_gamma) as bd:
                with bd[0]:
                    dma_bd(arg1, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_gamma)

            # 8 X input tasks (per col).  X lives in arg0 (NOT arg2 -- the
            # cached dispatcher passes the seq_len-major input here).
            x_tasks = []
            for col in range(N_COLS):
                offset = col * 524288
                t = dma_configure_task_for(f"air_channel_1_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg0, offset=offset, len=524288,
                               dimensions=[(2, 262144), (512, 512), (512, 1)])
                        EndOp()
                dma_start_task(t)
                x_tasks.append(t)

            # 8 Y output tasks (per col).  Normed X output to arg2.
            y_tasks = []
            for col in range(N_COLS):
                offset = col * 524288
                t = dma_configure_task_for(f"air_channel_2_{col}",
                                            issue_token=True)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg2, offset=offset, len=524288,
                               dimensions=[(2, 262144), (512, 512), (512, 1)])
                        EndOp()
                dma_start_task(t)
                y_tasks.append(t)

            # Free input tasks (gamma + 8 X tasks).
            dma_free_task(t_gamma)
            for t in x_tasks:
                dma_free_task(t)
            # Await output tasks.
            for t in y_tasks:
                dma_await_task(t)


# ---------------------------------------------------------------------------
# RoPE devices (@rk_rope_seg, @rq_rope_seg).
# ---------------------------------------------------------------------------
# Phase 4.5b: each RoPE device is an 8x1 herd (8 compute cols, 1 row).  Per
# core:
#   * 6 locks (ids 5..0), init pattern: (2, 0, 2, 0, 2, 0).
#       - id=5 init=2  s1_avail  (freqs slot semaphore, 2 ping-pong slots)
#       - id=4 init=0  s1_ready  (freqs data full)
#       - id=3 init=2  s0_avail  (x slot semaphore)
#       - id=2 init=0  s0_ready  (x data full)
#       - id=1 init=2  m0_done   (y slot semaphore)
#       - id=0 init=0  m0_full   (y data full)
#   * 6 buffers of 64xbf16 in L1 (HEAD_DIM elements per buffer).  AIR emit
#     order (per tile, descending sym -- highest first):
#       s0_pong, s1_pong, m0_pong, s0_ping, m0_ping, s1_ping.
#     Logical mapping:
#       m0_ping/m0_pong -> y output ping-pong
#       s0_ping/s0_pong -> x input ping-pong
#       s1_ping/s1_pong -> freqs input ping-pong
#   * aie.mem block: 3 DMA channels (ping-pong each).
#       MM2S 0: y_ping/y_pong out  (acq id=0, rel id=1)
#       S2MM 0: x_ping/x_pong in   (acq id=3, rel id=2)
#       S2MM 1: f_ping/f_pong in   (acq id=5, rel id=4)
#   * aie.core: outer cf.br infinite loop wrapping a scf.for from 0 to
#     N_ROPE_ITERS step 2 -- per inner iter the core acquires the input
#     ready locks, calls @rope twice (ping then pong, ping-pong on each
#     DMA channel pair).  The full iter releases 2 x y_done at the end
#     (one per ping/pong y_buf produced).
#
# Iter pattern (matches cached op-for-op):
#       scf.for arg0 = 0 to N_ROPE_ITERS step 2:
#         acq m0_done x2
#         acq s0_ready x1
#         acq s1_ready x1
#         rope(x_ping, f_ping, y_ping, head_dim_i32)
#         rel s1_avail x1
#         rel s0_avail x1
#         acq s0_ready x1
#         acq s1_ready x1
#         rope(x_pong, f_pong, y_pong, head_dim_i32)
#         rel s1_avail x1
#         rel s0_avail x1
#         rel m0_full x2
#       cf.br ^bb1
#
# N_ROPE_ITERS:
#   * rk_rope_seg: 2048   (each core processes 1024 head-chunks, ping/pong
#                          -> 2048/2 = 1024 inner iters; outer step=2)
#   * rq_rope_seg: 8192   (each core processes 4096 head-chunks)
#
# Flows per device (identical layout for rk and rq, only shim allocations
# differ):
#   * shim_C_0 DMA 0 -> tile_C_2 DMA 0  (x in, 8 flows, C=0..7)
#   * shim_C_0 DMA 1 -> tile_C_2 DMA 1  (freqs in, 8 flows, C=0..7)
#   * tile_C_2 DMA 0 -> shim_C_0 DMA 0  (y out, 8 flows, C=0..7)
#
# Shim allocations (channel-name convention from cached AIR text):
#   rk_rope_seg:
#       air_channel_54_C  (MM2S 0 on shim_C_0)  x in, sources arg6
#       air_channel_55_C  (MM2S 1 on shim_C_0)  freqs in, sources arg10
#       air_channel_56_C  (S2MM 0 on shim_C_0)  y out, sinks to arg12
#   rq_rope_seg:
#       air_channel_51_C  (MM2S 0 on shim_C_0)  x in, sources arg4
#       air_channel_52_C  (MM2S 1 on shim_C_0)  freqs in, sources arg9
#       air_channel_53_C  (S2MM 0 on shim_C_0)  y out, sinks to arg11
#
# Runtime sequence (per device):
#   * 8 x_tasks (1 per col)
#   * 8 freqs_tasks (1 per col)
#   * 8 y_tasks (1 per col, with issue_token=True)
#   * Then dma_free_task for all 16 inputs (x then freqs), then
#     dma_await_task for the 8 outputs.
#
# Per-col DMA strides:
#   rk_rope_seg: arg6/arg10/arg12 are memref<2048x512xbf16> / 1048576xbf16.
#     Each col gets 131072 elements (= 256 rows x 512 cols) at offset
#     col * 131072.  Dimensions: [(256, 512), (512, 1)].
#   rq_rope_seg: arg4/arg9/arg11 are memref<2048x2048xbf16> / 4194304xbf16.
#     Each col gets 524288 elements (= 2 x 262144) at offset col * 524288.
#     Dimensions: [(2, 262144), (512, 512), (512, 1)].
HEAD_DIM_ROPE = 64           # rope kernel chunk size (= HEAD_DIM)
_RK_ROPE_ITERS = 2048        # outer scf.for upper bound for rk
_RQ_ROPE_ITERS = 8192        # outer scf.for upper bound for rq


def _emit_rope_device(sym: str, *,
                       n_rope_iters: int,
                       chan_in_x: int,
                       chan_in_freqs: int,
                       chan_out: int,
                       x_arg_idx: int,
                       freqs_arg_idx: int,
                       out_arg_idx: int,
                       x_per_col_len: int,
                       x_per_col_offset_step: int,
                       x_dims: list,
                       freqs_per_col_len: int,
                       freqs_per_col_offset_step: int,
                       freqs_dims: list,
                       out_per_col_len: int,
                       out_per_col_offset_step: int,
                       out_dims: list,
                       ext_buf_dtype_shapes: list) -> None:
    """Emit one ``aie.device(npu2) @<sym> { ... }`` RoPE device.

    Must be called inside an active ``mlir_mod_ctx()``.

    All numeric args follow the cached AIR-stitched IR for Llama-3.2-1B
    prefill; see the docstring at the top of this section for the
    structural layout.  ``ext_buf_shapes`` is a list of three shape
    tuples for the 3 ``__air_external_buffer`` decls (matches cached).
    """
    from aie.dialects import memref as memref_dialect  # noqa: F401  (loaded for type ctx)
    from aie.extras import types as T

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Locks per compute tile, ids 5..0 with init (2,0,2,0,2,0).
        # Lock-id semantics:
        #   id=5 (init=2) freqs_avail   -- mem S2MM 1 acq
        #   id=4 (init=0) freqs_ready   -- mem S2MM 1 rel; core acq
        #   id=3 (init=2) x_avail       -- mem S2MM 0 acq
        #   id=2 (init=0) x_ready       -- mem S2MM 0 rel; core acq
        #   id=1 (init=2) y_done        -- mem MM2S 0 rel; core acq
        #   id=0 (init=0) y_full        -- mem MM2S 0 acq; core rel
        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "freqs_avail": lock(ct, lock_id=5, init=2),
                "freqs_ready": lock(ct, lock_id=4, init=0),
                "x_avail":     lock(ct, lock_id=3, init=2),
                "x_ready":     lock(ct, lock_id=2, init=0),
                "y_done":      lock(ct, lock_id=1, init=2),
                "y_full":      lock(ct, lock_id=0, init=0),
            }

        # Buffers per compute tile.  AIR emit order top->bottom (descending
        # sym-id, descending col): bufN..bufN-5.  Logical roles:
        #   slot 0 (highest sym): x_pong   (s0_pong)
        #   slot 1:               freqs_pong (s1_pong)
        #   slot 2:               y_pong   (m0_pong)
        #   slot 3:               x_ping   (s0_ping)
        #   slot 4:               y_ping   (m0_ping)
        #   slot 5 (lowest sym):  freqs_ping (s1_ping)
        _BF16_HD_L1 = bf16_memref(HEAD_DIM_ROPE, memory_space=2)
        core_buf = {col: {} for col in range(N_COLS)}
        for col in reversed(range(N_COLS)):
            ct = compute_tiles[col]
            core_buf[col]["x_pong"]     = buffer(ct, datatype=_BF16_HD_L1)
            core_buf[col]["freqs_pong"] = buffer(ct, datatype=_BF16_HD_L1)
            core_buf[col]["y_pong"]     = buffer(ct, datatype=_BF16_HD_L1)
            core_buf[col]["x_ping"]     = buffer(ct, datatype=_BF16_HD_L1)
            core_buf[col]["y_ping"]     = buffer(ct, datatype=_BF16_HD_L1)
            core_buf[col]["freqs_ping"] = buffer(ct, datatype=_BF16_HD_L1)

        # External buffers (3 -- opaque AIR metadata, kept for diff parity).
        _ext_names = [
            "__air_external_buffer",
            "__air_external_buffer_1",
            "__air_external_buffer_2",
        ]
        for nm, shp in zip(_ext_names, ext_buf_dtype_shapes):
            external_buffer(bf16_np(*shp), name=nm)

        # aie.mem block per compute tile.  Bb layout matches cached:
        #   bb0 -> dma_start MM2S 0 (y out), chain to bb4
        #     bb1: y_ping, acq y_full, rel y_done -> bb2
        #     bb2: y_pong, acq y_full, rel y_done -> bb1
        #   bb3 -> END
        #   bb4 -> dma_start S2MM 0 (x in), chain to bb7
        #     bb5: x_ping, acq x_avail, rel x_ready -> bb6
        #     bb6: x_pong, acq x_avail, rel x_ready -> bb5
        #   bb7 -> dma_start S2MM 1 (freqs in), chain to bb3
        #     bb8: f_ping, acq freqs_avail, rel freqs_ready -> bb9
        #     bb9: f_pong, acq freqs_avail, rel freqs_ready -> bb8
        def _make_core_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[4])
                with block[1]:
                    use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["y_ping"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["y_done"], LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["y_pong"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    EndOp()
                with block[4]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[7])
                with block[5]:
                    use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["x_ping"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["x_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[6]:
                    use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["x_pong"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["x_ready"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[3])
                with block[8]:
                    use_lock(_cl["freqs_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["freqs_ping"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["freqs_ready"], LockAction.Release, value=1)
                    next_bd(block[9])
                with block[9]:
                    use_lock(_cl["freqs_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["freqs_pong"], offset=0, len=HEAD_DIM_ROPE)
                    use_lock(_cl["freqs_ready"], LockAction.Release, value=1)
                    next_bd(block[8])

        for col in reversed(range(N_COLS)):
            _make_core_mem(compute_tiles[col], core_locks[col], core_buf[col])

        # Declare @rope external function once (after the mem blocks; the
        # cached emits ``func.func private @rope`` after all cores but
        # before the flows -- our placement is functionally equivalent and
        # the assign-core-link-files pass routes the link_with attr).
        _BF16_HD_L1_ty = bf16_memref(HEAD_DIM_ROPE, memory_space=2)
        rope_fn = external_func(
            "rope",
            inputs=[_BF16_HD_L1_ty, _BF16_HD_L1_ty, _BF16_HD_L1_ty, np.int32],
            link_with="rope_pythoc.o",
        )
        rope_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # aie.core body per compute tile.
        def _make_core_body(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                head_dim_c = arith.constant(HEAD_DIM_ROPE, T.i32())
                c0 = arith.constant(0, T.index())  # noqa: F841 (matches cached preamble)
                c2 = arith.constant(2, T.index())  # noqa: F841
                # Note: the cached lists these constants explicitly but the
                # scf.for builder synthesises its own.  We declare them so
                # the device-block matches cached header structure.

                # Infinite outer loop (cached uses cf.br ^bb1; we use a
                # while-true via range_(sys.maxsize) which lowers identically
                # after canonicalisation).
                for _outer in range_(_sys.maxsize):
                    for _inner in range_(0, n_rope_iters, 2):
                        # ping iter
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["freqs_ready"], LockAction.AcquireGreaterEqual, value=1)
                        rope_fn(_bufs["x_ping"], _bufs["freqs_ping"],
                                _bufs["y_ping"], head_dim_c)
                        use_lock(_cl["freqs_avail"], LockAction.Release, value=1)
                        use_lock(_cl["x_avail"], LockAction.Release, value=1)
                        # pong iter
                        use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["freqs_ready"], LockAction.AcquireGreaterEqual, value=1)
                        rope_fn(_bufs["x_pong"], _bufs["freqs_pong"],
                                _bufs["y_pong"], head_dim_c)
                        use_lock(_cl["freqs_avail"], LockAction.Release, value=1)
                        use_lock(_cl["x_avail"], LockAction.Release, value=1)
                        # y_full released twice (one per ping/pong produced)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)
                        use_lock(_cl["y_full"], LockAction.Release, value=1)

        for col in reversed(range(N_COLS)):
            _make_core_body(compute_tiles[col], core_locks[col], core_buf[col])

        # Flows: 3 groups of 8 (x in, freqs in, y out).
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 1,
                 compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)

        # Shim allocations: 8 S2MM (y out) then 8 MM2S 0 (x) then 8 MM2S 1
        # (freqs) -- matches cached emit order.
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_out}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_in_x}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_in_freqs}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # Runtime sequence.
        @runtime_sequence(*_rms_gemms_rope_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12):
            # Resolve host args by index (the unused ones are dropped to
            # silence linters).
            host_args = (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                         arg8, arg9, arg10, arg11, arg12)
            arg_x = host_args[x_arg_idx]
            arg_f = host_args[freqs_arg_idx]
            arg_y = host_args[out_arg_idx]

            # 8 x-input tasks.
            x_tasks = []
            for col in range(N_COLS):
                offset = col * x_per_col_offset_step
                t = dma_configure_task_for(f"air_channel_{chan_in_x}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_x, offset=offset, len=x_per_col_len,
                               dimensions=x_dims)
                        EndOp()
                dma_start_task(t)
                x_tasks.append(t)

            # 8 freqs-input tasks.
            f_tasks = []
            for col in range(N_COLS):
                offset = col * freqs_per_col_offset_step
                t = dma_configure_task_for(f"air_channel_{chan_in_freqs}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_f, offset=offset, len=freqs_per_col_len,
                               dimensions=freqs_dims)
                        EndOp()
                dma_start_task(t)
                f_tasks.append(t)

            # 8 y-output tasks.
            y_tasks = []
            for col in range(N_COLS):
                offset = col * out_per_col_offset_step
                t = dma_configure_task_for(f"air_channel_{chan_out}_{col}",
                                            issue_token=True)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_y, offset=offset, len=out_per_col_len,
                               dimensions=out_dims)
                        EndOp()
                dma_start_task(t)
                y_tasks.append(t)

            # Free 16 inputs, then await 8 outputs (cached order: 8 x +
            # 8 freqs freed, then 8 y awaited).
            for t in x_tasks:
                dma_free_task(t)
            for t in f_tasks:
                dma_free_task(t)
            for t in y_tasks:
                dma_await_task(t)



def _emit_rk_rope_seg() -> None:
    """Emit the placed-IRON ``rk_rope_seg`` device.

    Per cached runtime_sequence (lines 688-833 of
    ``rms_gemms_rope.npu.air.mlir``):
      * arg6  (memref<2048x512xbf16>)   -- K pre-RoPE input
      * arg10 (memref<1048576xbf16>)    -- RoPE-K freqs (cos/sin LUT)
      * arg12 (memref<2048x512xbf16>)   -- RoPE-K output (with issue_token)
      * channel ids: x=54, freqs=55, out=56.
    """
    _emit_rope_device(
        sym="rk_rope_seg",
        n_rope_iters=_RK_ROPE_ITERS,
        chan_in_x=54, chan_in_freqs=55, chan_out=56,
        x_arg_idx=6, freqs_arg_idx=10, out_arg_idx=12,
        # Each col gets 256 rows x 512 cols = 131072 bf16 elements.
        x_per_col_len=131072,
        x_per_col_offset_step=131072,
        x_dims=[(256, 512), (512, 1)],
        freqs_per_col_len=131072,
        freqs_per_col_offset_step=131072,
        freqs_dims=[(256, 512), (512, 1)],
        out_per_col_len=131072,
        out_per_col_offset_step=131072,
        out_dims=[(256, 512), (512, 1)],
        ext_buf_dtype_shapes=[
            (SEQ_LEN, KV_DIM),   # __air_external_buffer
            (1048576,),          # __air_external_buffer_1
            (SEQ_LEN, KV_DIM),   # __air_external_buffer_2
        ],
    )


def _emit_rq_rope_seg() -> None:
    """Emit the placed-IRON ``rq_rope_seg`` device.

    Per cached runtime_sequence (lines 1512-1657 of
    ``rms_gemms_rope.npu.air.mlir``):
      * arg4  (memref<2048x2048xbf16>)  -- Q pre-RoPE input
      * arg9  (memref<4194304xbf16>)    -- RoPE-Q freqs
      * arg11 (memref<2048x2048xbf16>)  -- RoPE-Q output (with issue_token)
      * channel ids: x=51, freqs=52, out=53.
    """
    _emit_rope_device(
        sym="rq_rope_seg",
        n_rope_iters=_RQ_ROPE_ITERS,
        chan_in_x=51, chan_in_freqs=52, chan_out=53,
        x_arg_idx=4, freqs_arg_idx=9, out_arg_idx=11,
        # Each col gets 524288 bf16 elements = 2 outer x (512 rows x 512 cols).
        x_per_col_len=524288,
        x_per_col_offset_step=524288,
        x_dims=[(2, 262144), (512, 512), (512, 1)],
        freqs_per_col_len=524288,
        freqs_per_col_offset_step=524288,
        freqs_dims=[(2, 262144), (512, 512), (512, 1)],
        out_per_col_len=524288,
        out_per_col_offset_step=524288,
        out_dims=[(2, 262144), (512, 512), (512, 1)],
        ext_buf_dtype_shapes=[
            (SEQ_LEN, EMB_DIM),  # __air_external_buffer
            (4194304,),          # __air_external_buffer_1
            (SEQ_LEN, EMB_DIM),  # __air_external_buffer_2
        ],
    )


# ---------------------------------------------------------------------------
# v_matmul_seg device (Phase 4.5c -- first GEMM).
# ---------------------------------------------------------------------------
# The cached v_matmul_seg is the largest single device in rms_gemms_rope
# (~7500 lines), with 32 cores in an 8x4 herd (cols 0..7, rows 2..5) and
# **asymmetric** mem-tile DMA topology between cols 0-3 (10 locks each)
# and cols 4-7 (6 locks each).  Cols 0-3 each host both an X-broadcast
# DMA chain (mem -> all 8 cores in one row, e.g. mem_0_1 -> row 2,
# mem_1_1 -> row 3, ...) and a V-weight DMA chain (shim -> mem -> 4
# cores in the same column).  Cols 4-7 only host the V-weight chain.
#
# Per compute tile (32 total):
#   * 5 buffers in L1 (memory_space=2):
#       buf_C       memref<1x1x16x8x8x8 xbf16, 2>     (C-out, single-buffered)
#       buf_A_pong  memref<1x1x4x8x8x8  xbf16, 2>     (A-in pong)
#       buf_B_ping  memref<1x1x16x4x8x8 xbf16, 2>     (B-in ping)
#       buf_A_ping  memref<1x1x4x8x8x8  xbf16, 2>     (A-in ping)
#       buf_B_pong  memref<1x1x16x4x8x8 xbf16, 2>     (B-in pong)
#   * 6 locks at ids 5..0, init=(2, 0, 2, 0, 1, 0):
#       id=5 init=2  B_sem        (mem MM2S 1 acq; core rel)
#       id=4 init=0  B_ready      (mem MM2S 1 rel; core acq)
#       id=3 init=2  A_sem        (mem MM2S 0 acq; core rel)
#       id=2 init=0  A_ready      (mem MM2S 0 rel; core acq)
#       id=1 init=1  C_done       (mem S2MM rel; core acq)  -- 1 slot
#       id=0 init=0  C_full       (core rel; mem S2MM acq)
#   * aie.mem block with 3 DMA channels:
#       MM2S 0 (C out): buf_C, self-loop (1-BD, len 8192, 3D dims)
#       S2MM 0 (A in): buf_A_ping <-> buf_A_pong (2-BD ping-pong, len 2048)
#       S2MM 1 (B in): buf_B_ping <-> buf_B_pong (2-BD ping-pong, len 4096)
#   * aie.core body:
#       acquire C_done x1 (once)
#       zero buf_C (16x8 inner nested 8x8 micro-tile writes)
#       for k_outer in 0..32:
#         acquire A_ready x1, B_ready x1
#         func.call bf16_gemm_kernel_bf16out(buf_A_ping, buf_B_ping, buf_C)
#         release B_sem x1, A_sem x1
#         acquire A_ready x1, B_ready x1
#         func.call bf16_gemm_kernel_bf16out(buf_A_pong, buf_B_pong, buf_C)
#         release B_sem x1, A_sem x1
#       release C_full x1
#       cf.br ^bb1
#
# Per mem tile (8 total):
#   cols 0-3 -- 10 locks at ids 9..0:
#       id=9 init=1  X_pong_sem    (MM2S 2 acq)
#       id=8 init=0  X_pong_ready  (S2MM 1 rel)
#       id=7 init=1  X_ping_sem    (MM2S 2 acq)
#       id=6 init=0  X_ping_ready  (S2MM 1 rel)
#       id=5 init=1  C_pong_sem    (MM2S 1 acq; S2MM 0 rel)
#       id=4 init=0  C_pong_ready  (MM2S 1 rel; S2MM 0 acq)
#       id=3 init=1  C_ping_sem    (MM2S 1 acq; S2MM 0 rel)
#       id=2 init=0  C_ping_ready  (MM2S 1 rel; S2MM 0 acq)
#       id=1 init=4  W_sem (4-way) (MM2S 0 acq by 4; S2MM 2..5 rel by 1)
#       id=0 init=0  W_ready       (MM2S 0 rel by 4; S2MM 2..5 acq by 1)
#   cols 4-7 -- 6 locks at ids 5..0:
#       id=5 init=1  C_pong_sem
#       id=4 init=0  C_pong_ready
#       id=3 init=1  C_ping_sem
#       id=2 init=0  C_ping_ready
#       id=1 init=4  W_sem (4-way)
#       id=0 init=0  W_ready
#
# Per mem-tile buffers:
#   cols 0..7: 1 V-weight buf  (1x4x64x128 bf16, L2)
#   cols 0..7: 2 C-out bufs    (1x1x64x64 bf16, L2)        -- ping/pong
#   cols 0..3: 2 X-input bufs  (1x1x64x128 bf16, L2)       -- ping/pong
#
# aie.memtile_dma per col:
#   cols 0-3 -- 9 channels:
#       MM2S 0: V-weight broadcast (self-loop bd, len 32768, 3D dims)
#               acq W_ready x4, rel W_sem x4
#       MM2S 1: C-out ping/pong (2-BD cycle, each len 4096, 3D dims)
#               iter A: acq C_ping_ready, rel C_ping_sem
#               iter B: acq C_pong_ready, rel C_pong_sem
#       MM2S 2: X-broadcast ping/pong (2-BD cycle, each len 8192, 4D dims)
#               iter A: acq X_ping_ready, rel X_ping_sem
#               iter B: acq X_pong_ready, rel X_pong_sem
#       S2MM 0: C-in ping/pong (mirror of MM2S 1)
#       S2MM 1: X-in ping/pong (mirror of MM2S 2)
#       S2MM 2..5: V-weight slices (4 channels, each 1-BD self-loop at
#                  offset = slice * 8192, acq W_sem x1, rel W_ready x1)
#   cols 4-7 -- 7 channels (no X DMA):
#       MM2S 0, MM2S 1, S2MM 0, S2MM 1..4 (V-weight slices)
#
# Flows (116 total):
#   * 8 shim->mem DMA 0   (V-weight in, 1 per col)
#   * 4 shim->mem DMA 1   (X-input in, cols 0-3 only)
#   * 8 mem->shim DMA 0   (C-out, 1 per col)
#   * 32 mem->core DMA 1 (V-weight, mem_col -> tile_col_{2..5})
#   * 32 mem->core DMA 2 (X-broadcast, mem_{0..3} -> tile_{0..7}_{2+col})
#   * 32 core->mem DMA 0 (C-out from core to mem-tile)
#         cols 0-3: tile_C_R -> mem_C_1 DMA (2+R-2) for R in 2..5
#         cols 4-7: tile_C_R -> mem_C_1 DMA (1+R-2) for R in 2..5
#
# Shim allocations (20 total):
#   air_channel_60_C  MM2S 0 on shim_C_0  (X input,   8 channels, sources arg2)
#   air_channel_61_C  S2MM 0 on shim_C_0  (V output,  8 channels, sinks   arg8)
#   air_channel_65_C  MM2S 1 on shim_C_0  (V weight,  cols 0-3,   sources arg7)
#
# Runtime sequence (20 dma_configure_task_for blocks):
#   X tasks (8):       arg2 @ offset col*131072, 4D dims, repeat_count=3
#   V weight (4):      arg7 @ offset col*128,   3D dims, repeat_count=3
#   V output (8):      arg8 @ offset col*32768, 3D dims, issue_token=True
#   then 12 dma_free_task (8 X + 4 V-weight), then 8 dma_await_task (V out).
V_MATMUL_K_OUTER = 32              # K-outer loop bound per core dispatch
V_MATMUL_C_M = 16                  # buf_C outer-M dim (16 m-blocks of 8x8)
V_MATMUL_C_N = 8                   # buf_C outer-N dim (8 n-blocks of 8x8)


def _emit_matmul_device(
    sym_name: str,
    *,
    weight_arg: int,
    x_arg: int,
    output_arg: int,
    output_shape: tuple,         # (M, N), e.g. (2048, 512) or (2048, 2048)
    weight_shape: tuple,         # (Kw, Nw), e.g. (2048, 512) or (2048, 2048)
    x_channel: int,              # shim MM2S 0 channel id (e.g. 60 for V)
    weight_channel: int,         # shim MM2S 1 channel id (e.g. 65 for V)
    output_channel: int,         # shim S2MM 0 channel id (e.g. 61 for V)
    extbuf_shapes: tuple,        # 3 shapes for the __air_external_buffer triplet
) -> None:
    """Emit a placed-IRON matmul segment device (v / k / q).

    Must be called inside an mlir_mod_ctx().  Registers one
    ``aie.device(npu2) @<sym_name>`` op.  Body-level divergence from
    cached: each K-outer iteration calls ``bf16_gemm_kernel_bf16out``
    once on the L1 (A, B, C) buffer triple instead of inlining the
    4-MAC vector.contract chain.  Infrastructure (tiles, locks with
    init values, buffers, flows, memtile_dma BD chains, shim allocs,
    runtime_sequence) matches the cached verbatim.

    The three matmul segments (v/k/q) all share the same compute
    topology (32 cores in 8x4, 8 mem tiles with asymmetric lock counts,
    same per-core GEMM kernel call).  They differ only in:
      * shim channel ids (parameter ``*_channel``)
      * host-arg indices for X / weight / output buffers
      * output and weight shapes (drive the runtime_sequence offsets and
        in Q's case the unrolled 4-way dispatch pattern)
      * the opaque ``__air_external_buffer`` metadata shapes
    """
    from aie.dialects import memref as memref_dialect
    from aie.dialects import vector as vector_dialect
    from aie.extras import types as T
    from aie.ir import UnitAttr

    out_M, out_N = output_shape

    @device(AIEDevice.npu2, sym_name=sym_name)
    def _dev():
        # 8 shim + 8 mem + 32 compute tiles (cols 0..7, rows 2..5).
        shim_tiles   = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles    = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = {}
        for col in range(N_COLS):
            for row in range(2, 6):  # rows 2, 3, 4, 5
                compute_tiles[(col, row)] = tile(col, row)

        # ------------------------------------------------------------
        # Locks.  Order matters for structural diff parity -- cached
        # emits per-col descending: col 7, col 6, ..., col 0 for the
        # mem tiles, then per-tile descending col-by-col for compute.
        # ------------------------------------------------------------
        # Mem-tile locks.  Cols 0-3 have 10 locks (ids 9..0), cols 4-7
        # have 6 (ids 5..0).  init pattern documented at the top of this
        # section.
        # Order of declaration matches cached: col 7 first (6 locks),
        # then col 6 (6), ..., col 4 (6), then col 3 (10), 2 (10),
        # 1 (10), 0 (10).
        mem_locks = {}
        for col in reversed(range(4, N_COLS)):    # cols 7,6,5,4 (6 locks each)
            mt = mem_tiles[col]
            mem_locks[col] = {
                "C_pong_sem":   lock(mt, lock_id=5, init=1),
                "C_pong_ready": lock(mt, lock_id=4, init=0),
                "C_ping_sem":   lock(mt, lock_id=3, init=1),
                "C_ping_ready": lock(mt, lock_id=2, init=0),
                "W_sem":        lock(mt, lock_id=1, init=4),
                "W_ready":      lock(mt, lock_id=0, init=0),
            }
        for col in reversed(range(4)):            # cols 3,2,1,0 (10 locks each)
            mt = mem_tiles[col]
            mem_locks[col] = {
                "X_pong_sem":   lock(mt, lock_id=9, init=1),
                "X_pong_ready": lock(mt, lock_id=8, init=0),
                "X_ping_sem":   lock(mt, lock_id=7, init=1),
                "X_ping_ready": lock(mt, lock_id=6, init=0),
                "C_pong_sem":   lock(mt, lock_id=5, init=1),
                "C_pong_ready": lock(mt, lock_id=4, init=0),
                "C_ping_sem":   lock(mt, lock_id=3, init=1),
                "C_ping_ready": lock(mt, lock_id=2, init=0),
                "W_sem":        lock(mt, lock_id=1, init=4),
                "W_ready":      lock(mt, lock_id=0, init=0),
            }

        # Compute-tile locks.  Cached order: row 2 col 0..7, row 3
        # col 0..7, row 4 col 0..7, row 5 col 0..7.
        core_locks = {}
        for row in range(2, 6):
            for col in range(N_COLS):
                ct = compute_tiles[(col, row)]
                core_locks[(col, row)] = {
                    "B_sem":   lock(ct, lock_id=5, init=2),
                    "B_ready": lock(ct, lock_id=4, init=0),
                    "A_sem":   lock(ct, lock_id=3, init=2),
                    "A_ready": lock(ct, lock_id=2, init=0),
                    "C_done":  lock(ct, lock_id=1, init=1),
                    "C_full":  lock(ct, lock_id=0, init=0),
                }

        # ------------------------------------------------------------
        # Buffers.  Mem-tile buffers first (descending col order per
        # category), then compute-tile buffers (descending row, then
        # descending col).
        # ------------------------------------------------------------
        # Cached emit order for mem-tile bufs (per category):
        #   buf631..buf624 : 1x4x64x128 (V-weight), col 0..7
        #   buf623..buf608 : 1x1x64x64  (C-out ping/pong, 2/col), col 0..7
        #   buf607..buf600 : 1x1x64x128 (X-in ping/pong, 2/col), cols 0..3
        BF16_VW_L2  = bf16_memref(1, 4, 64, 128, memory_space=1)
        BF16_CO_L2  = bf16_memref(1, 1, 64, 64,  memory_space=1)
        BF16_XI_L2  = bf16_memref(1, 1, 64, 128, memory_space=1)

        mem_buf = {col: {} for col in range(N_COLS)}
        # V-weight: 1 per col, col order 0..7.
        for col in range(N_COLS):
            mem_buf[col]["W"] = buffer(mem_tiles[col], datatype=BF16_VW_L2)
        # C-out ping/pong: 2 per col, order col 0 (ping then pong), col 1, ...
        for col in range(N_COLS):
            mem_buf[col]["C_ping"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
            mem_buf[col]["C_pong"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
        # X-input ping/pong: 2 each on cols 0..3 only.
        for col in range(4):
            mem_buf[col]["X_ping"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)
            mem_buf[col]["X_pong"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)

        # Compute-tile buffers.  Cached emit order: row 5 col 7 first,
        # then row 5 col 6, ..., row 5 col 0; then row 4 col 7, ...,
        # down to row 2 col 0.  Per tile (5 buffers, descending sym):
        #   buf_C       1x1x16x8x8x8
        #   buf_A_pong  1x1x4x8x8x8
        #   buf_B_ping  1x1x16x4x8x8
        #   buf_A_ping  1x1x4x8x8x8
        #   buf_B_pong  1x1x16x4x8x8
        BF16_C_L1  = bf16_memref(1, 1, 16, 8, 8, 8, memory_space=2)
        BF16_A_L1  = bf16_memref(1, 1, 4, 8, 8, 8, memory_space=2)
        BF16_B_L1  = bf16_memref(1, 1, 16, 4, 8, 8, memory_space=2)

        core_buf = {}
        for row in reversed(range(2, 6)):
            for col in reversed(range(N_COLS)):
                ct = compute_tiles[(col, row)]
                bufs = {}
                bufs["C"]      = buffer(ct, datatype=BF16_C_L1)
                bufs["A_pong"] = buffer(ct, datatype=BF16_A_L1)
                bufs["B_ping"] = buffer(ct, datatype=BF16_B_L1)
                bufs["A_ping"] = buffer(ct, datatype=BF16_A_L1)
                bufs["B_pong"] = buffer(ct, datatype=BF16_B_L1)
                core_buf[(col, row)] = bufs

        # External buffers (opaque AIR metadata; kept for diff parity).
        # Shapes vary across v/k/q matmul devices (V/K: 2048x2048 + 2x
        # 2048x512; Q: all 2048x2048).
        eb0, eb1, eb2 = extbuf_shapes
        external_buffer(bf16_np(*eb0), name="__air_external_buffer")
        external_buffer(bf16_np(*eb1), name="__air_external_buffer_1")
        external_buffer(bf16_np(*eb2), name="__air_external_buffer_2")

        # ------------------------------------------------------------
        # aie.mem blocks per compute tile.  Cached order: row 5 col 7
        # first, then col 6 .., row 4 col 7 .., ..., row 2 col 0.
        # ------------------------------------------------------------
        def _make_compute_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                # MM2S 0: C-out, self-loop.
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_cl["C_full"], LockAction.AcquireGreaterEqual, value=1)
                    # 3D dims: [<size=64, stride=8>, <size=16, stride=512>,
                    #           <size=8, stride=1>]  total len=8192
                    dma_bd(_bufs["C"], offset=0, len=8192,
                           dimensions=[(64, 8), (16, 512), (8, 1)])
                    use_lock(_cl["C_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[6])
                # S2MM 0: A-in ping/pong.
                with block[4]:
                    use_lock(_cl["A_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["A_ping"], offset=0, len=2048)
                    use_lock(_cl["A_ready"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(_cl["A_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["A_pong"], offset=0, len=2048)
                    use_lock(_cl["A_ready"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[6]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[2])
                # S2MM 1: B-in ping/pong.
                with block[7]:
                    use_lock(_cl["B_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["B_ping"], offset=0, len=4096)
                    use_lock(_cl["B_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(_cl["B_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["B_pong"], offset=0, len=4096)
                    use_lock(_cl["B_ready"], LockAction.Release, value=1)
                    next_bd(block[7])

        for row in reversed(range(2, 6)):
            for col in reversed(range(N_COLS)):
                _make_compute_mem(compute_tiles[(col, row)],
                                  core_locks[(col, row)],
                                  core_buf[(col, row)])

        # ------------------------------------------------------------
        # External function declaration.  Each compute core's link_with
        # attr is set by the aie-assign-core-link-files pass based on
        # this declaration.
        # ------------------------------------------------------------
        gemm_fn = external_func(
            "bf16_gemm_kernel_bf16out",
            inputs=[BF16_A_L1, BF16_B_L1, BF16_C_L1],
            link_with="bf16_gemm_pythoc_M16_N8_K4_AT_bf16out_s64_256_512_64_512_64.o",
        )
        gemm_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # ------------------------------------------------------------
        # aie.core body per compute tile.
        # ------------------------------------------------------------
        def _make_compute_core(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                # 6D identity permutation map for the zero-init transfer_write
                # (matches the cached's `permutation_map = (d0,d1,d2,d3,d4,d5)`
                # affine map).
                zero_perm = AffineMap.get(
                    6, 0,
                    [AffineDimExpr.get(0), AffineDimExpr.get(1),
                     AffineDimExpr.get(2), AffineDimExpr.get(3),
                     AffineDimExpr.get(4), AffineDimExpr.get(5)])
                vec_zero_ty = T.vector(1, 1, 1, 1, 8, 8, T.bf16())
                np_zero = np.zeros((1, 1, 1, 1, 8, 8), dtype=bfloat16)
                cst_zero = arith.constant(np_zero, vec_zero_ty)
                c0_idx = arith.constant(0, T.index())

                for _ in range_(_sys.maxsize):
                    use_lock(_cl["C_done"], LockAction.AcquireGreaterEqual, value=1)
                    # Zero buf_C inline: nested 16x8 over 8x8 micro-tiles.
                    for m_i in range_(0, V_MATMUL_C_M, 1):
                        for n_i in range_(0, V_MATMUL_C_N, 1):
                            vector_dialect.transfer_write(
                                None, cst_zero, _bufs["C"],
                                [c0_idx, c0_idx, m_i, n_i, c0_idx, c0_idx],
                                permutation_map=zero_perm,
                                in_bounds=[True, True, True, True, True, True])
                    # K-outer loop: 32 iters.
                    for _k_outer in range_(0, V_MATMUL_K_OUTER, 1):
                        # Iter A (ping).
                        use_lock(_cl["A_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["B_ready"], LockAction.AcquireGreaterEqual, value=1)
                        gemm_fn(_bufs["A_ping"], _bufs["B_ping"], _bufs["C"])
                        use_lock(_cl["B_sem"], LockAction.Release, value=1)
                        use_lock(_cl["A_sem"], LockAction.Release, value=1)
                        # Iter B (pong).
                        use_lock(_cl["A_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["B_ready"], LockAction.AcquireGreaterEqual, value=1)
                        gemm_fn(_bufs["A_pong"], _bufs["B_pong"], _bufs["C"])
                        use_lock(_cl["B_sem"], LockAction.Release, value=1)
                        use_lock(_cl["A_sem"], LockAction.Release, value=1)
                    use_lock(_cl["C_full"], LockAction.Release, value=1)

        for row in reversed(range(2, 6)):
            for col in reversed(range(N_COLS)):
                _make_compute_core(compute_tiles[(col, row)],
                                   core_locks[(col, row)],
                                   core_buf[(col, row)])

        # ------------------------------------------------------------
        # Flows.  Cached order:
        #   8 shim->mem DMA 0       (V-weight in)
        #   4 shim->mem DMA 1       (X-input in, cols 0-3)
        #   8 mem->shim DMA 0       (C-out)
        #   32 mem->core DMA 1      (V-weight broadcast: mem_C -> tile_C_R for R in 2..5)
        #   32 mem->core DMA 2      (X-broadcast: mem_{R-2} -> tile_C_R for C in 0..7,
        #                            R in 2..5; i.e. mem_0_1 to row 2, etc.)
        #   32 core->mem DMA 0      (C-out feedback to mem-tile)
        # ------------------------------------------------------------
        # 8 shim -> mem DMA 0 (V-weight in).
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 mem_tiles[col], WireBundle.DMA, 0)
        # 4 shim -> mem DMA 1 (X-input in, cols 0-3 only).
        for col in range(4):
            flow(shim_tiles[col], WireBundle.DMA, 1,
                 mem_tiles[col], WireBundle.DMA, 1)
        # 8 mem -> shim DMA 0 (C-out).
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)
        # 32 mem -> core DMA 1 (V-weight broadcast, mem_C_1 -> tile_C_R for R in 2..5).
        for col in range(N_COLS):
            for row in range(2, 6):
                flow(mem_tiles[col], WireBundle.DMA, 1,
                     compute_tiles[(col, row)], WireBundle.DMA, 0)
        # 32 mem -> core DMA 2 (X-broadcast: mem_{R-2}_1 broadcasts row R-aligned slice
        # to all 8 cores in that row).
        for row_offset in range(4):       # mem 0..3 -> rows 2..5
            for col in range(N_COLS):
                flow(mem_tiles[row_offset], WireBundle.DMA, 2,
                     compute_tiles[(col, 2 + row_offset)], WireBundle.DMA, 1)
        # 32 core -> mem DMA channels (C-out feedback).  Cached uses an
        # asymmetric mapping:
        #   cols 0-3: tile_C_R -> mem_C_1 DMA {2, 3, 4, 5} for R in {2, 3, 4, 5}
        #   cols 4-7: tile_C_R -> mem_C_1 DMA {1, 2, 3, 4} for R in {2, 3, 4, 5}
        # i.e. cols 0-3 reserve DMA 1 for X-broadcast, so the C-out chain
        # starts at DMA 2; cols 4-7 don't have an X channel so the C-out
        # chain starts at DMA 1.
        for col in range(4):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 2 + (row - 2))
        for col in range(4, N_COLS):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 1 + (row - 2))

        # ------------------------------------------------------------
        # aie.memtile_dma blocks (per col).
        # ------------------------------------------------------------
        def _make_memtile_dma_x_col(col):
            """Emit memtile_dma for cols 0-3 (with X-broadcast chain)."""
            ml = mem_locks[col]
            mt = mem_tiles[col]
            mb = mem_buf[col]
            R = (col, ml, mt, mb)
            del R  # closure capture below
            @memtile_dma(mt)
            def _mt_dma(block):
                # MM2S 0: V-weight broadcast (self-loop).
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=32768,
                           dimensions=[(64, 128), (4, 8192), (128, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong.
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ml["C_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=4096,
                           dimensions=[(8, 8), (64, 64), (8, 1)])
                    use_lock(ml["C_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ml["C_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=4096,
                           dimensions=[(8, 8), (64, 64), (8, 1)])
                    use_lock(ml["C_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[4])
                # MM2S 2: X-broadcast ping/pong.
                with block[6]:
                    dma_start(DMAChannelDir.MM2S, 2, dest=block[7], chain=block[9])
                with block[7]:
                    use_lock(ml["X_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_ping"], offset=0, len=8192,
                           dimensions=[(2, 4096), (16, 8), (32, 128), (8, 1)])
                    use_lock(ml["X_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(ml["X_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_pong"], offset=0, len=8192,
                           dimensions=[(2, 4096), (16, 8), (32, 128), (8, 1)])
                    use_lock(ml["X_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[7])
                # S2MM 0: C-in ping/pong (from compute).
                with block[9]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[10], chain=block[12])
                with block[10]:
                    use_lock(ml["C_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=4096)
                    use_lock(ml["C_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[11])
                with block[11]:
                    use_lock(ml["C_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=4096)
                    use_lock(ml["C_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[10])
                # S2MM 1: X-in ping/pong (from shim).
                with block[12]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[13], chain=block[15])
                with block[13]:
                    use_lock(ml["X_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_ping"], offset=0, len=8192)
                    use_lock(ml["X_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[14]:
                    use_lock(ml["X_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_pong"], offset=0, len=8192)
                    use_lock(ml["X_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[13])
                # S2MM 2..5: V-weight slice 0..3 (each 1-BD self-loop).
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 2, dest=block[16], chain=block[17])
                with block[16]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=0, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[16])
                with block[17]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[18], chain=block[19])
                with block[18]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=8192, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[18])
                with block[19]:
                    dma_start(DMAChannelDir.S2MM, 4, dest=block[20], chain=block[21])
                with block[20]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=16384, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[20])
                with block[21]:
                    dma_start(DMAChannelDir.S2MM, 5, dest=block[22], chain=block[2])
                with block[22]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=24576, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[22])

        def _make_memtile_dma_no_x_col(col):
            """Emit memtile_dma for cols 4-7 (no X-broadcast chain)."""
            ml = mem_locks[col]
            mt = mem_tiles[col]
            mb = mem_buf[col]
            @memtile_dma(mt)
            def _mt_dma(block):
                # MM2S 0: V-weight broadcast (self-loop).
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=32768,
                           dimensions=[(64, 128), (4, 8192), (128, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong.
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ml["C_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=4096,
                           dimensions=[(8, 8), (64, 64), (8, 1)])
                    use_lock(ml["C_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ml["C_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=4096,
                           dimensions=[(8, 8), (64, 64), (8, 1)])
                    use_lock(ml["C_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[4])
                # S2MM 0: C-in ping/pong (from compute).
                with block[6]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[9])
                with block[7]:
                    use_lock(ml["C_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=4096)
                    use_lock(ml["C_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(ml["C_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=4096)
                    use_lock(ml["C_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[7])
                # S2MM 1..4: V-weight slice 0..3 (each 1-BD self-loop).
                with block[9]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[10], chain=block[11])
                with block[10]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=0, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[10])
                with block[11]:
                    dma_start(DMAChannelDir.S2MM, 2, dest=block[12], chain=block[13])
                with block[12]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=8192, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[12])
                with block[13]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[14], chain=block[15])
                with block[14]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=16384, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 4, dest=block[16], chain=block[2])
                with block[16]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=24576, len=8192)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[16])

        # Cached emits memtile_dma in col 0..7 order.
        for col in range(N_COLS):
            if col < 4:
                _make_memtile_dma_x_col(col)
            else:
                _make_memtile_dma_no_x_col(col)

        # ------------------------------------------------------------
        # Shim allocations.  Cached order (per device):
        #   air_channel_<output_channel>_C (S2MM 0 on shim_C_0)  output,    8 channels
        #   air_channel_<x_channel>_C      (MM2S 0 on shim_C_0)  X input,   8 channels
        #   air_channel_<weight_channel>_C (MM2S 1 on shim_C_0)  W weight,  cols 0-3
        # ------------------------------------------------------------
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{output_channel}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{x_channel}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(4):
            shim_dma_allocation(
                f"air_channel_{weight_channel}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # ------------------------------------------------------------
        # Runtime sequence.
        # The V and K matmul devices emit a 2048x512 output and use a
        # single 4-way unrolled dma_bd dim on the X input (so 8 X-tasks,
        # 4 W-tasks, 8 output-tasks total).  The Q matmul device emits
        # a 2048x2048 output (4x larger) so it unrolls each X-channel
        # into 4 dispatches (32 X-tasks total), each W-channel into 4
        # dispatches (16 W-tasks), and adds repeat_count=3 to the
        # output tasks.  Q also uses a reverse await + interleaved free
        # ordering to match the cached IR.
        # ------------------------------------------------------------
        # Output shape decides dispatch fan-out:
        #   2048x512  : single-dispatch (4 inner outer iters baked into
        #               the x bd's leading <4, 1048576> dim) -- V and K
        #   2048x2048 : 4-dispatch unrolled (the leading <4, ...> is the
        #               dispatcher count, not a bd dim) -- Q
        if (out_M, out_N) == (2048, 512):
            n_dispatches = 1
        elif (out_M, out_N) == (2048, 2048):
            n_dispatches = 4
        else:
            raise ValueError(f"unsupported output_shape {output_shape!r}")

        @runtime_sequence(*_rms_gemms_rope_host_arg_types(),
                          sym_name=f"{sym_name}_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12):
            args = (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                    arg8, arg9, arg10, arg11, arg12)
            x_buf = args[x_arg]
            w_buf = args[weight_arg]
            y_buf = args[output_arg]

            x_tasks = []
            w_tasks = []
            y_tasks = []

            if n_dispatches == 1:
                # V/K shape: 8 X-tasks, 4 W-tasks, 8 Y-tasks.
                for col in range(N_COLS):
                    offset = col * 131072
                    t = dma_configure_task_for(
                        f"air_channel_{x_channel}_{col}",
                        repeat_count=3)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(x_buf, offset=offset, len=131072,
                                   dimensions=[(4, 1048576), (32, 64),
                                               (64, 2048), (64, 1)])
                            EndOp()
                    dma_start_task(t)
                    x_tasks.append(t)

                for col in range(4):
                    offset = col * 128
                    t = dma_configure_task_for(
                        f"air_channel_{weight_channel}_{col}",
                        repeat_count=3)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(w_buf, offset=offset, len=262144,
                                   dimensions=[(32, 32768), (64, 512),
                                               (128, 1)])
                            EndOp()
                    dma_start_task(t)
                    w_tasks.append(t)

                for col in range(N_COLS):
                    offset = col * 32768
                    t = dma_configure_task_for(
                        f"air_channel_{output_channel}_{col}",
                        issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(y_buf, offset=offset, len=131072,
                                   dimensions=[(4, 262144), (64, 512),
                                               (512, 1)])
                            EndOp()
                    dma_start_task(t)
                    y_tasks.append(t)

                # Frees: X first (in order), then W; then awaits.
                for t in x_tasks:
                    dma_free_task(t)
                for t in w_tasks:
                    dma_free_task(t)
                for t in y_tasks:
                    dma_await_task(t)

            else:
                # Q shape: 32 X-tasks, 16 W-tasks, 8 Y-tasks.
                # Outer order: col-major (col 0..7), and within each col,
                # dispatch-major (d 0..3).  X offset per dispatch:
                #   col*131072 + d * 1048576.
                # W offset is constant per col (col*128) -- the cached's
                # dispatches all re-issue the same bd-base.
                # Y offset per col: col*131072 (with repeat_count=3 to
                # cover the 4 inner outer iters).
                for col in range(N_COLS):
                    for d in range(4):
                        offset = col * 131072 + d * 1048576
                        t = dma_configure_task_for(
                            f"air_channel_{x_channel}_{col}",
                            repeat_count=3)
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(x_buf, offset=offset, len=131072,
                                       dimensions=[(32, 64), (64, 2048),
                                                   (64, 1)])
                                EndOp()
                        dma_start_task(t)
                        x_tasks.append(t)

                for col in range(4):
                    for d in range(4):
                        offset = col * 128
                        t = dma_configure_task_for(
                            f"air_channel_{weight_channel}_{col}",
                            repeat_count=3)
                        with bds(t) as bd:
                            with bd[0]:
                                dma_bd(w_buf, offset=offset, len=262144,
                                       dimensions=[(4, 512),
                                                   (32, 131072),
                                                   (64, 2048),
                                                   (128, 1)])
                                EndOp()
                        dma_start_task(t)
                        w_tasks.append(t)

                for col in range(N_COLS):
                    offset = col * 131072
                    t = dma_configure_task_for(
                        f"air_channel_{output_channel}_{col}",
                        issue_token=True, repeat_count=3)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(y_buf, offset=offset, len=131072,
                                   dimensions=[(4, 1048576),
                                               (4, 512),
                                               (64, 2048),
                                               (512, 1)])
                            EndOp()
                    dma_start_task(t)
                    y_tasks.append(t)

                # Q ordering: awaits first (reverse), then frees in
                # reverse-col groups of 4 (W first, then X).
                for t in reversed(y_tasks):
                    dma_await_task(t)
                # W frees: cols 3 -> 0, each col's 4 dispatches in order.
                for col in reversed(range(4)):
                    for d in range(4):
                        dma_free_task(w_tasks[col * 4 + d])
                # X frees: cols 7 -> 0, each col's 4 dispatches in order.
                for col in reversed(range(N_COLS)):
                    for d in range(4):
                        dma_free_task(x_tasks[col * 4 + d])


# ---------------------------------------------------------------------------
# Per-device thin wrappers around _emit_matmul_device.
#
# Host-arg index conventions (verified against the cached runtime_sequence
# blocks in reference_mlir/rms_gemms_rope.npu.air.mlir):
#   arg0 : Q-weight        (memref<2048x2048xbf16>)  -- unused by v/k
#   arg1 : RMSNorm gamma   (memref<2048xbf16>)
#   arg2 : X-input (RMS'd) (memref<2048x2048xbf16>)
#   arg3 : Q-output        (memref<2048x2048xbf16>)
#   arg4 : (post-RoPE Q)   (memref<2048x2048xbf16>)
#   arg5 : K-weight        (memref<2048x512xbf16>)
#   arg6 : K-output        (memref<2048x512xbf16>)
#   arg7 : V-weight        (memref<2048x512xbf16>)
#   arg8 : V-output        (memref<2048x512xbf16>)
#
# Wait -- the cached runtime_sequence shows:
#   v_matmul_seg: x=arg2, w=arg7, y=arg8   (channels 60/65/61)
#   k_matmul_seg: x=arg2, w=arg5, y=arg6   (channels 57/58/62)
#   q_matmul_seg: x=arg2, w=arg3, y=arg4   (channels 59/64/63)
#
# So in Q, arg3 is the Q-weight (2048x2048) and arg4 is Q-output
# (2048x2048).  This contradicts the naming convention used in the
# host-arg-types comment at the top of this file, but matches what the
# cached MLIR actually does.  The 4.5b RoPE phase already verified that
# arg4 is the pre-RoPE Q (output of q_matmul_seg, input to rq_rope_seg)
# and arg6 is the pre-RoPE K (output of k_matmul_seg, input to
# rk_rope_seg).  This is consistent.
# ---------------------------------------------------------------------------
def _emit_v_matmul_seg() -> None:
    """Emit the placed-IRON v_matmul_seg device."""
    _emit_matmul_device(
        "v_matmul_seg",
        weight_arg=7,
        x_arg=2,
        output_arg=8,
        output_shape=(SEQ_LEN, KV_DIM),       # 2048 x 512
        weight_shape=(EMB_DIM, KV_DIM),       # 2048 x 512
        x_channel=60,
        weight_channel=65,
        output_channel=61,
        extbuf_shapes=((SEQ_LEN, EMB_DIM),
                       (SEQ_LEN, KV_DIM),
                       (SEQ_LEN, KV_DIM)),
    )


def _emit_k_matmul_seg() -> None:
    """Emit the placed-IRON k_matmul_seg device."""
    _emit_matmul_device(
        "k_matmul_seg",
        weight_arg=5,
        x_arg=2,
        output_arg=6,
        output_shape=(SEQ_LEN, KV_DIM),       # 2048 x 512
        weight_shape=(EMB_DIM, KV_DIM),       # 2048 x 512
        x_channel=57,
        weight_channel=58,
        output_channel=62,
        extbuf_shapes=((SEQ_LEN, EMB_DIM),
                       (SEQ_LEN, KV_DIM),
                       (SEQ_LEN, KV_DIM)),
    )


def _emit_q_matmul_seg() -> None:
    """Emit the placed-IRON q_matmul_seg device."""
    _emit_matmul_device(
        "q_matmul_seg",
        weight_arg=3,
        x_arg=2,
        output_arg=4,
        output_shape=(SEQ_LEN, EMB_DIM),      # 2048 x 2048
        weight_shape=(EMB_DIM, EMB_DIM),      # 2048 x 2048
        x_channel=59,
        weight_channel=64,
        output_channel=63,
        extbuf_shapes=((SEQ_LEN, EMB_DIM),
                       (SEQ_LEN, EMB_DIM),
                       (SEQ_LEN, EMB_DIM)),
    )


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
#
# Outer wrapper that fires the 6 inner devices in topological order:
#   r_weighted_rms_norm_seg ->
#   q_matmul_seg -> k_matmul_seg -> v_matmul_seg ->
#   rq_rope_seg -> rk_rope_seg
# All 6 segments share the same 13-arg host signature
# (see ``_rms_gemms_rope_host_arg_types``).
# ---------------------------------------------------------------------------
_DISPATCHER_ORDER = (
    "r_weighted_rms_norm_seg",
    "q_matmul_seg",
    "k_matmul_seg",
    "v_matmul_seg",
    "rq_rope_seg",
    "rk_rope_seg",
)


def _emit_dispatcher_device() -> None:
    """Emit the outer unnamed ``aie.device(npu2) { ... }`` dispatcher.

    Carries an ``aiex.runtime_sequence @rms_gemms_rope`` whose body fires
    each of the 6 inner segment sequences via ``aiex.configure`` +
    ``aiex.run``.  Each inner sequence receives the full 13-arg list
    (matches the cached IR; the inner devices only use the subset they
    need).
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *_rms_gemms_rope_host_arg_types(),
            sym_name="rms_gemms_rope",
        )
        def _outer(*args):
            for sym in _DISPATCHER_ORDER:
                cfg = ConfigureOp(symbol=sym)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(
                        runtime_sequence_symbol=f"{sym}_sequence",
                        args=list(args),
                    )


# ---------------------------------------------------------------------------
# Splice helper.
# ---------------------------------------------------------------------------
def _splice_device(cached_text: str, device_sym: str, new_device_block: str) -> str:
    """Replace ``aie.device(npu2) @<device_sym> { ... }`` in cached_text.

    The cached device may have a trailing attribute dict
    ``} {dlti.dl_spec = ...}`` after the closing brace -- we consume that
    too so the replacement is clean.
    """
    marker = f"aie.device(npu2) @{device_sym}"
    start = cached_text.find(marker)
    if start < 0:
        raise RuntimeError(f"could not find device {device_sym!r} in cached MLIR")

    # Find the opening `{` after the device name.
    brace_open = cached_text.find("{", start)
    if brace_open < 0:
        raise RuntimeError(f"could not find opening brace for device {device_sym!r}")

    depth = 0
    i = brace_open
    n = len(cached_text)
    body_close = -1
    while i < n:
        c = cached_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body_close = i
                break
        i += 1
    if body_close < 0:
        raise RuntimeError(f"unbalanced braces for device {device_sym!r}")

    # Possibly consume a trailing attribute dict ` {...}` after the body close.
    j = body_close + 1
    # Skip whitespace.
    while j < n and cached_text[j] in " \t":
        j += 1
    if j < n and cached_text[j] == "{":
        # Brace-count the trailing dict.
        depth = 0
        k = j
        while k < n:
            c = cached_text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    j = k + 1
                    break
            k += 1
    end = j

    return cached_text[:start] + new_device_block + cached_text[end:]


def _splice_dispatcher_device(cached_text: str, new_device_block: str) -> str:
    """Replace the unnamed ``aie.device(npu2) { ... }`` dispatcher device.

    The cached file has exactly one unnamed ``aie.device(npu2)`` op (all
    others carry an ``@<sym>`` attribute).  Match on
    ``aie.device(npu2) {`` (with a literal `{` after the paren, no `@`).
    """
    # The named devices in cached look like ``aie.device(npu2) @<sym>``;
    # the unnamed dispatcher looks like ``aie.device(npu2) {`` (whitespace
    # optional).  Search for the latter explicitly to avoid matching named
    # devices.
    import re
    pattern = re.compile(r"aie\.device\(npu2\)\s*\{")
    # Find the unique unnamed match.
    matches = []
    for m in pattern.finditer(cached_text):
        # Skip matches that are part of a named device (i.e. there's `@`
        # between `aie.device(npu2)` and `{`).  But our regex already
        # requires `{` directly after the paren+whitespace, so any match
        # is unnamed.
        matches.append(m)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 unnamed aie.device(npu2) dispatcher in "
            f"cached MLIR; found {len(matches)}"
        )
    m = matches[0]
    start = m.start()
    brace_open = m.end() - 1  # position of the literal `{`

    depth = 0
    i = brace_open
    n = len(cached_text)
    body_close = -1
    while i < n:
        c = cached_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                body_close = i
                break
        i += 1
    if body_close < 0:
        raise RuntimeError("unbalanced braces for unnamed dispatcher device")

    # Consume any trailing attribute dict ` {...}` after the body close.
    j = body_close + 1
    while j < n and cached_text[j] in " \t":
        j += 1
    if j < n and cached_text[j] == "{":
        depth = 0
        k = j
        while k < n:
            c = cached_text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    j = k + 1
                    break
            k += 1
    end = j

    return cached_text[:start] + new_device_block + cached_text[end:]


def _extract_dispatcher_device(module_text: str) -> str:
    """Extract the unnamed ``aie.device(npu2) { ... }`` from an emitted module.

    Mirrors ``_extract_single_device`` but matches on the unnamed
    dispatcher (no ``@<sym>`` attribute).
    """
    import re
    pattern = re.compile(r"aie\.device\(npu2\)\s*\{")
    matches = list(pattern.finditer(module_text))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 unnamed aie.device(npu2) dispatcher in "
            f"emitted module; found {len(matches)}"
        )
    m = matches[0]
    start = m.start()
    brace_open = m.end() - 1

    depth = 0
    i = brace_open
    n = len(module_text)
    while i < n:
        c = module_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Include trailing attribute dict if present.
                j = i + 1
                while j < n and module_text[j] in " \t":
                    j += 1
                if j < n and module_text[j] == "{":
                    depth2 = 0
                    k = j
                    while k < n:
                        cc = module_text[k]
                        if cc == "{":
                            depth2 += 1
                        elif cc == "}":
                            depth2 -= 1
                            if depth2 == 0:
                                i = k
                                break
                        k += 1
                return module_text[start:i + 1]
        i += 1
    raise RuntimeError("unbalanced braces extracting dispatcher device")


def _extract_single_device(module_text: str, device_sym: str) -> str:
    """Extract just ``aie.device(npu2) @<device_sym> { ... }`` from a module text.

    The dialect bindings emit a ``module { ... }`` wrapper around devices;
    we strip that wrapper and return only the named device block.
    """
    marker = f"aie.device(npu2) @{device_sym}"
    start = module_text.find(marker)
    if start < 0:
        raise RuntimeError(f"device {device_sym!r} not found in emitted module")
    brace_open = module_text.find("{", start)
    if brace_open < 0:
        raise RuntimeError(f"no opening brace for {device_sym!r}")
    depth = 0
    i = brace_open
    n = len(module_text)
    while i < n:
        c = module_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Include trailing attribute dict if present.
                j = i + 1
                while j < n and module_text[j] in " \t":
                    j += 1
                if j < n and module_text[j] == "{":
                    depth2 = 0
                    k = j
                    while k < n:
                        cc = module_text[k]
                        if cc == "{":
                            depth2 += 1
                        elif cc == "}":
                            depth2 -= 1
                            if depth2 == 0:
                                i = k
                                break
                        k += 1
                return module_text[start:i + 1]
        i += 1
    raise RuntimeError(f"unbalanced braces extracting {device_sym!r}")


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_rms_gemms_rope_module(seq_len: int = SEQ_LEN,
                                 emb_dim: int = EMB_DIM,
                                 kv_dim: int = KV_DIM,
                                 n_heads: int = N_HEADS,
                                 n_kv_heads: int = N_KV_HEADS,
                                 head_dim: int = HEAD_DIM,
                                 *,
                                 verbose: bool = False) -> str:
    """Build the prefill RMS+GEMMS+RoPE MLIR module.

    Phase 4.5e: all 7 devices are emitted via placed-IRON --
    ``r_weighted_rms_norm_seg``, ``rk_rope_seg``, ``rq_rope_seg``,
    ``v_matmul_seg``, ``k_matmul_seg``, ``q_matmul_seg``, and the outer
    unnamed dispatcher.  Only the leading ``module { ... }`` wrapper and
    inter-device whitespace still come from the cached MLIR via splice;
    every device body is placed-IRON.

    All dimensions must match the Llama-3.2-1B values; the cached AIR
    layout is shape-specialized.
    """
    if (seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim) != \
            (SEQ_LEN, EMB_DIM, KV_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM):
        raise ValueError(
            "rms_gemms_rope builder is currently fixed to Llama-3.2-1B prefill "
            f"dimensions; got seq_len={seq_len}, emb_dim={emb_dim}, kv_dim={kv_dim}, "
            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}."
        )

    # Build a fresh module containing the placed devices.
    with mlir_mod_ctx() as ctx:
        _emit_r_weighted_rms_norm_seg()
        _emit_rk_rope_seg()
        _emit_rq_rope_seg()
        _emit_v_matmul_seg()
        _emit_k_matmul_seg()
        _emit_q_matmul_seg()
        _emit_dispatcher_device()
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    placed_text = str(module)
    placed_rms = _extract_single_device(placed_text, "r_weighted_rms_norm_seg")
    placed_rk_rope = _extract_single_device(placed_text, "rk_rope_seg")
    placed_rq_rope = _extract_single_device(placed_text, "rq_rope_seg")
    placed_v_matmul = _extract_single_device(placed_text, "v_matmul_seg")
    placed_k_matmul = _extract_single_device(placed_text, "k_matmul_seg")
    placed_q_matmul = _extract_single_device(placed_text, "q_matmul_seg")
    placed_dispatcher = _extract_dispatcher_device(placed_text)

    # Load the cached prefill MLIR and splice in the placed devices.
    project_root = Path(__file__).resolve().parents[1]
    cached_path = project_root / "reference_mlir" / "rms_gemms_rope.npu.air.mlir"
    cached_text = cached_path.read_text()
    original_len = len(cached_text)

    spliced = _splice_device(cached_text, "r_weighted_rms_norm_seg", placed_rms)
    spliced = _splice_device(spliced, "rk_rope_seg", placed_rk_rope)
    spliced = _splice_device(spliced, "rq_rope_seg", placed_rq_rope)
    spliced = _splice_device(spliced, "v_matmul_seg", placed_v_matmul)
    spliced = _splice_device(spliced, "k_matmul_seg", placed_k_matmul)
    spliced = _splice_device(spliced, "q_matmul_seg", placed_q_matmul)
    spliced = _splice_dispatcher_device(spliced, placed_dispatcher)

    if verbose:
        print(f"  [rms_gemms_rope builder] Spliced placed-IRON "
              f"r_weighted_rms_norm_seg + rk_rope_seg + rq_rope_seg + "
              f"v/k/q_matmul_seg + dispatcher into cached MLIR "
              f"({original_len} -> {len(spliced)} bytes).")

    return spliced


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs cached MLIR).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="Output path (default: stdout)",
                        default=None)
    parser.add_argument("--device-only", action="store_true",
                        help="Emit just the placed device, not the spliced module")
    args = parser.parse_args()
    if args.device_only:
        with mlir_mod_ctx() as ctx:
            _emit_r_weighted_rms_norm_seg()
            _emit_rk_rope_seg()
            _emit_rq_rope_seg()
            _emit_v_matmul_seg()
            _emit_k_matmul_seg()
            _emit_q_matmul_seg()
            _emit_dispatcher_device()
            mod = ctx.module
        text = str(mod)
    else:
        text = build_rms_gemms_rope_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
