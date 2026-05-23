# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b prefill RMS+GEMMS+RoPE kernel.

Phase 4.5a scope: emit the ``@r_weighted_rms_norm_seg`` device on placed
IRON and splice it into the cached ``rms_gemms_rope.npu.air.mlir`` text,
leaving the other 6 devices (4 GEMM segments, 2 RoPE segments, 1
dispatcher) untouched.  Subsequent phases (4.5b/c/d/e) extend the
splice to the remaining devices.

Splice mechanism::

    cached_text =
      aie.device @rk_rope_seg { ... }
      aie.device @rq_rope_seg { ... }
      aie.device @v_matmul_seg { ... }
      aie.device @k_matmul_seg { ... }
      aie.device @q_matmul_seg { ... }
      aie.device @r_weighted_rms_norm_seg { ... }   <-- REPLACED
      aie.device (dispatcher) { ... }

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
from aie.ir import AffineDimExpr, AffineMap, UnitAttr

from ._emit import bf16_memref, bf16_np


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

    Phase 4.5b: the ``r_weighted_rms_norm_seg``, ``rk_rope_seg``, and
    ``rq_rope_seg`` devices are emitted via placed-IRON; the other 4
    devices (3 GEMM segments + 1 dispatcher) come from the cached MLIR
    text via string splice.

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
        module = ctx.module

    placed_text = str(module)
    placed_rms = _extract_single_device(placed_text, "r_weighted_rms_norm_seg")
    placed_rk_rope = _extract_single_device(placed_text, "rk_rope_seg")
    placed_rq_rope = _extract_single_device(placed_text, "rq_rope_seg")

    # Load the cached prefill MLIR and splice in the placed devices.
    project_root = Path(__file__).resolve().parents[1]
    cached_path = project_root / "reference_mlir" / "rms_gemms_rope.npu.air.mlir"
    cached_text = cached_path.read_text()
    original_len = len(cached_text)

    spliced = _splice_device(cached_text, "r_weighted_rms_norm_seg", placed_rms)
    spliced = _splice_device(spliced, "rk_rope_seg", placed_rk_rope)
    spliced = _splice_device(spliced, "rq_rope_seg", placed_rq_rope)

    if verbose:
        print(f"  [rms_gemms_rope builder] Spliced placed-IRON "
              f"r_weighted_rms_norm_seg + rk_rope_seg + rq_rope_seg into "
              f"cached MLIR ({original_len} -> {len(spliced)} bytes).")

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
            mod = ctx.module
        text = str(mod)
    else:
        text = build_rms_gemms_rope_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
