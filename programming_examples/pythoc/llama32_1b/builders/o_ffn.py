# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b prefill o_ffn kernel.

Phase 4.6e scope: 7 of 9 devices are emitted from placed IRON --
``@rm_weighted_rms_norm_seg`` (4.6a), ``@ra_add_seg`` + ``@fa_add_seg``
(4.6b), ``@sw_silu_mul_seg`` (4.6c), ``@og_matmul_seg`` (4.6d),
``@gg_matmul_seg`` (4.6e), and the outer unnamed dispatcher
``aie.device(npu2)`` that hosts ``aiex.runtime_sequence @o_ffn``
(4.6f).  The 2 remaining GEMM devices (``ug_matmul_seg``,
``dg_matmul_seg``) come from the cached ``o_ffn.npu.air.mlir`` text
via string splice; phases 4.6f-g are deferred.

Splice mechanism::

    cached_text =
      aie.device @fa_add_seg { ... }                  <-- REPLACED (4.6b)
      aie.device @dg_matmul_seg { ... }               <-- cached (deferred)
      aie.device @sw_silu_mul_seg { ... }             <-- REPLACED (4.6c)
      aie.device @ug_matmul_seg { ... }               <-- cached (deferred)
      aie.device @gg_matmul_seg { ... }               <-- REPLACED (4.6e)
      aie.device @rm_weighted_rms_norm_seg { ... }    <-- REPLACED (4.6a)
      aie.device @ra_add_seg { ... }                  <-- REPLACED (4.6b)
      aie.device @og_matmul_seg { ... }               <-- REPLACED (4.6d)
      aie.device (dispatcher) { ... }                 <-- REPLACED (4.6f)

The cached ``rm_weighted_rms_norm_seg`` (lines 25351-26580 of
``reference_mlir/o_ffn.npu.air.mlir``, 1230 lines) is structurally
identical to the rms_gemms_rope ``@r_weighted_rms_norm_seg`` already
landed by Phase 4.5a; see the docstring on ``_emit_rm_weighted_rms_norm_seg``
below for details.

The cached ``ra_add_seg`` (lines 26581-27547, 967 lines) and
``fa_add_seg`` (lines 11-977, 967 lines) are byte-identical except for
device sym name, shim channel ids, and runtime_sequence host arg
indices.  Both implement a 1x8 herd of compute tiles, each performing
inline ``arith.addf`` over ping-pong 2048xbf16 L1 buffers, with 6 locks
per tile (all ping-pong: init pattern 2,0,2,0,2,0), 6 buffers per tile
(2 inputs + output, ping/pong), and 3 DMA channels per col (2 inputs +
1 output).

The cached ``sw_silu_mul_seg`` (lines 8753-9576, 824 lines) is a 1x8
strip of compute tiles, each calling the ``silu_and_mul_bf16``
PythoC kernel twice per outer iteration (ping/pong) over 4096xbf16 L1
buffers.  The locks (6 per tile, init pattern 2,0,2,0,2,0) and buffers
(6 per tile: 3 ping + 3 pong) follow the same layout as the add
devices.  Inputs: arg8 (gate), arg10 (up); output: arg11 (silu*up).
Channels: 54 = MM2S 0 (gate), 55 = MM2S 1 (up), 56 = S2MM 0 (out).

Channel ids and host args per add device:

  device          chan_in1  chan_in2  chan_out    in1_arg  in2_arg  out_arg
  ra_add_seg      16        17        18           arg2     arg3     arg4
  fa_add_seg      73        74        75           arg13    arg4     arg14
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

# Reuse the splice/extract helpers from rms_gemms_rope -- they're
# device-agnostic (brace-counting parser keyed on the device sym name,
# or on the unnamed-dispatcher signature).
from .rms_gemms_rope import (
    _extract_dispatcher_device,
    _extract_single_device,
    _splice_device,
    _splice_dispatcher_device,
)


# ---------------------------------------------------------------------------
# Constants matching the cached o_ffn AIR-stitched IR for Llama-3.2-1B
# prefill.
# ---------------------------------------------------------------------------
EMB_DIM = 2048             # model hidden size
SEQ_LEN = 2048             # prefill sequence length
HIDDEN_DIM = 8192          # FFN intermediate size
N_COLS = 8                 # 8 compute tile columns (1x8 herd)
VEC_LANES = 16             # bf16 vector lane width
SCRATCH_LEN = 16           # scratch buffer length (= VEC_LANES)
N_OUTER_STEPS = 256        # outer scf.for upper bound
OUTER_STEP = 2             # step (so 128 inner iters)

# Shim channel ids used by rm_weighted_rms_norm_seg in the cached o_ffn IR.
_CHAN_GAMMA = 19           # MM2S 0 on shim_0_0
_CHAN_X     = 20           # MM2S 0 (cols 1-7) / MM2S 1 (col 0)
_CHAN_Y     = 21           # S2MM 0 per col


# ---------------------------------------------------------------------------
# Host arg layout for the @rm_weighted_rms_norm_seg_sequence (15 args).
# Verified against the cached IR's runtime_sequence at line 26476:
#
#   arg0 : 2048x2048 bf16    (unused by this device)
#   arg1 : 2048x2048 bf16    (unused)
#   arg2 : 2048x2048 bf16    (unused)
#   arg3 : 2048x2048 bf16    (unused)
#   arg4 : 2048x2048 bf16    X INPUT  (post-attn residual)  <-- READ
#   arg5 : 2048      bf16    GAMMA (RMSNorm weight)         <-- READ
#   arg6 : 2048x2048 bf16    Y OUTPUT (normed FFN input)    <-- WRITE
#   arg7 : 2048x8192 bf16    (unused -- Wgate)
#   arg8 : 2048x8192 bf16    (unused -- Wup)
#   arg9 : 2048x8192 bf16    (unused)
#   arg10: 2048x8192 bf16    (unused)
#   arg11: 2048x8192 bf16    (unused)
#   arg12: 8192x2048 bf16    (unused -- Wdown)
#   arg13: 2048x2048 bf16    (unused)
#   arg14: 4194304   bf16    (unused -- flat work buffer)
def _o_ffn_host_arg_types():
    return [
        bf16_np(EMB_DIM, EMB_DIM),       # arg0
        bf16_np(EMB_DIM, EMB_DIM),       # arg1
        bf16_np(EMB_DIM, EMB_DIM),       # arg2
        bf16_np(EMB_DIM, EMB_DIM),       # arg3
        bf16_np(EMB_DIM, EMB_DIM),       # arg4  (X input)
        bf16_np(EMB_DIM,),               # arg5  (gamma)
        bf16_np(EMB_DIM, EMB_DIM),       # arg6  (Y output)
        bf16_np(EMB_DIM, HIDDEN_DIM),    # arg7
        bf16_np(EMB_DIM, HIDDEN_DIM),    # arg8
        bf16_np(EMB_DIM, HIDDEN_DIM),    # arg9
        bf16_np(EMB_DIM, HIDDEN_DIM),    # arg10
        bf16_np(EMB_DIM, HIDDEN_DIM),    # arg11
        bf16_np(HIDDEN_DIM, EMB_DIM),    # arg12
        bf16_np(EMB_DIM, EMB_DIM),       # arg13
        bf16_np(4194304,),               # arg14
    ]


# ---------------------------------------------------------------------------
# Emit the @rm_weighted_rms_norm_seg device.
# ---------------------------------------------------------------------------
def _emit_rm_weighted_rms_norm_seg() -> None:
    """Emit the placed-IRON rm_weighted_rms_norm_seg device.

    Must be called inside an ``mlir_mod_ctx()``; registers one
    ``aie.device(npu2) @rm_weighted_rms_norm_seg`` op.
    """
    from aie.dialects import memref as memref_dialect
    from aie.dialects import vector as vector_dialect
    from aie.dialects._vector_enum_gen import CombiningKind
    from aie.dialects import math as math_dialect
    from aie.extras import types as T

    @device(AIEDevice.npu2, sym_name="rm_weighted_rms_norm_seg")
    def _dev():
        # 8 shim tiles + 8 compute tiles (1x8 herd).
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Locks per compute tile -- 6 ids descending (5..0), in AIR's order.
        # Init values per cached IR (lines 25368-25415):
        #   id=5 (x_avail)   init=2
        #   id=4 (x_ready)   init=0
        #   id=3 (w_avail)   init=1
        #   id=2 (w_ready)   init=0
        #   id=1 (y_done)    init=2
        #   id=0 (y_full)    init=0
        # Cached order: col 0 first, then col 1, ..., col 7 (ascending).
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

        # Buffers per compute tile.  AIR emit order per tile (top->bottom):
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

        # External buffers (opaque AIR metadata; lines 25472-25474).
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

        # aie.core body per compute tile.  Inline RMSNorm math.
        def _make_core_body(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                # Constants (preamble matches cached order).
                cst_zero_bf16 = arith.constant(0.0, T.bf16())
                cst_norm_div = arith.constant(2048.0, T.bf16())
                cst_eps = arith.constant(1.001360e-05, T.bf16())
                vec16_ty = T.vector(VEC_LANES, T.bf16())
                np_zero = np.zeros((VEC_LANES,), dtype=bfloat16)
                cst_vec_zero = arith.constant(np_zero, vec16_ty)
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

        # Flows.  Cached order (lines 26435-26458):
        #   shim_0_0 DMA 0 -> tile_X_2 DMA 0  (gamma broadcast, 8 flows)
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 0)
        #   shim_0_0 DMA 1 -> tile_0_2 DMA 1  (col 0 x input)
        flow(shim_tiles[0], WireBundle.DMA, 1,
             compute_tiles[0], WireBundle.DMA, 1)
        #   shim_C_0 DMA 0 -> tile_C_2 DMA 1 for C in 1..7  (per-col x input)
        for col in range(1, N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 1)
        #   tile_C_2 DMA 0 -> shim_C_0 DMA 0  (y output per col, 8 flows)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)

        # Shim allocations.  Cached order (lines 26459-26475):
        #   8 S2MM 0 (y out) air_channel_21_C  for C in 0..7
        #   1 MM2S 0 on shim_0_0  air_channel_19    (gamma broadcast)
        #   1 MM2S 1 on shim_0_0  air_channel_20_0  (col 0 x input)
        #   7 MM2S 0 on shim_C_0  air_channel_20_C  for C in 1..7
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_Y}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        shim_dma_allocation(
            f"air_channel_{_CHAN_GAMMA}", shim_tiles[0],
            DMAChannelDir.MM2S, 0)
        shim_dma_allocation(
            f"air_channel_{_CHAN_X}_0", shim_tiles[0],
            DMAChannelDir.MM2S, 1)
        for col in range(1, N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_X}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)

        # Runtime sequence.  Layout (per cached IR, lines 26476-26579):
        #   t_gamma  = MM2S 0 on shim_0_0:  arg5 -> all 8 cores (broadcast)
        #             dims = [(4, 512), (512, 1)]  len=2048
        #   t_x_C    = MM2S X on shim_C_0:  arg4 (X input slice)
        #             offset = C * 524288  len=524288
        #             dims = [(2, 262144), (512, 512), (512, 1)]
        #   t_y_C    = S2MM 0 on shim_C_0:  arg6 (Y output)
        #             offset = C * 524288  len=524288
        #             dims = [(2, 262144), (512, 512), (512, 1)]
        # Then 9 dma_free_task (gamma + 8 x inputs), 8 dma_await_task (y outputs).
        @runtime_sequence(*_o_ffn_host_arg_types(),
                          sym_name="rm_weighted_rms_norm_seg_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12, arg13, arg14):
            del arg0, arg1, arg2, arg3, arg7, arg8, arg9, arg10, arg11
            del arg12, arg13, arg14

            # Gamma broadcast (single task, shim_0_0 MM2S 0). Reads arg5.
            t_gamma = dma_configure_task_for(f"air_channel_{_CHAN_GAMMA}")
            with bds(t_gamma) as bd:
                with bd[0]:
                    dma_bd(arg5, offset=0, len=EMB_DIM,
                           dimensions=[(4, 512), (512, 1)])
                    EndOp()
            dma_start_task(t_gamma)

            # 8 X input tasks (per col).  X lives in arg4.
            x_tasks = []
            for col in range(N_COLS):
                offset = col * 524288
                t = dma_configure_task_for(f"air_channel_{_CHAN_X}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg4, offset=offset, len=524288,
                               dimensions=[(2, 262144), (512, 512), (512, 1)])
                        EndOp()
                dma_start_task(t)
                x_tasks.append(t)

            # 8 Y output tasks (per col).  Normed X output to arg6.
            y_tasks = []
            for col in range(N_COLS):
                offset = col * 524288
                t = dma_configure_task_for(f"air_channel_{_CHAN_Y}_{col}",
                                            issue_token=True)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg6, offset=offset, len=524288,
                               dimensions=[(2, 262144), (512, 512), (512, 1)])
                        EndOp()
                dma_start_task(t)
                y_tasks.append(t)

            # Free input tasks (gamma + 8 X tasks), in cached order.
            dma_free_task(t_gamma)
            for t in x_tasks:
                dma_free_task(t)
            # Await output tasks.
            for t in y_tasks:
                dma_await_task(t)


# ---------------------------------------------------------------------------
# Parameterized add-device emitter.
#
# The two add devices (ra_add_seg, fa_add_seg) are byte-identical except
# for: (a) device sym name, (b) shim channel ids, (c) host arg indices.
# Each device has 1x8 compute tiles with the following per-tile layout:
#
#   Locks (6 ids, all ping-pong init=(2,0,2,0,2,0)):
#     id=5 (init=2) in1_avail   -- mem S2MM 1 acq
#     id=4 (init=0) in1_ready   -- mem S2MM 1 rel; core acq
#     id=3 (init=2) in2_avail   -- mem S2MM 0 acq
#     id=2 (init=0) in2_ready   -- mem S2MM 0 rel; core acq
#     id=1 (init=2) out_done    -- mem MM2S 0 rel; core acq
#     id=0 (init=0) out_full    -- mem MM2S 0 acq; core rel
#
#   Buffers (6 per tile, top->bottom in cached, descending sym-id;
#   tile-iter order is col 7 first, then col 6, ..., col 0):
#     slot 0 (highest): in2_pong
#     slot 1:           in1_pong
#     slot 2:           out_pong
#     slot 3:           in2_ping
#     slot 4:           out_ping
#     slot 5 (lowest):  in1_ping
#
#   aie.mem block per tile (3 DMA channels):
#     MM2S 0 (out): bb1=out_ping, bb2=out_pong; lock id=0 acq, id=1 rel
#     S2MM 0 (in2): bb5=in2_ping, bb6=in2_pong; lock id=3 acq, id=2 rel
#     S2MM 1 (in1): bb8=in1_ping, bb9=in1_pong; lock id=5 acq, id=4 rel
#
#   aie.core body (cf.br ^bb1 infinite loop wrapping):
#     scf.for arg0 = 0 to 524288 step 4096 {
#       # ping iter
#       acq(out_done) x2; acq(in2_ready); acq(in1_ready)
#       scf.for i = 0 to 2048 step 16:
#         out_ping[i:i+16] = in2_ping[i:i+16] + in1_ping[i:i+16]
#       rel(in1_avail); rel(in2_avail)
#       # pong iter
#       acq(in2_ready); acq(in1_ready)
#       scf.for i = 0 to 2048 step 16:
#         out_pong[i:i+16] = in2_pong[i:i+16] + in1_pong[i:i+16]
#       rel(in1_avail); rel(in2_avail)
#       rel(out_full) x2
#     }
#
#   Flows (24 total):
#     8x: shim_C_0 DMA 0 -> tile_C_2 DMA 0  (in2)
#     8x: shim_C_0 DMA 1 -> tile_C_2 DMA 1  (in1)
#     8x: tile_C_2 DMA 0 -> shim_C_0 DMA 0  (out)
#
#   Shim allocations (24 total, cached order):
#     8x air_channel_<CHAN_OUT>_C  on shim_C_0  S2MM 0
#     8x air_channel_<CHAN_IN2>_C  on shim_C_0  MM2S 0
#     8x air_channel_<CHAN_IN1>_C  on shim_C_0  MM2S 1
#
#   Runtime sequence (24 dma_configure_task_for, 24 dma_start_task, 16
#   dma_free_task on inputs, 8 dma_await_task on outputs):
#     Each col gets a 524288-bf16 slice at offset col*524288 with dims
#     [(2, 262144), (512, 512), (512, 1)].  Output tasks use
#     issue_token = true.
# ---------------------------------------------------------------------------
_ADD_PER_COL_LEN = 524288
_ADD_PER_COL_OFFSET = 524288
_ADD_DIMS = [(2, 262144), (512, 512), (512, 1)]
_ADD_OUTER_UB = 524288
_ADD_OUTER_STEP = 4096

# Per-device channel ids and host arg indices (verified from cached IR).
#   ra_add_seg: lines 26581-27547, channels 16/17/18, args arg2/arg3/arg4
#   fa_add_seg: lines 11-977,       channels 73/74/75, args arg13/arg4/arg14
_RA_ADD_PARAMS = dict(
    sym="ra_add_seg",
    herd_name="ra_add_herd",
    chan_in2=16,  # in2 = first cached MM2S 0 channel  (arg2 for ra)
    chan_in1=17,  # in1 = MM2S 1 channel              (arg3 for ra)
    chan_out=18,  # out = S2MM 0 channel              (arg4 for ra)
    arg_in2_idx=2,
    arg_in1_idx=3,
    arg_out_idx=4,
)
_FA_ADD_PARAMS = dict(
    sym="fa_add_seg",
    herd_name="fa_add_herd",
    chan_in2=73,
    chan_in1=74,
    chan_out=75,
    arg_in2_idx=13,
    arg_in1_idx=4,
    arg_out_idx=14,
)


def _emit_add_device(*,
                     sym: str,
                     herd_name: str,
                     chan_in2: int,
                     chan_in1: int,
                     chan_out: int,
                     arg_in2_idx: int,
                     arg_in1_idx: int,
                     arg_out_idx: int) -> None:
    """Emit one ``aie.device(npu2) @<sym> { ... }`` add device.

    Must be called inside an active ``mlir_mod_ctx()``; registers one
    ``aie.device`` op named ``<sym>`` matching the cached AIR-stitched IR
    op-for-op (modulo SSA naming).
    """
    from aie.dialects import memref as memref_dialect
    from aie.dialects import vector as vector_dialect
    from aie.extras import types as T

    @device(AIEDevice.npu2, sym_name=sym)
    def _dev():
        # 8 shim tiles + 8 compute tiles (1x8 herd).
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Locks per compute tile.  Ids descending (5..0), init pattern
        # (2,0,2,0,2,0).  Lock-id semantics:
        #   id=5 (init=2) in1_avail
        #   id=4 (init=0) in1_ready
        #   id=3 (init=2) in2_avail
        #   id=2 (init=0) in2_ready
        #   id=1 (init=2) out_done
        #   id=0 (init=0) out_full
        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "in1_avail": lock(ct, lock_id=5, init=2),
                "in1_ready": lock(ct, lock_id=4, init=0),
                "in2_avail": lock(ct, lock_id=3, init=2),
                "in2_ready": lock(ct, lock_id=2, init=0),
                "out_done":  lock(ct, lock_id=1, init=2),
                "out_full":  lock(ct, lock_id=0, init=0),
            }

        # Buffers per compute tile.  Emit order top->bottom (descending
        # sym-id, descending col): bufN..bufN-5 -> in2_pong, in1_pong,
        # out_pong, in2_ping, out_ping, in1_ping.
        _BF16_2048_L1 = bf16_memref(EMB_DIM, memory_space=2)
        core_buf = {col: {} for col in range(N_COLS)}
        for col in reversed(range(N_COLS)):
            ct = compute_tiles[col]
            core_buf[col]["in2_pong"] = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["in1_pong"] = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["out_pong"] = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["in2_ping"] = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["out_ping"] = buffer(ct, datatype=_BF16_2048_L1)
            core_buf[col]["in1_ping"] = buffer(ct, datatype=_BF16_2048_L1)

        # External buffers (3, opaque AIR metadata; cached has these as
        # 2048x2048, 2048x2048, and either 2048x2048 (ra) or 4194304 (fa)).
        # The dispatcher routes the run-time args via aiex.run, so the
        # external_buffer decls here are inert metadata.  For diff parity
        # we keep the cached shapes per device: ra emits three 2048x2048
        # decls; fa emits two 2048x2048 + one 4194304.
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer")
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_1")
        if sym == "fa_add_seg":
            external_buffer(bf16_np(4194304,), name="__air_external_buffer_2")
        else:
            external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_2")

        # aie.mem block per compute tile.
        #   MM2S 0 (out): bb1=out_ping, bb2=out_pong; lock id=0 acq, id=1 rel
        #   S2MM 0 (in2): bb5=in2_ping, bb6=in2_pong; lock id=3 acq, id=2 rel
        #   S2MM 1 (in1): bb8=in1_ping, bb9=in1_pong; lock id=5 acq, id=4 rel
        def _make_core_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[4])
                with block[1]:
                    use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["out_ping"], offset=0, len=EMB_DIM)
                    use_lock(_cl["out_done"], LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["out_pong"], offset=0, len=EMB_DIM)
                    use_lock(_cl["out_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    EndOp()
                with block[4]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[7])
                with block[5]:
                    use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["in2_ping"], offset=0, len=EMB_DIM)
                    use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[6]:
                    use_lock(_cl["in2_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["in2_pong"], offset=0, len=EMB_DIM)
                    use_lock(_cl["in2_ready"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[3])
                with block[8]:
                    use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["in1_ping"], offset=0, len=EMB_DIM)
                    use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                    next_bd(block[9])
                with block[9]:
                    use_lock(_cl["in1_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["in1_pong"], offset=0, len=EMB_DIM)
                    use_lock(_cl["in1_ready"], LockAction.Release, value=1)
                    next_bd(block[8])

        for col in reversed(range(N_COLS)):
            _make_core_mem(compute_tiles[col], core_locks[col], core_buf[col])

        # aie.core body per compute tile.  Inline elementwise addf.
        def _make_core_body(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                # Constants -- cached preamble order.
                cst_zero_bf16 = arith.constant(0.0, T.bf16())
                vec16_ty = T.vector(VEC_LANES, T.bf16())
                perm = AffineMap.get(1, 0, [AffineDimExpr.get(0)])
                c0 = arith.constant(0, T.index())

                def _add_pass(in1_buf, in2_buf, out_buf):
                    """Inline elementwise add: out = in2 + in1 (cached order)."""
                    for i in range_(0, EMB_DIM, VEC_LANES):
                        sub_in2 = memref_dialect.subview(
                            in2_buf, [i], [VEC_LANES], [1])
                        sub_in1 = memref_dialect.subview(
                            in1_buf, [i], [VEC_LANES], [1])
                        sub_out = memref_dialect.subview(
                            out_buf, [i], [VEC_LANES], [1])
                        v_in2 = vector_dialect.transfer_read(
                            vec16_ty, sub_in2, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_in1 = vector_dialect.transfer_read(
                            vec16_ty, sub_in1, [c0],
                            permutation_map=perm, padding=cst_zero_bf16,
                            in_bounds=[True])
                        v_sum = arith.addf(v_in2, v_in1)
                        vector_dialect.transfer_write(
                            None, v_sum, sub_out, [c0],
                            permutation_map=perm, in_bounds=[True])

                # Infinite outer loop (cf.br ^bb1 in cached IR).
                for _outer in range_(_sys.maxsize):
                    # Outer scf.for: 0 to 524288 step 4096 (128 iters).
                    for _i_outer in range_(0, _ADD_OUTER_UB, _ADD_OUTER_STEP):
                        # Ping iter: drain out_done x2, fill in2_ready x1,
                        # fill in1_ready x1; compute; release inputs.
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        _add_pass(_bufs["in1_ping"],
                                  _bufs["in2_ping"],
                                  _bufs["out_ping"])
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        # Pong iter.
                        use_lock(_cl["in2_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["in1_ready"], LockAction.AcquireGreaterEqual, value=1)
                        _add_pass(_bufs["in1_pong"],
                                  _bufs["in2_pong"],
                                  _bufs["out_pong"])
                        use_lock(_cl["in1_avail"], LockAction.Release, value=1)
                        use_lock(_cl["in2_avail"], LockAction.Release, value=1)
                        # Output produced for both ping and pong.
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)

        for col in reversed(range(N_COLS)):
            _make_core_body(compute_tiles[col], core_locks[col], core_buf[col])

        # Flows (24 total, cached order):
        #   8x: shim_C_0 DMA 0 -> tile_C_2 DMA 0  (in2)
        #   8x: shim_C_0 DMA 1 -> tile_C_2 DMA 1  (in1)
        #   8x: tile_C_2 DMA 0 -> shim_C_0 DMA 0  (out)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 1,
                 compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)

        # Shim allocations (24 total, cached order):
        #   8x out (S2MM 0) then 8x in2 (MM2S 0) then 8x in1 (MM2S 1).
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_out}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_in2}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{chan_in1}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # Runtime sequence.  Cached order (matches both ra_add_seg and
        # fa_add_seg):
        #   8 in2 tasks (MM2S 0), 8 in1 tasks (MM2S 1), 8 out tasks
        #   (S2MM 0, issue_token=true).
        #   Then 16 dma_free_task (in2 + in1), 8 dma_await_task (out).
        @runtime_sequence(*_o_ffn_host_arg_types(),
                          sym_name=f"{sym}_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12, arg13, arg14):
            host_args = (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                         arg8, arg9, arg10, arg11, arg12, arg13, arg14)
            arg_in2 = host_args[arg_in2_idx]
            arg_in1 = host_args[arg_in1_idx]
            arg_out = host_args[arg_out_idx]

            # 8 in2-input tasks (MM2S 0).
            in2_tasks = []
            for col in range(N_COLS):
                offset = col * _ADD_PER_COL_OFFSET
                t = dma_configure_task_for(f"air_channel_{chan_in2}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_in2, offset=offset, len=_ADD_PER_COL_LEN,
                               dimensions=_ADD_DIMS)
                        EndOp()
                dma_start_task(t)
                in2_tasks.append(t)

            # 8 in1-input tasks (MM2S 1).
            in1_tasks = []
            for col in range(N_COLS):
                offset = col * _ADD_PER_COL_OFFSET
                t = dma_configure_task_for(f"air_channel_{chan_in1}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_in1, offset=offset, len=_ADD_PER_COL_LEN,
                               dimensions=_ADD_DIMS)
                        EndOp()
                dma_start_task(t)
                in1_tasks.append(t)

            # 8 out tasks (S2MM 0, issue_token=true).
            out_tasks = []
            for col in range(N_COLS):
                offset = col * _ADD_PER_COL_OFFSET
                t = dma_configure_task_for(f"air_channel_{chan_out}_{col}",
                                           issue_token=True)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_out, offset=offset, len=_ADD_PER_COL_LEN,
                               dimensions=_ADD_DIMS)
                        EndOp()
                dma_start_task(t)
                out_tasks.append(t)

            # Free 16 inputs (in2 then in1), then await 8 outputs.
            for t in in2_tasks:
                dma_free_task(t)
            for t in in1_tasks:
                dma_free_task(t)
            for t in out_tasks:
                dma_await_task(t)


def _emit_ra_add_seg() -> None:
    """Emit the placed-IRON ``@ra_add_seg`` device.

    Reads arg2 and arg3 (both 2048x2048 bf16), writes arg4 (2048x2048
    bf16).  Channels 16/17/18.
    """
    _emit_add_device(**_RA_ADD_PARAMS)


def _emit_fa_add_seg() -> None:
    """Emit the placed-IRON ``@fa_add_seg`` device.

    Reads arg13 and arg4 (both 2048x2048 bf16), writes arg14 (flat
    4194304 bf16, same total bytes).  Channels 73/74/75.
    """
    _emit_add_device(**_FA_ADD_PARAMS)


# ---------------------------------------------------------------------------
# @sw_silu_mul_seg device.
#
# Per cached IR (lines 8753-9576, 824 lines), 1x8 herd; each compute
# tile invokes the ``silu_and_mul_bf16`` external kernel twice per
# outer-loop iteration (ping/pong) over L1 buffers of 4096xbf16.
#
# Per-tile layout (matches sw_silu_mul cached order; SAME as add devices):
#   Locks (6 ids, init pattern 2,0,2,0,2,0):
#     id=5 (init=2) up_avail        -- mem S2MM 1 acq           (lock_C_2)
#     id=4 (init=0) up_ready        -- mem S2MM 1 rel; core acq (lock_C_2_n)
#     id=3 (init=2) gate_avail      -- mem S2MM 0 acq           (lock_C_2_n+1)
#     id=2 (init=0) gate_ready      -- mem S2MM 0 rel; core acq (lock_C_2_n+2)
#     id=1 (init=2) out_done        -- mem MM2S 0 rel; core acq (lock_C_2_n+3)
#     id=0 (init=0) out_full        -- mem MM2S 0 acq; core rel (lock_C_2_n+4)
#
#   Buffers (6 per tile, 4096xbf16, top->bottom in cached desc sym-id;
#   tile-iter order is col 7 first):
#     slot 0 (highest sym): gate_pong   (in2_pong)
#     slot 1:               up_pong     (in1_pong)
#     slot 2:               out_pong
#     slot 3:               gate_ping   (in2_ping)
#     slot 4:               out_ping
#     slot 5 (lowest sym):  up_ping     (in1_ping)
#
#   aie.mem block per tile (3 DMA channels):
#     MM2S 0 (out):   bb1=out_ping, bb2=out_pong; acq out_full, rel out_done
#     S2MM 0 (gate):  bb5=gate_ping, bb6=gate_pong; acq gate_avail, rel gate_ready
#     S2MM 1 (up):    bb8=up_ping, bb9=up_pong; acq up_avail, rel up_ready
#
#   aie.core body (cf.br ^bb1 infinite loop wrapping):
#     scf.for arg0 = 0 to 16777216 step 65536 {           # 256 iters
#       # ping iter
#       acq(out_done) x2; acq(gate_ready); acq(up_ready)
#       silu_and_mul_bf16(gate_ping, up_ping, out_ping, 4096)
#       rel(up_avail); rel(gate_avail)
#       # pong iter
#       acq(gate_ready); acq(up_ready)
#       silu_and_mul_bf16(gate_pong, up_pong, out_pong, 4096)
#       rel(up_avail); rel(gate_avail)
#       rel(out_full) x2
#     }
#
#   Flows (24 total): same shape as add devices.
#
#   Shim allocations (24 total, cached order):
#     8x S2MM 0 (chan 56, silu*up out)  on shim_C_0
#     8x MM2S 0 (chan 54, gate in)      on shim_C_0
#     8x MM2S 1 (chan 55, up in)        on shim_C_0
#
#   Runtime sequence:
#     8 gate-input tasks  (MM2S 0, channel 54, arg8)
#     8 up-input tasks    (MM2S 1, channel 55, arg10)
#     8 out tasks         (S2MM 0, channel 56, arg11, issue_token=true)
#     -> 16 dma_free_task (gate+up), 8 dma_await_task (out).
#
#   Per-col DMA: each col reads/writes 2097152 bf16 from a
#   2048x8192 buffer at offset col * 4096; dims
#   [(512, 32768), (8, 512), (512, 1)] -- 512 rows * 8 partial-cols *
#   512 inner = 2M bf16 elements.
# ---------------------------------------------------------------------------
_SILU_L1_LEN = 4096            # per-call element count (L1 buffer size)
_SILU_OUTER_UB = 16777216      # outer scf.for upper bound
_SILU_OUTER_STEP = 65536       # outer scf.for step (= 16 * 4096)

_SILU_PER_COL_LEN = 2097152
_SILU_PER_COL_OFFSET = 4096
_SILU_DIMS = [(512, 32768), (8, 512), (512, 1)]

# Per cached IR shim_dma_allocation block (lines 9406-9429).
_CHAN_SILU_GATE = 54   # MM2S 0
_CHAN_SILU_UP   = 55   # MM2S 1
_CHAN_SILU_OUT  = 56   # S2MM 0

# Host arg indices per cached runtime_sequence body (lines 9430-9575):
#   gate -> arg8, up -> arg10, out -> arg11.
_SILU_ARG_GATE = 8
_SILU_ARG_UP   = 10
_SILU_ARG_OUT  = 11


def _emit_sw_silu_mul_seg() -> None:
    """Emit the placed-IRON ``@sw_silu_mul_seg`` device.

    Must be called inside an active ``mlir_mod_ctx()``; registers one
    ``aie.device(npu2) @sw_silu_mul_seg`` op matching the cached
    AIR-stitched IR op-for-op (modulo SSA naming).
    """
    from aie.extras import types as T

    @device(AIEDevice.npu2, sym_name="sw_silu_mul_seg")
    def _dev():
        # 8 shim tiles + 8 compute tiles (1x8 herd).
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # Locks per compute tile.  Ids descending (5..0), init pattern
        # (2,0,2,0,2,0).  Lock-id semantics (see comment block above).
        core_locks = {}
        for col in range(N_COLS):
            ct = compute_tiles[col]
            core_locks[col] = {
                "up_avail":   lock(ct, lock_id=5, init=2),
                "up_ready":   lock(ct, lock_id=4, init=0),
                "gate_avail": lock(ct, lock_id=3, init=2),
                "gate_ready": lock(ct, lock_id=2, init=0),
                "out_done":   lock(ct, lock_id=1, init=2),
                "out_full":   lock(ct, lock_id=0, init=0),
            }

        # Buffers per compute tile.  Emit order top->bottom (descending
        # sym-id, descending col): bufN..bufN-5 -> gate_pong, up_pong,
        # out_pong, gate_ping, out_ping, up_ping.
        _BF16_4096_L1 = bf16_memref(_SILU_L1_LEN, memory_space=2)
        core_buf = {col: {} for col in range(N_COLS)}
        for col in reversed(range(N_COLS)):
            ct = compute_tiles[col]
            core_buf[col]["gate_pong"] = buffer(ct, datatype=_BF16_4096_L1)
            core_buf[col]["up_pong"]   = buffer(ct, datatype=_BF16_4096_L1)
            core_buf[col]["out_pong"]  = buffer(ct, datatype=_BF16_4096_L1)
            core_buf[col]["gate_ping"] = buffer(ct, datatype=_BF16_4096_L1)
            core_buf[col]["out_ping"]  = buffer(ct, datatype=_BF16_4096_L1)
            core_buf[col]["up_ping"]   = buffer(ct, datatype=_BF16_4096_L1)

        # External buffers (3, opaque AIR metadata; all 2048x8192 in cached).
        external_buffer(bf16_np(EMB_DIM, HIDDEN_DIM),
                        name="__air_external_buffer")
        external_buffer(bf16_np(EMB_DIM, HIDDEN_DIM),
                        name="__air_external_buffer_1")
        external_buffer(bf16_np(EMB_DIM, HIDDEN_DIM),
                        name="__air_external_buffer_2")

        # aie.mem block per compute tile.
        #   MM2S 0 (out):  bb1=out_ping,  bb2=out_pong;  acq out_full, rel out_done
        #   S2MM 0 (gate): bb5=gate_ping, bb6=gate_pong; acq gate_avail, rel gate_ready
        #   S2MM 1 (up):   bb8=up_ping,   bb9=up_pong;   acq up_avail, rel up_ready
        def _make_core_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[4])
                with block[1]:
                    use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["out_ping"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["out_done"], LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(_cl["out_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["out_pong"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["out_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    EndOp()
                with block[4]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[5], chain=block[7])
                with block[5]:
                    use_lock(_cl["gate_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["gate_ping"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["gate_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[6]:
                    use_lock(_cl["gate_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["gate_pong"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["gate_ready"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[3])
                with block[8]:
                    use_lock(_cl["up_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["up_ping"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["up_ready"], LockAction.Release, value=1)
                    next_bd(block[9])
                with block[9]:
                    use_lock(_cl["up_avail"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["up_pong"], offset=0, len=_SILU_L1_LEN)
                    use_lock(_cl["up_ready"], LockAction.Release, value=1)
                    next_bd(block[8])

        for col in reversed(range(N_COLS)):
            _make_core_mem(compute_tiles[col], core_locks[col], core_buf[col])

        # Declare @silu_and_mul_bf16 external function (cached emits it
        # after the cores but before the flows -- placement here is
        # functionally equivalent and the assign-core-link-files pass
        # routes link_with onto each core).
        _BF16_4096_L1_ty = bf16_memref(_SILU_L1_LEN, memory_space=2)
        silu_fn = external_func(
            "silu_and_mul_bf16",
            inputs=[_BF16_4096_L1_ty, _BF16_4096_L1_ty, _BF16_4096_L1_ty,
                    np.int32],
            link_with="silu_and_mul_bf16.o",
        )
        silu_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # aie.core body per compute tile.
        def _make_core_body(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                len_c = arith.constant(_SILU_L1_LEN, T.i32())  # noqa: F841

                # Infinite outer loop (cf.br ^bb1 in cached IR).
                for _outer in range_(_sys.maxsize):
                    for _inner in range_(0, _SILU_OUTER_UB, _SILU_OUTER_STEP):
                        # Ping iter: drain out_done x2, fill gate_ready + up_ready,
                        # compute, release inputs.
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["out_done"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["gate_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["up_ready"], LockAction.AcquireGreaterEqual, value=1)
                        silu_fn(_bufs["gate_ping"], _bufs["up_ping"],
                                _bufs["out_ping"], len_c)
                        use_lock(_cl["up_avail"], LockAction.Release, value=1)
                        use_lock(_cl["gate_avail"], LockAction.Release, value=1)
                        # Pong iter.
                        use_lock(_cl["gate_ready"], LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_cl["up_ready"], LockAction.AcquireGreaterEqual, value=1)
                        silu_fn(_bufs["gate_pong"], _bufs["up_pong"],
                                _bufs["out_pong"], len_c)
                        use_lock(_cl["up_avail"], LockAction.Release, value=1)
                        use_lock(_cl["gate_avail"], LockAction.Release, value=1)
                        # Output produced for both ping and pong.
                        use_lock(_cl["out_full"], LockAction.Release, value=1)
                        use_lock(_cl["out_full"], LockAction.Release, value=1)

        for col in reversed(range(N_COLS)):
            _make_core_body(compute_tiles[col], core_locks[col], core_buf[col])

        # Flows (24 total, cached order):
        #   8x: shim_C_0 DMA 0 -> tile_C_2 DMA 0   (gate)
        #   8x: shim_C_0 DMA 1 -> tile_C_2 DMA 1   (up)
        #   8x: tile_C_2 DMA 0 -> shim_C_0 DMA 0   (out)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0,
                 compute_tiles[col], WireBundle.DMA, 0)
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 1,
                 compute_tiles[col], WireBundle.DMA, 1)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0,
                 shim_tiles[col], WireBundle.DMA, 0)

        # Shim allocations (24 total, cached order):
        #   8x out (S2MM 0, chan 56), 8x gate (MM2S 0, chan 54),
        #   8x up (MM2S 1, chan 55).
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_SILU_OUT}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_SILU_GATE}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_SILU_UP}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # Runtime sequence.  Cached order:
        #   8 gate tasks (MM2S 0, arg8), 8 up tasks (MM2S 1, arg10),
        #   8 out tasks (S2MM 0, arg11, issue_token=true).
        #   Then 16 dma_free_task (gate + up), 8 dma_await_task (out).
        @runtime_sequence(*_o_ffn_host_arg_types(),
                          sym_name="sw_silu_mul_seg_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12, arg13, arg14):
            host_args = (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                         arg8, arg9, arg10, arg11, arg12, arg13, arg14)
            arg_gate = host_args[_SILU_ARG_GATE]
            arg_up   = host_args[_SILU_ARG_UP]
            arg_out  = host_args[_SILU_ARG_OUT]

            # 8 gate-input tasks (MM2S 0).
            gate_tasks = []
            for col in range(N_COLS):
                offset = col * _SILU_PER_COL_OFFSET
                t = dma_configure_task_for(
                    f"air_channel_{_CHAN_SILU_GATE}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_gate, offset=offset,
                               len=_SILU_PER_COL_LEN,
                               dimensions=_SILU_DIMS)
                        EndOp()
                dma_start_task(t)
                gate_tasks.append(t)

            # 8 up-input tasks (MM2S 1).
            up_tasks = []
            for col in range(N_COLS):
                offset = col * _SILU_PER_COL_OFFSET
                t = dma_configure_task_for(
                    f"air_channel_{_CHAN_SILU_UP}_{col}")
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_up, offset=offset,
                               len=_SILU_PER_COL_LEN,
                               dimensions=_SILU_DIMS)
                        EndOp()
                dma_start_task(t)
                up_tasks.append(t)

            # 8 out tasks (S2MM 0, issue_token=true).
            out_tasks = []
            for col in range(N_COLS):
                offset = col * _SILU_PER_COL_OFFSET
                t = dma_configure_task_for(
                    f"air_channel_{_CHAN_SILU_OUT}_{col}", issue_token=True)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(arg_out, offset=offset,
                               len=_SILU_PER_COL_LEN,
                               dimensions=_SILU_DIMS)
                        EndOp()
                dma_start_task(t)
                out_tasks.append(t)

            # Free 16 inputs (gate then up), then await 8 outputs.
            for t in gate_tasks:
                dma_free_task(t)
            for t in up_tasks:
                dma_free_task(t)
            for t in out_tasks:
                dma_await_task(t)


# ---------------------------------------------------------------------------
# og_matmul_seg device (Phase 4.6d -- O-projection GEMM of o_ffn).
# ---------------------------------------------------------------------------
# Structurally mirrors the rms_gemms_rope @v_matmul_seg device (8x4 herd,
# 8 mem tiles with asymmetric lock counts: cols 0-3 = 10 locks, cols 4-7
# = 6 locks).  Per-tile compute structure is identical to v's except for
# L1 buffer shapes (half the C and B sizes) and the core body's loop
# structure (nested 8 outer x 4 ping-pong vs v's flat 32 ping-pong).
#
# Per compute tile (32 total):
#   * 5 buffers in L1 (memory_space=2):
#       buf_C       memref<1x1x8x8x8x8 xbf16, 2>      (C-out, 4096 elts)
#       buf_A_pong  memref<1x1x4x8x8x8  xbf16, 2>     (A-in pong, 2048 elts)
#       buf_B_pong  memref<1x1x8x4x8x8  xbf16, 2>     (B-in pong, 2048 elts)
#       buf_B_ping  memref<1x1x8x4x8x8  xbf16, 2>     (B-in ping, 2048 elts)
#       buf_A_ping  memref<1x1x4x8x8x8  xbf16, 2>     (A-in ping, 2048 elts)
#     NOTE: cached IR emits these in order (C, A_pong, B_pong, B_ping,
#     A_ping); the labeling here matches the lock pattern in the cached
#     aie.mem block (bb4=A_ping, bb5=A_pong, bb7=B_ping, bb8=B_pong).
#   * 6 locks at ids 5..0, init=(2, 0, 2, 0, 1, 0) -- same as v's compute
#     tiles.
#   * aie.mem block (3 DMA channels, same structure as v's):
#       MM2S 0 (C out): buf_C self-loop, len 4096, 3D dims
#       S2MM 0 (A in):  buf_A_ping <-> buf_A_pong ping-pong, len 2048
#       S2MM 1 (B in):  buf_B_ping <-> buf_B_pong ping-pong, len 2048
#   * aie.core body:
#       acquire C_done x1
#       zero buf_C (nested 8x8 over 8x8 micro-tiles)
#       for m_outer in 0..8:
#         for n_inner_pair in 0..4:        # 4 ping/pong pairs per outer
#           acquire A_ready, B_ready; gemm(A_ping, B_ping, C); release B_sem, A_sem
#           acquire A_ready, B_ready; gemm(A_pong, B_pong, C); release B_sem, A_sem
#       release C_full x1
#       cf.br ^bb1
#
# Per mem tile (8 total): same asymmetric lock + DMA channel layout as
# v's mem tiles.  Buffer shapes differ:
#   cols 0..7: 1 W-weight buf (1x4x64x64 bf16, L2, 16384 elts)
#   cols 0..7: 2 C-out bufs   (1x1x64x256 bf16, L2, 16384 elts each)   ping/pong
#   cols 0..3: 2 X-input bufs (1x1x256x64 bf16, L2, 16384 elts each)   ping/pong
#
# aie.memtile_dma per col:
#   cols 0-3 -- 9 channels (with X chain):
#       MM2S 0: W-weight broadcast (self-loop, len 16384, 3D dims)
#       MM2S 1: C-out ping/pong (len 16384 each, 3D dims)
#       MM2S 2: X-broadcast ping/pong (len 16384 each, 4D dims)
#       S2MM 0: C-in ping/pong (mirror of MM2S 1)
#       S2MM 1: X-in ping/pong (mirror of MM2S 2)
#       S2MM 2..5: W-weight slices (4 chans, each 1-BD self-loop at
#                  offset = slice * 4096, len 4096 -- half of v's 8192)
#   cols 4-7 -- 7 channels (no X chain):
#       MM2S 0, MM2S 1, S2MM 0, S2MM 1..4 (W-weight slices)
#
# Flows (116 total): same shape as v_matmul.
#
# Shim allocations (20 total):
#   air_channel_78_C  MM2S 0 on shim_C_0  (X input  = arg0,  8 cols)
#   air_channel_77_C  MM2S 1 on shim_C_0  (W weight = arg1,  cols 0-3)
#   air_channel_84_C  S2MM 0 on shim_C_0  (Y output = arg2,  8 cols)
#
# Runtime sequence: 32 X-tasks (4 per col), 16 W-tasks (4 per col 0-3),
# 8 Y-tasks (1 per col).  X & W use repeat_count=7 (8 inner iters).
# Y tasks use issue_token=true + repeat_count=3 (4 inner iters).  Free
# order (cached): 8 awaits (reverse), then 48 frees in reverse-col
# groups of 4 (W cols 3->0, then X cols 7->0).
OG_MATMUL_M_OUTER = 8              # outer M loop bound per core dispatch
OG_MATMUL_N_INNER_HALF = 4         # inner-N step (= 8/2; 2 calls per step)
OG_MATMUL_C_M = 8                  # buf_C outer-M dim (8 m-blocks of 8x8)
OG_MATMUL_C_N = 8                  # buf_C outer-N dim (8 n-blocks of 8x8)

# Channel ids for og_matmul_seg in the cached IR.
_CHAN_OG_X      = 78               # MM2S 0 (X input) on shim_C_0
_CHAN_OG_WEIGHT = 77               # MM2S 1 (W weight) on shim_C_0
_CHAN_OG_Y      = 84               # S2MM 0 (Y output) on shim_C_0


def _emit_og_matmul_seg() -> None:
    """Emit the placed-IRON og_matmul_seg device (O-projection GEMM).

    Must be called inside an mlir_mod_ctx().  Registers one
    ``aie.device(npu2) @og_matmul_seg`` op.  Body-level divergence from
    cached: each inner ping/pong call invokes
    ``bf16_gemm_kernel_bf16out`` on the L1 (A, B, C) buffer triple
    instead of inlining a 4-MAC vector.contract chain.  Infrastructure
    (tiles, locks, buffers, flows, memtile_dma BD chains, shim allocs,
    runtime_sequence) matches the cached verbatim.

    Host args (verified at cached runtime_sequence @og_matmul_seg_sequence,
    line 34952):
      arg0: 2048x2048 bf16  X input  (attention-output residual, post-flash-attn)
      arg1: 2048x2048 bf16  W weight (O-projection Wo)
      arg2: 2048x2048 bf16  Y output (target buffer)
    """
    from aie.dialects import memref as memref_dialect
    from aie.dialects import vector as vector_dialect
    from aie.extras import types as T
    from aie.ir import UnitAttr

    @device(AIEDevice.npu2, sym_name="og_matmul_seg")
    def _dev():
        # 8 shim + 8 mem + 32 compute tiles (cols 0..7, rows 2..5).
        shim_tiles    = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles     = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = {}
        for col in range(N_COLS):
            for row in range(2, 6):
                compute_tiles[(col, row)] = tile(col, row)

        # ------------------------------------------------------------
        # Locks.  Cached emit order: mem tiles col 7 first (6 locks),
        # then 6 5 4 (6 each), then 3 2 1 0 (10 each).  Then compute
        # tiles row 2 col 0..7, row 3 col 0..7, ..., row 5 col 0..7
        # (ascending row-major).
        # ------------------------------------------------------------
        mem_locks = {}
        for col in reversed(range(4, N_COLS)):    # 7, 6, 5, 4 (6 locks each)
            mt = mem_tiles[col]
            mem_locks[col] = {
                "C_pong_sem":   lock(mt, lock_id=5, init=1),
                "C_pong_ready": lock(mt, lock_id=4, init=0),
                "C_ping_sem":   lock(mt, lock_id=3, init=1),
                "C_ping_ready": lock(mt, lock_id=2, init=0),
                "W_sem":        lock(mt, lock_id=1, init=4),
                "W_ready":      lock(mt, lock_id=0, init=0),
            }
        for col in reversed(range(4)):            # 3, 2, 1, 0 (10 locks each)
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

        # Compute-tile locks: row-major ascending (row 2 cols 0..7,
        # row 3 cols 0..7, ..., row 5 cols 0..7).
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
        # Buffers.  Mem-tile buffers first (col 0..7 ascending per
        # category), then compute-tile buffers (descending row, then
        # descending col).
        # ------------------------------------------------------------
        # Cached mem-tile emit order:
        #   buf191..buf184 : 1x4x64x64  (W-weight, col 0..7)
        #   buf183..buf168 : 1x1x64x256 (C-out ping/pong, 2 per col,
        #                                cols 0..7 ascending)
        #   buf167..buf160 : 1x1x256x64 (X-input ping/pong, 2 per col,
        #                                cols 0..3 ascending)
        BF16_W_L2  = bf16_memref(1, 4, 64, 64,  memory_space=1)
        BF16_CO_L2 = bf16_memref(1, 1, 64, 256, memory_space=1)
        BF16_XI_L2 = bf16_memref(1, 1, 256, 64, memory_space=1)

        mem_buf = {col: {} for col in range(N_COLS)}
        # W-weight: 1 per col, col order 0..7.
        for col in range(N_COLS):
            mem_buf[col]["W"] = buffer(mem_tiles[col], datatype=BF16_W_L2)
        # C-out ping/pong: 2 per col, order col 0 (ping then pong), col 1, ...
        for col in range(N_COLS):
            mem_buf[col]["C_ping"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
            mem_buf[col]["C_pong"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
        # X-input ping/pong: 2 each on cols 0..3 only.
        for col in range(4):
            mem_buf[col]["X_ping"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)
            mem_buf[col]["X_pong"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)

        # Compute-tile buffers.  Cached emit order: row 5 col 7 first,
        # then row 5 col 6, ..., down to row 2 col 0.  Per tile, the
        # cached declaration order is (C, A_pong, B_pong, B_ping,
        # A_ping) -- different from v_matmul's (C, A_pong, B_ping,
        # A_ping, B_pong).  We follow og's order here so the emitted
        # MLIR's buffer-decl sequence matches the cached IR sym-for-sym.
        BF16_C_L1 = bf16_memref(1, 1, 8, 8, 8, 8, memory_space=2)
        BF16_A_L1 = bf16_memref(1, 1, 4, 8, 8, 8, memory_space=2)
        BF16_B_L1 = bf16_memref(1, 1, 8, 4, 8, 8, memory_space=2)

        core_buf = {}
        for row in reversed(range(2, 6)):
            for col in reversed(range(N_COLS)):
                ct = compute_tiles[(col, row)]
                bufs = {}
                bufs["C"]      = buffer(ct, datatype=BF16_C_L1)
                bufs["A_pong"] = buffer(ct, datatype=BF16_A_L1)
                bufs["B_pong"] = buffer(ct, datatype=BF16_B_L1)
                bufs["B_ping"] = buffer(ct, datatype=BF16_B_L1)
                bufs["A_ping"] = buffer(ct, datatype=BF16_A_L1)
                core_buf[(col, row)] = bufs

        # External buffers (opaque AIR metadata; kept for diff parity).
        # og uses (2048x2048, 2048x2048, 2048x2048) -- the X/W/Y triple.
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer")
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_1")
        external_buffer(bf16_np(EMB_DIM, EMB_DIM), name="__air_external_buffer_2")

        # ------------------------------------------------------------
        # aie.mem blocks per compute tile.  Cached order: row 5 col 7
        # first, then col 6 .., row 4 col 7 .., ..., row 2 col 0.
        # ------------------------------------------------------------
        def _make_compute_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                # MM2S 0: C-out self-loop, len 4096 elts (= 8192 bytes).
                # 3D dims: [<size=64, stride=8>, <size=8, stride=512>,
                #           <size=8, stride=1>]  total walk = 4096
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_cl["C_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["C"], offset=0, len=4096,
                           dimensions=[(64, 8), (8, 512), (8, 1)])
                    use_lock(_cl["C_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[6])
                # S2MM 0: A-in ping/pong (len 2048).
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
                # S2MM 1: B-in ping/pong (len 2048).
                with block[7]:
                    use_lock(_cl["B_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["B_ping"], offset=0, len=2048)
                    use_lock(_cl["B_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(_cl["B_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["B_pong"], offset=0, len=2048)
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
            link_with="bf16_gemm_pythoc_M8_N8_K4_AT_bf16out_s64_512_64_256_64_512.o",
        )
        gemm_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # ------------------------------------------------------------
        # aie.core body per compute tile.  Mirrors cached's nested-loop
        # structure: 8 outer (arg0) x 4 inner step=2 (arg1) ping/pong
        # pairs = 32 ping/pong pairs (= 64 kernel calls) per dispatch.
        # ------------------------------------------------------------
        def _make_compute_core(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
                # 6D identity permutation map for the zero-init transfer_write.
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
                    # Zero buf_C inline: nested 8x8 over 8x8 micro-tiles.
                    for m_i in range_(0, OG_MATMUL_C_M, 1):
                        for n_i in range_(0, OG_MATMUL_C_N, 1):
                            vector_dialect.transfer_write(
                                None, cst_zero, _bufs["C"],
                                [c0_idx, c0_idx, m_i, n_i, c0_idx, c0_idx],
                                permutation_map=zero_perm,
                                in_bounds=[True, True, True, True, True, True])
                    # Outer-M loop: 8 iters.
                    for _m_outer in range_(0, OG_MATMUL_M_OUTER, 1):
                        # Inner ping/pong: 4 step=2 pairs (cached uses
                        # `arg1 = 0..8 step 2`, with 2 kernel calls per
                        # arg1 value).
                        for _n_pair in range_(0, OG_MATMUL_N_INNER_HALF, 1):
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
        # Flows.  Same shape as v_matmul (116 total):
        #   8 shim->mem DMA 0       (W-weight in)
        #   4 shim->mem DMA 1       (X-input in, cols 0-3)
        #   8 mem->shim DMA 0       (C-out)
        #   32 mem->core DMA 1      (W-weight broadcast: mem_C -> 4 cores
        #                             in same col)
        #   32 mem->core DMA 2      (X-broadcast: mem_{0..3} -> 8 cores
        #                             in row 2..5 respectively)
        #   32 core->mem DMA 0      (C-out feedback to mem-tile;
        #                             cols 0-3 use mem DMA 2..5;
        #                             cols 4-7 use mem DMA 1..4)
        # ------------------------------------------------------------
        # 8 shim -> mem DMA 0 (W-weight in).
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
        # 32 mem -> core DMA 1 (W-weight broadcast).
        for col in range(N_COLS):
            for row in range(2, 6):
                flow(mem_tiles[col], WireBundle.DMA, 1,
                     compute_tiles[(col, row)], WireBundle.DMA, 0)
        # 32 mem -> core DMA 2 (X-broadcast).
        for row_offset in range(4):       # mem 0..3 -> rows 2..5
            for col in range(N_COLS):
                flow(mem_tiles[row_offset], WireBundle.DMA, 2,
                     compute_tiles[(col, 2 + row_offset)], WireBundle.DMA, 1)
        # 32 core -> mem DMA channels (C-out feedback).
        for col in range(4):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 2 + (row - 2))
        for col in range(4, N_COLS):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 1 + (row - 2))

        # ------------------------------------------------------------
        # aie.memtile_dma blocks (per col).  Cached emits col 0..7
        # ascending.
        # ------------------------------------------------------------
        def _make_memtile_dma_x_col(col):
            """Emit memtile_dma for cols 0-3 (with X-broadcast chain)."""
            ml = mem_locks[col]
            mt = mem_tiles[col]
            mb = mem_buf[col]
            @memtile_dma(mt)
            def _mt_dma(block):
                # MM2S 0: W-weight broadcast (self-loop, len 16384).
                # 3D dims: [<size=64, stride=64>, <size=4, stride=4096>,
                #           <size=64, stride=1>]
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=16384,
                           dimensions=[(64, 64), (4, 4096), (64, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong (len 16384 each).
                # 3D dims: [<size=32, stride=8>, <size=64, stride=256>,
                #           <size=8, stride=1>]
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ml["C_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=16384,
                           dimensions=[(32, 8), (64, 256), (8, 1)])
                    use_lock(ml["C_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ml["C_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=16384,
                           dimensions=[(32, 8), (64, 256), (8, 1)])
                    use_lock(ml["C_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[4])
                # MM2S 2: X-broadcast ping/pong (len 16384 each).
                # 4D dims: [<size=8, stride=2048>, <size=8, stride=8>,
                #           <size=32, stride=64>, <size=8, stride=1>]
                with block[6]:
                    dma_start(DMAChannelDir.MM2S, 2, dest=block[7], chain=block[9])
                with block[7]:
                    use_lock(ml["X_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_ping"], offset=0, len=16384,
                           dimensions=[(8, 2048), (8, 8), (32, 64), (8, 1)])
                    use_lock(ml["X_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(ml["X_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_pong"], offset=0, len=16384,
                           dimensions=[(8, 2048), (8, 8), (32, 64), (8, 1)])
                    use_lock(ml["X_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[7])
                # S2MM 0: C-in ping/pong (from compute, len 16384).
                with block[9]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[10], chain=block[12])
                with block[10]:
                    use_lock(ml["C_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=16384)
                    use_lock(ml["C_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[11])
                with block[11]:
                    use_lock(ml["C_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=16384)
                    use_lock(ml["C_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[10])
                # S2MM 1: X-in ping/pong (from shim, len 16384).
                with block[12]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[13], chain=block[15])
                with block[13]:
                    use_lock(ml["X_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_ping"], offset=0, len=16384)
                    use_lock(ml["X_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[14]:
                    use_lock(ml["X_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["X_pong"], offset=0, len=16384)
                    use_lock(ml["X_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[13])
                # S2MM 2..5: W-weight slice 0..3 (each 1-BD self-loop,
                # len 4096 -- og's W buf is 16384 elts total, so 4 x 4096
                # slices; v_matmul's was 32768 elts / 4 = 8192 per slice).
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 2, dest=block[16], chain=block[17])
                with block[16]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=0, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[16])
                with block[17]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[18], chain=block[19])
                with block[18]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=4096, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[18])
                with block[19]:
                    dma_start(DMAChannelDir.S2MM, 4, dest=block[20], chain=block[21])
                with block[20]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=8192, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[20])
                with block[21]:
                    dma_start(DMAChannelDir.S2MM, 5, dest=block[22], chain=block[2])
                with block[22]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=12288, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[22])

        def _make_memtile_dma_no_x_col(col):
            """Emit memtile_dma for cols 4-7 (no X-broadcast chain)."""
            ml = mem_locks[col]
            mt = mem_tiles[col]
            mb = mem_buf[col]
            @memtile_dma(mt)
            def _mt_dma(block):
                # MM2S 0: W-weight broadcast (self-loop, len 16384).
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=16384,
                           dimensions=[(64, 64), (4, 4096), (64, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong.
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ml["C_ping_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=16384,
                           dimensions=[(32, 8), (64, 256), (8, 1)])
                    use_lock(ml["C_ping_sem"], LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ml["C_pong_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=16384,
                           dimensions=[(32, 8), (64, 256), (8, 1)])
                    use_lock(ml["C_pong_sem"], LockAction.Release, value=1)
                    next_bd(block[4])
                # S2MM 0: C-in ping/pong (from compute).
                with block[6]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[9])
                with block[7]:
                    use_lock(ml["C_ping_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_ping"], offset=0, len=16384)
                    use_lock(ml["C_ping_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
                with block[8]:
                    use_lock(ml["C_pong_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["C_pong"], offset=0, len=16384)
                    use_lock(ml["C_pong_ready"], LockAction.Release, value=1)
                    next_bd(block[7])
                # S2MM 1..4: W-weight slice 0..3.
                with block[9]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[10], chain=block[11])
                with block[10]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=0, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[10])
                with block[11]:
                    dma_start(DMAChannelDir.S2MM, 2, dest=block[12], chain=block[13])
                with block[12]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=4096, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[12])
                with block[13]:
                    dma_start(DMAChannelDir.S2MM, 3, dest=block[14], chain=block[15])
                with block[14]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=8192, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[14])
                with block[15]:
                    dma_start(DMAChannelDir.S2MM, 4, dest=block[16], chain=block[2])
                with block[16]:
                    use_lock(ml["W_sem"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mb["W"], offset=12288, len=4096)
                    use_lock(ml["W_ready"], LockAction.Release, value=1)
                    next_bd(block[16])

        for col in range(N_COLS):
            if col < 4:
                _make_memtile_dma_x_col(col)
            else:
                _make_memtile_dma_no_x_col(col)

        # ------------------------------------------------------------
        # Shim allocations.  Cached order (lines 34316-34339):
        #   8x air_channel_84  S2MM 0 (Y output)
        #   8x air_channel_78  MM2S 0 (X input)
        #   4x air_channel_77  MM2S 1 (W weight, cols 0-3)
        # ------------------------------------------------------------
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_OG_Y}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_OG_X}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(4):
            shim_dma_allocation(
                f"air_channel_{_CHAN_OG_WEIGHT}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # ------------------------------------------------------------
        # Runtime sequence.  og uses arg0=X, arg1=W, arg2=Y.  Output
        # is 2048x2048 -- 4x bigger than v/k matmul, so each shim
        # channel fans out into 4 dispatches per col (X: 32 dispatches
        # total; W: 16 dispatches total).  Y output uses 8 single
        # dispatches with repeat_count=3 (4 inner iters).
        #
        # Cached free order (verified at lines 35233-35288): 8 awaits
        # (reverse, col 7 -> 0), then 48 frees in reverse-col groups
        # of 4 (W cols 3 -> 0, then X cols 7 -> 0).
        # ------------------------------------------------------------
        @runtime_sequence(*_o_ffn_host_arg_types(),
                          sym_name="og_matmul_seg_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12, arg13, arg14):
            x_buf = arg0           # 2048x2048 X input  (attn-output residual)
            w_buf = arg1           # 2048x2048 W weight (O-projection Wo)
            y_buf = arg2           # 2048x2048 Y output

            x_tasks = []
            w_tasks = []
            y_tasks = []

            # 32 X-input tasks (MM2S 0, 4 per col).  BD dims (3D):
            #   [<size=8, stride=256>, <size=64, stride=2048>,
            #    <size=256, stride=1>]
            # Per col c, dispatch d (d in 0..3) offset:
            #   c * 131072 + d * 1048576
            for col in range(N_COLS):
                for d in range(4):
                    offset = col * 131072 + d * 1048576
                    t = dma_configure_task_for(
                        f"air_channel_{_CHAN_OG_X}_{col}",
                        repeat_count=7)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(x_buf, offset=offset, len=131072,
                                   dimensions=[(8, 256),
                                               (64, 2048),
                                               (256, 1)])
                            EndOp()
                    dma_start_task(t)
                    x_tasks.append(t)

            # 16 W-weight tasks (MM2S 1, 4 per col on cols 0-3).
            # BD dims (4D):
            #   [<size=8, stride=256>, <size=8, stride=524288>,
            #    <size=256, stride=2048>, <size=64, stride=1>]
            # Per col c, all 4 dispatches use the same offset
            #   c * 64 (W is row-major 2048x2048, col c gets 64-wide
            #   horizontal stripe at column-byte-offset col*64*2 = c*128
            #   ... actually arg1 is bf16 so offset is in elements:
            #   col*64 elements = col*128 bytes; cached MLIR shows
            #   offset=0, 64, 128, 192 for cols 0,1,2,3).
            for col in range(4):
                for d in range(4):
                    offset = col * 64
                    t = dma_configure_task_for(
                        f"air_channel_{_CHAN_OG_WEIGHT}_{col}",
                        repeat_count=7)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(w_buf, offset=offset, len=131072,
                                   dimensions=[(8, 256),
                                               (8, 524288),
                                               (256, 2048),
                                               (64, 1)])
                            EndOp()
                    dma_start_task(t)
                    w_tasks.append(t)

            # 8 Y-output tasks (S2MM 0, 1 per col, issue_token=true,
            # repeat_count=3).  BD dims (4D):
            #   [<size=4, stride=1048576>, <size=8, stride=256>,
            #    <size=64, stride=2048>, <size=256, stride=1>]
            for col in range(N_COLS):
                offset = col * 131072
                t = dma_configure_task_for(
                    f"air_channel_{_CHAN_OG_Y}_{col}",
                    issue_token=True, repeat_count=3)
                with bds(t) as bd:
                    with bd[0]:
                        dma_bd(y_buf, offset=offset, len=131072,
                               dimensions=[(4, 1048576),
                                           (8, 256),
                                           (64, 2048),
                                           (256, 1)])
                        EndOp()
                dma_start_task(t)
                y_tasks.append(t)

            # Cached free/await ordering (lines 35233-35288):
            #   1. 8 awaits in REVERSE col order (y col 7 -> 0)
            #   2. 16 W frees in REVERSE col order (cols 3 -> 0, each
            #      col's 4 dispatches in ASCENDING dispatch order)
            #   3. 32 X frees in REVERSE col order (cols 7 -> 0, each
            #      col's 4 dispatches in ASCENDING dispatch order)
            for t in reversed(y_tasks):
                dma_await_task(t)
            for col in reversed(range(4)):
                for d in range(4):
                    dma_free_task(w_tasks[col * 4 + d])
            for col in reversed(range(N_COLS)):
                for d in range(4):
                    dma_free_task(x_tasks[col * 4 + d])


# ---------------------------------------------------------------------------
# gg_matmul_seg device (Phase 4.6e -- FFN gate-projection GEMM of o_ffn).
# ---------------------------------------------------------------------------
# Structurally mirrors the rms_gemms_rope ``@v_matmul_seg`` device (8x4
# herd, 8 mem tiles with asymmetric lock counts: cols 0-3 = 10 locks,
# cols 4-7 = 6 locks).  The per-core body, per-core aie.mem block,
# per-mem-tile memtile_dma block, and L2 buffer shapes are
# byte-identical to v_matmul.  gg differs from v_matmul only above the
# compute layer:
#   * shim channel ids:    86 (X), 85 (W), 82 (Y)         vs v's 60/65/61
#   * host arg indices:    arg6 (X), arg7 (W), arg8 (Y)   vs v's arg2/arg7/arg8
#   * external_buffer shapes: 2048x2048, 2048x8192, 2048x8192
#     (X is 2048x2048, W/Y are 2048x8192)               vs v's 2048x2048 + 2x2048x512
#   * runtime sequence dispatch fan-out: 4 dispatches per col on every
#     channel (32 X-tasks, 16 W-tasks, 32 Y-tasks)       vs v's 8/4/8
#     because gg's output is 2048x8192 (16x larger than v's 2048x512;
#     4x larger than Q's 2048x2048), and gg's BD len is half (X uses
#     len=131072 like v/Q, but each col emits 4 dispatches at base
#     offsets col*131072 + d*1048576).
#
# Cached gg runtime sequence (verified at lines 24868-25349):
#   X (arg6, channel 86, MM2S 0):
#     32 tasks (4 per col), each `len=131072`, 3D dims
#       [<size=32, stride=64>, <size=64, stride=2048>, <size=64, stride=1>]
#     offset = col*131072 + d*1048576;  repeat_count=15  (16 inner iters
#     per dispatch)
#   W (arg7, channel 85, MM2S 1, cols 0-3 only):
#     16 tasks (4 per col), each `len=262144`, 4D dims
#       [<size=16, stride=512>, <size=32, stride=524288>,
#        <size=64, stride=8192>, <size=128, stride=1>]
#     offset = col*128 (constant per col, 4 dispatches re-issue same bd-base);
#     repeat_count=15
#   Y (arg8, channel 82, S2MM 0):
#     32 tasks (4 per col), each `len=524288`, 3D dims
#       [<size=16, stride=512>, <size=64, stride=8192>, <size=512, stride=1>]
#     offset = col*524288 + d*4194304;  issue_token=true  (no repeat_count)
#
# Free/await order (cached):
#   1. 32 dma_await on Y in col-major reverse (col 7 first, dispatches 0..3
#      ascending; ... down to col 0)
#   2. 16 dma_free on W in col-major reverse (col 3 first, dispatches 0..3
#      ascending; ... down to col 0)
#   3. 32 dma_free on X in col-major reverse (col 7 first, dispatches 0..3
#      ascending; ... down to col 0)
#
# Kernel reuse note: gg's per-core body and L1 buffer shapes
# (C=1x1x16x8x8x8, A=1x1x4x8x8x8, B=1x1x16x4x8x8) are identical to
# v_matmul's, so gg links against the existing v_matmul kernel
# ``bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o`` --
# no new compile is needed.
# ---------------------------------------------------------------------------
GG_MATMUL_K_OUTER = 32             # K-outer loop bound per core dispatch
GG_MATMUL_C_M = 16                 # buf_C outer-M dim (16 m-blocks of 8x8)
GG_MATMUL_C_N = 8                  # buf_C outer-N dim (8 n-blocks of 8x8)

# Channel ids for gg_matmul_seg in the cached IR.
_CHAN_GG_X      = 86               # MM2S 0 (X input)   on shim_C_0
_CHAN_GG_WEIGHT = 85               # MM2S 1 (W weight)  on shim_C_0
_CHAN_GG_Y      = 82               # S2MM 0 (Y output)  on shim_C_0

# Host-arg indices for the @gg_matmul_seg_sequence
# (verified at cached o_ffn.npu.air.mlir:24868).
_GG_ARG_X = 6                      # 2048x2048 bf16  normed FFN input
_GG_ARG_W = 7                      # 2048x8192 bf16  Wgate
_GG_ARG_Y = 8                      # 2048x8192 bf16  gate output


def _emit_gg_matmul_seg() -> None:
    """Emit the placed-IRON ``@gg_matmul_seg`` device (FFN gate-proj GEMM).

    Must be called inside an active ``mlir_mod_ctx()``; registers one
    ``aie.device(npu2) @gg_matmul_seg`` op.  Body-level divergence from
    the cached IR: each K-outer iteration calls ``bf16_gemm_kernel_bf16out``
    once on the L1 (A, B, C) buffer triple instead of inlining the
    4-MAC vector.contract chain.  Infrastructure (tiles, locks, buffers,
    flows, memtile_dma BD chains, shim allocs, runtime_sequence) matches
    the cached AIR-stitched IR op-for-op (modulo SSA naming).

    Host args (verified at cached runtime_sequence @gg_matmul_seg_sequence,
    o_ffn.npu.air.mlir:24868):
      arg6 : 2048x2048 bf16   X input  (normed_x from rm_weighted_rms_norm_seg)
      arg7 : 2048x8192 bf16   W weight (Wgate)
      arg8 : 2048x8192 bf16   Y output (gate projection result)
    """
    from aie.dialects import memref as memref_dialect  # noqa: F401
    from aie.dialects import vector as vector_dialect
    from aie.extras import types as T
    from aie.ir import UnitAttr

    @device(AIEDevice.npu2, sym_name="gg_matmul_seg")
    def _dev():
        # 8 shim + 8 mem + 32 compute tiles (cols 0..7, rows 2..5).
        shim_tiles    = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles     = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = {}
        for col in range(N_COLS):
            for row in range(2, 6):
                compute_tiles[(col, row)] = tile(col, row)

        # ------------------------------------------------------------
        # Locks.  Cached emit order: mem tiles col 7 first (6 locks),
        # then 6, 5, 4 (6 each), then 3, 2, 1, 0 (10 each).  Then
        # compute-tile locks row-major ascending (row 2 col 0..7,
        # row 3 col 0..7, ..., row 5 col 0..7).
        # ------------------------------------------------------------
        mem_locks = {}
        for col in reversed(range(4, N_COLS)):    # 7, 6, 5, 4 (6 locks each)
            mt = mem_tiles[col]
            mem_locks[col] = {
                "C_pong_sem":   lock(mt, lock_id=5, init=1),
                "C_pong_ready": lock(mt, lock_id=4, init=0),
                "C_ping_sem":   lock(mt, lock_id=3, init=1),
                "C_ping_ready": lock(mt, lock_id=2, init=0),
                "W_sem":        lock(mt, lock_id=1, init=4),
                "W_ready":      lock(mt, lock_id=0, init=0),
            }
        for col in reversed(range(4)):            # 3, 2, 1, 0 (10 locks each)
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

        # Compute-tile locks: row-major ascending.
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
        # Buffers.  Mem-tile buffers first (col 0..7 ascending per
        # category), then compute-tile buffers (descending row, then
        # descending col).
        #
        # Cached mem-tile emit order:
        #   buf487..buf480 : 1x4x64x128  (W-weight, col 0..7)
        #   buf479..buf464 : 1x1x64x64   (C-out ping/pong, 2 per col,
        #                                 cols 0..7 ascending)
        #   buf463..buf456 : 1x1x64x128  (X-input ping/pong, 2 per col,
        #                                 cols 0..3 ascending)
        # ------------------------------------------------------------
        BF16_W_L2  = bf16_memref(1, 4, 64, 128, memory_space=1)
        BF16_CO_L2 = bf16_memref(1, 1, 64,  64, memory_space=1)
        BF16_XI_L2 = bf16_memref(1, 1, 64, 128, memory_space=1)

        mem_buf = {col: {} for col in range(N_COLS)}
        # W-weight: 1 per col, col order 0..7.
        for col in range(N_COLS):
            mem_buf[col]["W"] = buffer(mem_tiles[col], datatype=BF16_W_L2)
        # C-out ping/pong: 2 per col, col 0 (ping, pong), col 1, ...
        for col in range(N_COLS):
            mem_buf[col]["C_ping"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
            mem_buf[col]["C_pong"] = buffer(mem_tiles[col], datatype=BF16_CO_L2)
        # X-input ping/pong: 2 each on cols 0..3 only.
        for col in range(4):
            mem_buf[col]["X_ping"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)
            mem_buf[col]["X_pong"] = buffer(mem_tiles[col], datatype=BF16_XI_L2)

        # Compute-tile buffers.  Cached emit order: row 5 col 7 first,
        # then row 5 col 6, ..., down to row 2 col 0.  Per tile (5
        # buffers, descending sym): C, A_pong, B_pong, A_ping, B_ping.
        # (This is the cached gg per-tile order verified at lines
        # 17801-17805; the actual ping/pong roles are derived from how
        # the aie.mem block's bb4/bb5/bb7/bb8 read them.)
        BF16_C_L1 = bf16_memref(1, 1, 16, 8, 8, 8, memory_space=2)
        BF16_A_L1 = bf16_memref(1, 1, 4,  8, 8, 8, memory_space=2)
        BF16_B_L1 = bf16_memref(1, 1, 16, 4, 8, 8, memory_space=2)

        core_buf = {}
        for row in reversed(range(2, 6)):
            for col in reversed(range(N_COLS)):
                ct = compute_tiles[(col, row)]
                bufs = {}
                bufs["C"]      = buffer(ct, datatype=BF16_C_L1)
                bufs["A_pong"] = buffer(ct, datatype=BF16_A_L1)
                bufs["B_pong"] = buffer(ct, datatype=BF16_B_L1)
                bufs["A_ping"] = buffer(ct, datatype=BF16_A_L1)
                bufs["B_ping"] = buffer(ct, datatype=BF16_B_L1)
                core_buf[(col, row)] = bufs

        # External buffers (opaque AIR metadata; kept for diff parity).
        # gg uses (2048x2048, 2048x8192, 2048x8192) -- the X/W/Y triple.
        external_buffer(bf16_np(EMB_DIM, EMB_DIM),       name="__air_external_buffer")
        external_buffer(bf16_np(EMB_DIM, HIDDEN_DIM),    name="__air_external_buffer_1")
        external_buffer(bf16_np(EMB_DIM, HIDDEN_DIM),    name="__air_external_buffer_2")

        # ------------------------------------------------------------
        # aie.mem blocks per compute tile.  Cached order: row 5 col 7
        # first, ..., row 2 col 0.  Layout identical to v_matmul:
        #   MM2S 0 (C out): buf_C self-loop, len 8192, 3D dims
        #   S2MM 0 (A in):  A_ping <-> A_pong ping-pong, len 2048
        #   S2MM 1 (B in):  B_ping <-> B_pong ping-pong, len 4096
        # ------------------------------------------------------------
        def _make_compute_mem(_ct, _cl, _bufs):
            @mem(_ct)
            def _core_mem(block):
                # MM2S 0: C-out self-loop, len 8192 elts.
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_cl["C_full"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_bufs["C"], offset=0, len=8192,
                           dimensions=[(64, 8), (16, 512), (8, 1)])
                    use_lock(_cl["C_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[6])
                # S2MM 0: A-in ping/pong (len 2048).
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
                # S2MM 1: B-in ping/pong (len 4096).
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
        # External function declaration.  link_with points at the
        # existing v_matmul PythoC kernel -- gg's per-core body is
        # byte-identical to v's, so the same .o serves both devices.
        # ------------------------------------------------------------
        gemm_fn = external_func(
            "bf16_gemm_kernel_bf16out",
            inputs=[BF16_A_L1, BF16_B_L1, BF16_C_L1],
            link_with="bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o",
        )
        gemm_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # ------------------------------------------------------------
        # aie.core body per compute tile.  Mirrors v_matmul's body:
        #   acquire C_done x1
        #   zero buf_C (16x8 nested over 8x8 micro-tiles)
        #   for k_outer in 0..32:
        #     acquire A_ready, B_ready; gemm(A_ping, B_ping, C); release B_sem, A_sem
        #     acquire A_ready, B_ready; gemm(A_pong, B_pong, C); release B_sem, A_sem
        #   release C_full x1
        #   cf.br ^bb1
        # ------------------------------------------------------------
        def _make_compute_core(_ct, _cl, _bufs):
            @core(_ct)
            def _core_body():
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
                    for m_i in range_(0, GG_MATMUL_C_M, 1):
                        for n_i in range_(0, GG_MATMUL_C_N, 1):
                            vector_dialect.transfer_write(
                                None, cst_zero, _bufs["C"],
                                [c0_idx, c0_idx, m_i, n_i, c0_idx, c0_idx],
                                permutation_map=zero_perm,
                                in_bounds=[True, True, True, True, True, True])
                    # K-outer loop: 32 iters.
                    for _k_outer in range_(0, GG_MATMUL_K_OUTER, 1):
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
        # Flows (116 total, same shape as v_matmul).
        # ------------------------------------------------------------
        # 8 shim -> mem DMA 0 (W-weight in).
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
        # 32 mem -> core DMA 1 (W-weight broadcast: mem_C -> 4 cores in same col).
        for col in range(N_COLS):
            for row in range(2, 6):
                flow(mem_tiles[col], WireBundle.DMA, 1,
                     compute_tiles[(col, row)], WireBundle.DMA, 0)
        # 32 mem -> core DMA 2 (X-broadcast: mem_{R-2} -> all 8 cores in row R).
        for row_offset in range(4):       # mem 0..3 -> rows 2..5
            for col in range(N_COLS):
                flow(mem_tiles[row_offset], WireBundle.DMA, 2,
                     compute_tiles[(col, 2 + row_offset)], WireBundle.DMA, 1)
        # 32 core -> mem DMA channels (C-out feedback to mem-tile;
        # cols 0-3 use mem DMA 2..5, cols 4-7 use mem DMA 1..4 because
        # cols 0-3 reserve DMA 1 for X-broadcast).
        for col in range(4):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 2 + (row - 2))
        for col in range(4, N_COLS):
            for row in range(2, 6):
                flow(compute_tiles[(col, row)], WireBundle.DMA, 0,
                     mem_tiles[col], WireBundle.DMA, 1 + (row - 2))

        # ------------------------------------------------------------
        # aie.memtile_dma blocks (per col, byte-identical to v_matmul).
        # ------------------------------------------------------------
        def _make_memtile_dma_x_col(col):
            """Emit memtile_dma for cols 0-3 (with X-broadcast chain)."""
            ml = mem_locks[col]
            mt = mem_tiles[col]
            mb = mem_buf[col]
            @memtile_dma(mt)
            def _mt_dma(block):
                # MM2S 0: W-weight broadcast (self-loop, len 32768).
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=32768,
                           dimensions=[(64, 128), (4, 8192), (128, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong (len 4096 each).
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
                # MM2S 2: X-broadcast ping/pong (len 8192 each).
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
                # S2MM 0: C-in ping/pong (from compute, len 4096).
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
                # S2MM 1: X-in ping/pong (from shim, len 8192).
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
                # S2MM 2..5: W-weight slice 0..3 (each 1-BD self-loop,
                # len 8192 -- 4 slices of the 32768-elt W buf).
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
                # MM2S 0: W-weight broadcast (self-loop, len 32768).
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ml["W_ready"], LockAction.AcquireGreaterEqual, value=4)
                    dma_bd(mb["W"], offset=0, len=32768,
                           dimensions=[(64, 128), (4, 8192), (128, 1)])
                    use_lock(ml["W_sem"], LockAction.Release, value=4)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                # MM2S 1: C-out ping/pong (len 4096 each).
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
                # S2MM 1..4: W-weight slice 0..3 (each 1-BD self-loop, len 8192).
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

        for col in range(N_COLS):
            if col < 4:
                _make_memtile_dma_x_col(col)
            else:
                _make_memtile_dma_no_x_col(col)

        # ------------------------------------------------------------
        # Shim allocations.  Cached order (lines 24848-24867):
        #   8x air_channel_82  S2MM 0 (Y output)
        #   8x air_channel_86  MM2S 0 (X input)
        #   4x air_channel_85  MM2S 1 (W weight, cols 0-3)
        # ------------------------------------------------------------
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_GG_Y}_{col}", shim_tiles[col],
                DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(
                f"air_channel_{_CHAN_GG_X}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 0)
        for col in range(4):
            shim_dma_allocation(
                f"air_channel_{_CHAN_GG_WEIGHT}_{col}", shim_tiles[col],
                DMAChannelDir.MM2S, 1)

        # ------------------------------------------------------------
        # Runtime sequence.  gg uses arg6=X, arg7=W, arg8=Y.  Output is
        # 2048x8192 (4x bigger than og's 2048x2048; 16x bigger than v's
        # 2048x512).  Each shim channel fans out into 4 dispatches per
        # col on all three streams.  See the device-level comment block
        # above for the bd-base offsets and dim layouts.
        #
        # Cached free/await order (lines 25269-25348):
        #   1. 32 awaits on Y in col-major REVERSE (col 7 first, dispatches
        #      0..3 ascending; ... down to col 0)
        #   2. 16 frees on W in col-major REVERSE (cols 3 -> 0, each
        #      col's 4 dispatches in ascending order)
        #   3. 32 frees on X in col-major REVERSE (cols 7 -> 0, each
        #      col's 4 dispatches in ascending order)
        # ------------------------------------------------------------
        @runtime_sequence(*_o_ffn_host_arg_types(),
                          sym_name="gg_matmul_seg_sequence")
        def _seq(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                 arg9, arg10, arg11, arg12, arg13, arg14):
            host_args = (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                         arg8, arg9, arg10, arg11, arg12, arg13, arg14)
            x_buf = host_args[_GG_ARG_X]    # 2048x2048 normed X
            w_buf = host_args[_GG_ARG_W]    # 2048x8192 Wgate
            y_buf = host_args[_GG_ARG_Y]    # 2048x8192 gate output

            x_tasks = []
            w_tasks = []
            y_tasks = []

            # 32 X-input tasks (MM2S 0, 4 per col, repeat_count=15).  BD
            # dims (3D): [<size=32, stride=64>, <size=64, stride=2048>,
            # <size=64, stride=1>].  Per col c, dispatch d:
            #   offset = c * 131072 + d * 1048576.
            for col in range(N_COLS):
                for d in range(4):
                    offset = col * 131072 + d * 1048576
                    t = dma_configure_task_for(
                        f"air_channel_{_CHAN_GG_X}_{col}",
                        repeat_count=15)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(x_buf, offset=offset, len=131072,
                                   dimensions=[(32, 64),
                                               (64, 2048),
                                               (64, 1)])
                            EndOp()
                    dma_start_task(t)
                    x_tasks.append(t)

            # 16 W-weight tasks (MM2S 1, 4 per col on cols 0-3,
            # repeat_count=15).  BD dims (4D):
            #   [<size=16, stride=512>, <size=32, stride=524288>,
            #    <size=64, stride=8192>, <size=128, stride=1>]
            # Per col c, all 4 dispatches use offset = c * 128 (cached
            # MLIR re-issues the same bd-base 4 times per col).
            for col in range(4):
                for d in range(4):
                    offset = col * 128
                    t = dma_configure_task_for(
                        f"air_channel_{_CHAN_GG_WEIGHT}_{col}",
                        repeat_count=15)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(w_buf, offset=offset, len=262144,
                                   dimensions=[(16, 512),
                                               (32, 524288),
                                               (64, 8192),
                                               (128, 1)])
                            EndOp()
                    dma_start_task(t)
                    w_tasks.append(t)

            # 32 Y-output tasks (S2MM 0, 4 per col, issue_token=true).
            # BD dims (3D):
            #   [<size=16, stride=512>, <size=64, stride=8192>,
            #    <size=512, stride=1>]
            # Per col c, dispatch d:
            #   offset = c * 524288 + d * 4194304.
            for col in range(N_COLS):
                for d in range(4):
                    offset = col * 524288 + d * 4194304
                    t = dma_configure_task_for(
                        f"air_channel_{_CHAN_GG_Y}_{col}",
                        issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(y_buf, offset=offset, len=524288,
                                   dimensions=[(16, 512),
                                               (64, 8192),
                                               (512, 1)])
                            EndOp()
                    dma_start_task(t)
                    y_tasks.append(t)

            # Cached free/await ordering (lines 25269-25348).
            #   1. 32 awaits on Y in col-major REVERSE.
            for col in reversed(range(N_COLS)):
                for d in range(4):
                    dma_await_task(y_tasks[col * 4 + d])
            #   2. 16 W frees in col-major REVERSE (cols 3 -> 0).
            for col in reversed(range(4)):
                for d in range(4):
                    dma_free_task(w_tasks[col * 4 + d])
            #   3. 32 X frees in col-major REVERSE (cols 7 -> 0).
            for col in reversed(range(N_COLS)):
                for d in range(4):
                    dma_free_task(x_tasks[col * 4 + d])


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
#
# Outer wrapper that fires the 8 inner devices in the cached file's
# topological order:
#   og_matmul_seg ->
#   ra_add_seg ->
#   rm_weighted_rms_norm_seg ->
#   gg_matmul_seg ->
#   ug_matmul_seg ->
#   sw_silu_mul_seg ->
#   dg_matmul_seg ->
#   fa_add_seg
# All 8 segments share the same 15-arg host signature
# (see ``_o_ffn_host_arg_types``).  Verified against cached
# reference_mlir/o_ffn.npu.air.mlir lines 35291-35318.
# ---------------------------------------------------------------------------
_DISPATCHER_ORDER = (
    "og_matmul_seg",
    "ra_add_seg",
    "rm_weighted_rms_norm_seg",
    "gg_matmul_seg",
    "ug_matmul_seg",
    "sw_silu_mul_seg",
    "dg_matmul_seg",
    "fa_add_seg",
)


def _emit_dispatcher_device() -> None:
    """Emit the outer unnamed ``aie.device(npu2) { ... }`` dispatcher.

    Carries an ``aiex.runtime_sequence @o_ffn`` whose body fires
    each of the 8 inner segment sequences via ``aiex.configure`` +
    ``aiex.run``.  Each inner sequence receives the full 15-arg list
    (matches the cached IR; the inner devices only use the subset
    they need).
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            *_o_ffn_host_arg_types(),
            sym_name="o_ffn",
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
# Public entry point.
# ---------------------------------------------------------------------------
def build_o_ffn_module(seq_len: int = SEQ_LEN,
                       emb_dim: int = EMB_DIM,
                       hidden_dim: int = HIDDEN_DIM,
                       *,
                       verbose: bool = False,
                       omit_while_true_loop: bool = False) -> str:
    """Build the prefill o_ffn MLIR module.

    Phase 4.6e (incremental): 7 of 9 devices are emitted via placed-IRON --
    ``rm_weighted_rms_norm_seg``, ``ra_add_seg``, ``fa_add_seg``,
    ``sw_silu_mul_seg``, ``og_matmul_seg``, ``gg_matmul_seg``, plus the
    outer unnamed dispatcher device.  The 2 remaining GEMM devices
    (``ug_matmul_seg``, ``dg_matmul_seg``) come from the cached MLIR
    text via string splice; phases 4.6f-g are deferred.

    All dimensions must match the Llama-3.2-1B values; the cached AIR
    layout is shape-specialized.
    """
    if (seq_len, emb_dim, hidden_dim) != (SEQ_LEN, EMB_DIM, HIDDEN_DIM):
        raise ValueError(
            "o_ffn builder is currently fixed to Llama-3.2-1B prefill "
            f"dimensions; got seq_len={seq_len}, emb_dim={emb_dim}, "
            f"hidden_dim={hidden_dim}."
        )
    del omit_while_true_loop  # unused (no while-true loop in this device)

    # Build a fresh module containing the placed devices.
    with mlir_mod_ctx() as ctx:
        _emit_rm_weighted_rms_norm_seg()
        _emit_ra_add_seg()
        _emit_fa_add_seg()
        _emit_sw_silu_mul_seg()
        _emit_og_matmul_seg()
        _emit_gg_matmul_seg()
        _emit_dispatcher_device()
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    # Use ``assume_verified=True`` here -- the dispatcher's
    # ``aiex.configure`` ops reference the 2 cached GEMM device syms
    # (``ug_matmul_seg``, ``dg_matmul_seg``) which are NOT present in
    # this freshly-emitted module (they live only in the cached MLIR
    # text and are stitched back in via ``_splice_device`` below).
    # Without ``assume_verified=True`` the verifier flags "No such
    # device: '@ug_matmul_seg'" and the printer falls back to the
    # generic op form, which breaks the brace-counting ``_extract_*``
    # helpers.
    placed_text = module.operation.get_asm(assume_verified=True)
    placed_rms = _extract_single_device(placed_text, "rm_weighted_rms_norm_seg")
    placed_ra  = _extract_single_device(placed_text, "ra_add_seg")
    placed_fa  = _extract_single_device(placed_text, "fa_add_seg")
    placed_sw  = _extract_single_device(placed_text, "sw_silu_mul_seg")
    placed_og  = _extract_single_device(placed_text, "og_matmul_seg")
    placed_gg  = _extract_single_device(placed_text, "gg_matmul_seg")
    placed_dispatcher = _extract_dispatcher_device(placed_text)

    # Load the cached prefill MLIR and splice in the placed devices.
    project_root = Path(__file__).resolve().parents[1]
    cached_path = project_root / "reference_mlir" / "o_ffn.npu.air.mlir"
    cached_text = cached_path.read_text()
    original_len = len(cached_text)

    spliced = _splice_device(cached_text, "rm_weighted_rms_norm_seg", placed_rms)
    spliced = _splice_device(spliced, "ra_add_seg", placed_ra)
    spliced = _splice_device(spliced, "fa_add_seg", placed_fa)
    spliced = _splice_device(spliced, "sw_silu_mul_seg", placed_sw)
    spliced = _splice_device(spliced, "og_matmul_seg", placed_og)
    spliced = _splice_device(spliced, "gg_matmul_seg", placed_gg)
    spliced = _splice_dispatcher_device(spliced, placed_dispatcher)

    if verbose:
        print(f"  [o_ffn builder] Spliced placed-IRON rm_weighted_rms_norm_seg "
              f"+ ra_add_seg + fa_add_seg + sw_silu_mul_seg + og_matmul_seg "
              f"+ gg_matmul_seg + dispatcher into cached MLIR ({original_len} "
              f"-> {len(spliced)} bytes).")

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
                        help="Emit just the 7 placed devices "
                             "(rm_weighted_rms_norm_seg, ra_add_seg, "
                             "fa_add_seg, sw_silu_mul_seg, og_matmul_seg, "
                             "gg_matmul_seg, dispatcher) -- skips cached "
                             "splice")
    args = parser.parse_args()
    if args.device_only:
        with mlir_mod_ctx() as ctx:
            _emit_rm_weighted_rms_norm_seg()
            _emit_ra_add_seg()
            _emit_fa_add_seg()
            _emit_sw_silu_mul_seg()
            _emit_og_matmul_seg()
            _emit_gg_matmul_seg()
            _emit_dispatcher_device()
            mod = ctx.module
        # See note in build_o_ffn_module() about assume_verified=True:
        # the dispatcher references 2 GEMM devs not present in this
        # standalone module, so the verifier fails -- skip it.
        text = mod.operation.get_asm(assume_verified=True)
    else:
        text = build_o_ffn_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
