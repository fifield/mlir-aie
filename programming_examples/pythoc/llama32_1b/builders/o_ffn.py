# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b prefill o_ffn kernel.

Phase 4.6a scope: only the ``@rm_weighted_rms_norm_seg`` device is emitted
from placed IRON.  The remaining 8 devices (``og_matmul_seg``,
``ra_add_seg``, ``gg_matmul_seg``, ``ug_matmul_seg``, ``sw_silu_mul_seg``,
``dg_matmul_seg``, ``fa_add_seg``, plus the outer unnamed dispatcher) come
from the cached ``o_ffn.npu.air.mlir`` text via string splice.

Splice mechanism::

    cached_text =
      aie.device @fa_add_seg { ... }                  <-- cached
      aie.device @dg_matmul_seg { ... }               <-- cached
      aie.device @sw_silu_mul_seg { ... }             <-- cached
      aie.device @ug_matmul_seg { ... }               <-- cached
      aie.device @gg_matmul_seg { ... }               <-- cached
      aie.device @rm_weighted_rms_norm_seg { ... }    <-- REPLACED
      aie.device @ra_add_seg { ... }                  <-- cached
      aie.device @og_matmul_seg { ... }               <-- cached
      aie.device (dispatcher) { ... }                 <-- cached

The cached ``rm_weighted_rms_norm_seg`` (lines 25351-26580 of
``reference_mlir/o_ffn.npu.air.mlir``, 1230 lines) is structurally
identical to the rms_gemms_rope ``@r_weighted_rms_norm_seg`` already
landed by Phase 4.5a -- same 1x8 herd of compute tiles, same 6-lock
per-tile pattern (init 2,0,1,0,2,0), same 7-buffer L1 layout, same
3-DMA-channel mem block, same inline RMSNorm math (``arith.addf``
sum-of-squares -> ``math.rsqrt`` -> ``arith.mulf`` rescale).  It differs
only in:

  * device sym name (``rm_weighted_rms_norm_seg`` vs ``r_weighted_rms_norm_seg``)
  * shim channel ids (gamma=19, x_in=20_C, y_out=21_C vs 0, 1_C, 2_C)
  * host arg signature: 15 args here, gamma=arg5, x_in=arg4, y_out=arg6
    (vs 13 args, gamma=arg1, x_in/y_out=arg0/arg2 in rms_gemms_rope)

The runtime_sequence DMA offsets and dimensions are identical to the
rms_gemms_rope's r_weighted_rms_norm_seg: each col gets 524288 bf16
elements at offset col*524288, dims [(2, 262144), (512, 512), (512, 1)],
plus a single gamma-broadcast task on shim_0_0 MM2S 0 of len 2048 with
dims [(4, 512), (512, 1)].
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
from aie.ir import AffineDimExpr, AffineMap

from ._emit import bf16_memref, bf16_np

# Reuse the splice helper from rms_gemms_rope -- it's device-agnostic
# (brace-counting parser keyed on the device sym name).
from .rms_gemms_rope import _splice_device, _extract_single_device


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
# Public entry point.
# ---------------------------------------------------------------------------
def build_o_ffn_module(seq_len: int = SEQ_LEN,
                       emb_dim: int = EMB_DIM,
                       hidden_dim: int = HIDDEN_DIM,
                       *,
                       verbose: bool = False,
                       omit_while_true_loop: bool = False) -> str:
    """Build the prefill o_ffn MLIR module.

    Phase 4.6a: only the ``rm_weighted_rms_norm_seg`` device is emitted
    via placed-IRON; the other 8 devices come from the cached MLIR text
    via string splice.

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

    # Build a fresh module containing the placed device.
    with mlir_mod_ctx() as ctx:
        _emit_rm_weighted_rms_norm_seg()
        module = ctx.module

    placed_text = str(module)
    placed_rms = _extract_single_device(placed_text, "rm_weighted_rms_norm_seg")

    # Load the cached prefill MLIR and splice in the placed device.
    project_root = Path(__file__).resolve().parents[1]
    cached_path = project_root / "reference_mlir" / "o_ffn.npu.air.mlir"
    cached_text = cached_path.read_text()
    original_len = len(cached_text)

    spliced = _splice_device(cached_text, "rm_weighted_rms_norm_seg", placed_rms)

    if verbose:
        print(f"  [o_ffn builder] Spliced placed-IRON rm_weighted_rms_norm_seg "
              f"into cached MLIR ({original_len} -> {len(spliced)} bytes).")

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
            _emit_rm_weighted_rms_norm_seg()
            mod = ctx.module
        text = str(mod)
    else:
        text = build_o_ffn_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
