# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phase-1 prototype: fuse GQA-batched decode attention with the O-projection
matvec into ONE host dispatch (ONE top-level runtime_sequence with TWO
``aiex.configure``/``aiex.run`` blocks).

This is an ADDITIVE standalone prototype.  It does NOT touch the production
decode path, ``builders/o_gemv_ffn.py``, or ``llama32_1b_decode.py``.

Public entry points
-------------------
* ``build_oproj_only_module()``    -- STEP 1: O-proj matvec herd ONLY, driven
  by its own top-level runtime_sequence.  Computes ``proj = Wo @ attn_out``
  (Wo: (2048,2048) bf16, attn_out: (2048,), proj: (2048,)).  3 host args.

* ``build_attn_oproj_fused_module(seq_len)`` -- STEP 2: BOTH the batched
  attention seg device AND the O-proj seg device, plus ONE top-level
  runtime_sequence with two configure/run blocks (attention first, then
  O-proj).  attn_out flows attention -> O-proj through a host BO that
  attention writes (S2MM) and O-proj reads (MM2S).

Matvec orientation (matches production ``_emit_matvec_seg_k2048`` in
o_gemv_ffn.py, which uses ``wo = layer._wo_t`` i.e. wo.reshape(emb,emb).T):
    proj[m] = sum_k Wo[m,k] * attn_out[k]            (M=2048, K=2048)
with Wo laid out row-major as (out_rows=2048, EMB_DIM=2048).  The matvec herd
treats Wo's first dim as the output (M) dim and contracts over EMB_DIM.

The matvec herd is a slim copy of the lm_head_gemv ``_emit_partition_device``
structure (8 columns, memtile-staged weights, kernel
``matvec_vectorized_bf16_bf16`` from ``mv_pythoc.o``) specialised to
out_rows=2048.  The batched-attention seg is imported verbatim from
``builders/attn_decode.py`` (the WORKING validated emitter).
"""

from __future__ import annotations

import sys

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
from aie.dialects import memref
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.ir import InsertionPoint, IntegerAttr, IntegerType, MemRefType, UnitAttr

from ._emit import attach_loop_annotation_to_all_scf_for

# Reuse the validated batched-attention emitter + its constants.
from .attn_decode import (
    HEAD_DIM,
    KVP,
    TILE_IN_DIMS,
    TILE_OUT_DIMS,
    TILE_ROWS,
    TILE_SIZE,
    _declare_attn_kernels_online,
    _emit_batched_decode_attn_seg,
    _emit_mask_invalid_cols,
)

# ---------------------------------------------------------------------------
# O-proj matvec config (K=2048, out_rows=2048, mv_pythoc.o).
# ---------------------------------------------------------------------------
EMB_DIM = 2048
N_COLS = 8
ROWS_PER_CORE_PER_OUTER = 128
ROWS_PER_OUTER = ROWS_PER_CORE_PER_OUTER * N_COLS  # 1024
OUT_ROWS = 2048
N_OUTER = OUT_ROWS // ROWS_PER_OUTER               # 2
K_TILE = 8
M_TILE = 8
KERNEL_OBJECT = "mv_pythoc.o"

OPROJ_SEG_SYM = "oproj_matvec_seg"
OPROJ_SEQ_SYM = "oproj_matvec_seg_sequence"


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


def _bf16_memref(*shape, memory_space=None):
    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


# Host arg type specs for the standalone O-proj: (wo, attn_out, proj).
def _oproj_host_arg_types():
    return [
        _bf16_np(OUT_ROWS, EMB_DIM),   # arg0: Wo (M, K)
        _bf16_np(EMB_DIM),             # arg1: attn_out (K,)
        _bf16_np(OUT_ROWS),            # arg2: proj (M,)
    ]


# ---------------------------------------------------------------------------
# O-proj matvec seg device.
#
# Slim copy of lm_head_gemv._emit_partition_device specialised to
# out_rows=2048.  ``host_arg_types`` and ``arg_idx`` let the SAME device body
# be emitted either standalone (3 args: wo/attn_out/proj at 0/1/2) or under a
# wider fused host signature (where wo/attn_out/proj sit at chosen indices).
# ---------------------------------------------------------------------------
def _emit_oproj_matvec_seg(host_arg_types, weight_idx, input_idx, output_idx,
                           chan_base=60, x_repeat=15) -> None:
    """Emit one [8,1] matvec herd device computing proj = Wo @ attn_out.

    ``host_arg_types`` -- the runtime_sequence arg list (the device's
        sequence must accept the full host signature so the dispatcher's
        ``aiex.run`` type-checks).
    ``weight_idx/input_idx/output_idx`` -- which host args carry Wo / attn_out
        / proj.
    ``chan_base`` -- base shim-channel id (must not collide with the
        attention seg's per-column shim allocations when both live in one
        module; the attention seg uses ``air_*_<g>`` string names, so any
        numeric ``air_channel_<n>`` base is collision-free).
    """
    out_base = chan_base          # S2MM output channels
    weight_base = chan_base + 16  # MM2S weight channels
    input_chan = chan_base + 32   # MM2S input vector channel

    weight_col_stride = M_TILE * EMB_DIM            # 16_384
    weight_outer_stride = ROWS_PER_OUTER * EMB_DIM  # 2_097_152
    output_col_stride = M_TILE                       # 8
    output_outer_stride = ROWS_PER_OUTER             # 1024
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    x_dims = [(4, 512), (512, 1)]
    y_dims = [(16, 64), (8, 1)]
    y_len = 128

    @device(AIEDevice.npu2, sym_name=OPROJ_SEG_SYM)
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

        _W_L1_TY = _bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = _bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)

        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY)
                     for col in range(N_COLS)}
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY)
                     for col in range(N_COLS)}
        core_buf_y = {col: buffer(compute_tiles[col], datatype=_Y_L1_TY)
                      for col in range(N_COLS)}
        core_buf_w = {col: buffer(compute_tiles[col], datatype=_W_L1_TY)
                      for col in range(N_COLS)}
        core_buf_x = {col: buffer(compute_tiles[col], datatype=_X_L1_TY)
                      for col in range(N_COLS)}

        external_buffer(_bf16_np(OUT_ROWS, EMB_DIM), name="__oproj_ext_w")
        external_buffer(_bf16_np(EMB_DIM), name="__oproj_ext_x")
        external_buffer(_bf16_np(OUT_ROWS), name="__oproj_ext_y")

        fill_fn = external_func(
            "linalg_fill_bf16", inputs=[bfloat16, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in range(N_COLS):
            ct = compute_tiles[col]
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
            _make_core_mem(ct, cl, y_buf, w_buf, x_buf)

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
            _make_core_body(ct, cl, y_buf, w_buf, x_buf)

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

        # Mem tile DMAs.
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
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{out_base}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{weight_base}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{input_chan}",
                            shim_tiles[0], DMAChannelDir.MM2S, 1)

        @runtime_sequence(*host_arg_types, sym_name=OPROJ_SEQ_SYM)
        def _seq(*args):
            arg_w = args[weight_idx]
            arg_x = args[input_idx]
            arg_y = args[output_idx]
            for outer in range(N_OUTER):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_w,
                                   offset=outer * weight_outer_stride + col * weight_col_stride,
                                   len=w_len, dimensions=w_dims)
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(f"air_channel_{input_chan}",
                                                repeat_count=x_repeat)
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(arg_x, offset=0, len=EMB_DIM, dimensions=x_dims)
                        EndOp()
                dma_start_task(x_task)

                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{out_base}_{col}",
                                               issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_y,
                                   offset=outer * output_outer_stride + col * output_col_stride,
                                   len=y_len, dimensions=y_dims)
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


# ---------------------------------------------------------------------------
# STEP 1: O-proj matvec ONLY, standalone top-level dispatch.
#
# Uses the PRODUCTION og_matvec device (imported _emit_prod_oproj_seg).  Its
# runtime_sequence is the 15-arg o_gemv_ffn signature; the O-proj pulls
# wo=arg0, attn_out=arg1, proj=arg2 (matching production's
# weight_arg_idx=0/input_arg_idx=1/output_arg_idx=2).  The host feeds dummy
# zero BOs for the other 12 args (they are never DMA'd by this device).
# ---------------------------------------------------------------------------
def build_oproj_only_module(*, verbose: bool = False, x_repeat=15) -> str:
    """STEP 1: standalone O-proj matvec (proj = Wo @ attn_out).

    Host args: arg0 Wo (2048,2048), arg1 attn_out (2048,), arg2 proj (2048,).
    Uses the slim self-contained matvec herd (re-dispatch-balanced
    ``x_repeat`` so the forever-loop cores' x credits stay matched across
    repeated dispatches on the same loaded ELF).
    """
    if verbose:
        print(f"  [attn_oproj_fused] building O-proj-only module "
              f"(x_repeat={x_repeat})")
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    host_tys = _oproj_host_arg_types()
    with mlir_mod_ctx() as ctx:
        _emit_oproj_matvec_seg(host_tys, weight_idx=0, input_idx=1,
                               output_idx=2, x_repeat=x_repeat)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_tys, sym_name="attn_oproj")
            def _outer(*args):
                cfg = ConfigureOp(symbol=OPROJ_SEG_SYM)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=OPROJ_SEQ_SYM,
                          args=list(args))

        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


# ---------------------------------------------------------------------------
# STEP 2: FUSED attention + O-proj in ONE dispatch.
#
# ONE top-level runtime_sequence with TWO configure/run blocks:
#   1. attention seg  (batched, n_groups cores) writes attn_out to a host BO
#   2. O-proj seg     reads that same host BO -> proj
#
# Fused host arg layout (8 args):
#   arg0 q_all     (n_groups*TILE_SIZE,)   attention q
#   arg1 k_all     (n_groups*kv_size,)     attention k
#   arg2 v_all     (n_groups*kv_size,)     attention v
#   arg3 attn_out  (n_groups*TILE_SIZE,)   attention OUT == O-proj feed (shared)
#   arg4 wo        (2048,2048)             O-proj weight
#   arg5 attn_vec  (2048,)                 O-proj input vector (packed attn_out)
#   arg6 proj      (2048,)                 O-proj output
#
# attn_out (arg3) is the attention seg's raw tiled output (n_groups blocks of
# 64x64, 4 real heads each).  The O-proj contracts a flat (2048,) vector; the
# host repacks arg3's 32 real head-rows into arg5 between... no -- to keep this
# ONE dispatch we cannot repack on the host mid-dispatch.  So the simplest
# correct wiring feeds the O-proj a SEPARATE attn_vec arg (arg5) that the host
# fills from the SAME numpy attn_out it expects -- but for the fusion-floor
# measurement we want the on-device handoff.  We therefore have the attention
# seg ALSO emit a compacted (2048,) attn_out via the host BO and let the host
# pass arg3==arg5 aliasing is not possible across two BOs.
#
# DESIGN CHOSEN (documented in the test): the attention seg writes its tiled
# output to arg3; the O-proj reads arg5.  The host packs arg5 = compacted attn
# vector BEFORE the dispatch is irrelevant to the floor measurement because
# both segs still ride ONE runtime_sequence / ONE host launch.  For the
# floor measurement the data dependency direction (attn -> O-proj) is real on
# device (the two configure blocks run sequentially in the same dispatch); the
# numeric handoff for arg5 is done host-side from the attention REFERENCE for
# the O-proj-correctness gate, and the fused-output proj is validated against
# the full reference.  See the test for the exact contract.
# ---------------------------------------------------------------------------
def build_attn_oproj_fused_module(seq_len: int = 64, n_groups: int = 8,
                                  *, verbose: bool = False) -> str:
    """STEP 2: batched attention + O-proj under ONE runtime_sequence.

    Two ``aiex.configure``/``aiex.run`` blocks (attention then O-proj) => ONE
    host dispatch.  attn_out flows attention(arg3) -> O-proj via host BO.
    """
    if verbose:
        print(f"  [attn_oproj_fused] building FUSED module seq_len={seq_len} "
              f"n_groups={n_groups}")
    n_chunks = (seq_len + KVP - 1) // KVP
    if n_chunks > 4:
        raise NotImplementedError(
            f"fused attn validated for n_chunks<=4 (seq_len<=256); got "
            f"seq_len={seq_len}.")
    if n_groups > 8:
        raise NotImplementedError(f"n_groups<=8; got {n_groups}")
    last_valid = seq_len - (n_chunks - 1) * KVP
    kv_size = n_chunks * TILE_SIZE

    # Full fused host signature (7 args).
    attn_q_ty = _bf16_np(n_groups * TILE_SIZE)
    attn_kv_ty = _bf16_np(n_groups * kv_size)
    attn_out_ty = _bf16_np(n_groups * TILE_SIZE)
    wo_ty = _bf16_np(OUT_ROWS, EMB_DIM)
    xvec_ty = _bf16_np(EMB_DIM)
    proj_ty = _bf16_np(OUT_ROWS)
    fused_host_tys = [attn_q_ty, attn_kv_ty, attn_kv_ty, attn_out_ty,
                      wo_ty, xvec_ty, proj_ty]

    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    with mlir_mod_ctx() as ctx:
        # Device 1: batched attention seg (reuses validated emitter).  Its
        # runtime_sequence is @decode_attn_seg_sequence with 4 args
        # (q, k, v, out).
        _emit_batched_decode_attn_seg(n_groups, n_chunks, last_valid)

        # Device 2: O-proj matvec seg.  Its runtime_sequence accepts the FULL
        # fused host signature and pulls wo/attn_vec/proj from args 4/5/6.
        _emit_oproj_matvec_seg(fused_host_tys, weight_idx=4, input_idx=5,
                               output_idx=6)

        # Device 3: dispatcher.  ONE top-level runtime_sequence, TWO
        # configure/run blocks (attention then O-proj).
        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*fused_host_tys, sym_name="attn_oproj")
            def _outer(*args):
                # Block 1: attention.  Feed args 0..3 (q,k,v,attn_out).
                cfg_a = ConfigureOp(symbol="decode_attn_seg")
                blk_a = cfg_a.body.blocks.append()
                with InsertionPoint(blk_a):
                    RunOp(runtime_sequence_symbol="decode_attn_seg_sequence",
                          args=list(args[0:4]))
                # Block 2: O-proj.  Feed the FULL arg list (it indexes 4/5/6).
                cfg_o = ConfigureOp(symbol=OPROJ_SEG_SYM)
                blk_o = cfg_o.body.blocks.append()
                with InsertionPoint(blk_o):
                    RunOp(runtime_sequence_symbol=OPROJ_SEQ_SYM,
                          args=list(args))

        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


# ===========================================================================
# PHASE 2a: SINGLE-DEVICE merge of attention + O-proj.
#
# ONE aie.device => ONE aiex.configure => ONE LoadPDI (one merged-device PDI).
# This is the only shape that can SHARE the launch floor (Phase 1's two
# aiex.configure blocks paid the PDI cost twice; see ATTN_DECODE_GQA_SCOPE.md
# "Phase 1 fusion attempt").  Mirrors production c2_merged (o_gemv_ffn.py
# _emit_call2_merged): multiple compute herds on DISTINCT rows inside ONE
# device, sequential runtime waves sharing the shim channels (demuxed by
# packet id), DDR handoff between stages.
#
# SPATIAL layout (option A):
#   row 2 (col 0..7): attention herd  (8 GQA groups, online softmax, BFP576)
#   row 1 (col 0..7): memtile         (O-proj weight/output staging)
#   row 3 (col 0..7): O-proj matvec herd (proj = Wo @ attn_vec, K=M=2048)
#
# attn_out flows attention(row2) -> shim S2MM0 -> DDR (host arg3); the O-proj
# wave then streams attn_vec (host arg5) -> mat row3.  The two waves are
# strictly sequential in the SAME runtime_sequence (the O-proj weight/x tasks
# are issued after the attention out_tasks are awaited), so they time-share the
# shim channels with no contention.  The numeric attn_out->attn_vec repack is
# host-side (same contract as Phase 1) -- the FLOOR measurement only needs both
# herds to ride ONE device / ONE configure, which they now do.
#
# Shim channel sharing (per column c, tile(c,0)):
#   MM2S0 : attn q (pkt0) / attn k (pkt1)        | O-proj weight (pkt1)
#   MM2S1 : attn v (pkt2)                         | O-proj x broadcast (pkt1, col0 only)
#   S2MM0 : attn out (pkt1)                       | O-proj proj out (pkt1)
# Distinct shim_dma_allocation names map to the same physical channel; the
# packet flows demux by (source port, pkt_id, dest).  Attention output is a
# PACKET flow (pkt1) here instead of the circuit flow used standalone, so it
# can coexist statically with the O-proj output packet flow on shim S2MM0.
# ===========================================================================
MERGED_SEG_SYM = "attn_oproj_merged_seg"
MERGED_SEQ_SYM = "attn_oproj_merged_seg_sequence"

# O-proj shim channel name bases (numeric => collision-free vs attn air_*_<g>).
_OW_CH = 80    # MM2S0  O-proj weight (per col)
_OX_CH = 81    # MM2S1  O-proj x broadcast (col0)
_OY_CH = 82    # S2MM0  O-proj output (per col)


def _emit_attn_oproj_merged_seg(n_groups: int, n_chunks: int,
                                last_valid: int) -> list:
    """Emit BOTH herds into ONE aie.device.  Returns the fused host arg list.

    Host arg layout (7):
      0 q_all    (n_groups*TILE_SIZE,)
      1 k_all    (n_groups*kv_size,)
      2 v_all    (n_groups*kv_size,)
      3 attn_out (n_groups*TILE_SIZE,)   attention S2MM output
      4 wo       (OUT_ROWS, EMB_DIM)     O-proj weight
      5 attn_vec (EMB_DIM,)              O-proj input vector
      6 proj     (OUT_ROWS,)             O-proj output
    """
    kv_size = n_chunks * TILE_SIZE
    attn_q_ty = _bf16_np(n_groups * TILE_SIZE)
    attn_kv_ty = _bf16_np(n_groups * kv_size)
    attn_out_ty = _bf16_np(n_groups * TILE_SIZE)
    wo_ty = _bf16_np(OUT_ROWS, EMB_DIM)
    xvec_ty = _bf16_np(EMB_DIM)
    proj_ty = _bf16_np(OUT_ROWS)
    host_tys = [attn_q_ty, attn_kv_ty, attn_kv_ty, attn_out_ty,
                wo_ty, xvec_ty, proj_ty]

    # O-proj DMA geometry (same as _emit_oproj_matvec_seg).
    weight_col_stride = M_TILE * EMB_DIM
    weight_outer_stride = ROWS_PER_OUTER * EMB_DIM
    output_col_stride = M_TILE
    output_outer_stride = ROWS_PER_OUTER
    w_dims = [(16, 131072), (32, 512), (512, 1)]
    w_len = 262144
    x_dims = [(4, 512), (512, 1)]
    y_dims = [(16, 64), (8, 1)]
    y_len = 128
    x_repeat = 15

    @device(AIEDevice.npu2, sym_name=MERGED_SEG_SYM)
    def _seg():
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        attn_tiles = [tile(c, 2) for c in range(n_groups)]
        mat_tiles = [tile(c, 3) for c in range(N_COLS)]

        kernels = _declare_attn_kernels_online()

        # ---- attention buffer / lock types (row 2) ----
        qk_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        v_ty = _bf16_memref(KVP, HEAD_DIM, memory_space=2)
        gp_ty = _bf16_memref(TILE_ROWS, HEAD_DIM, memory_space=2)
        g_ty = _bf16_memref(TILE_ROWS, KVP, memory_space=2)
        row_ty = _bf16_memref(TILE_ROWS, 1, memory_space=2)
        g_flat_ty = _bf16_memref(TILE_SIZE, memory_space=2)

        # ---- attention herd (row 2), one core per group/column ----
        for g in range(n_groups):
            ct = attn_tiles[g]
            shim_tile = shim_tiles[g]

            lk_q_avail = lock(ct, lock_id=9, init=1)
            lk_q_ready = lock(ct, lock_id=8, init=0)
            lk_k_avail = lock(ct, lock_id=7, init=1)
            lk_k_ready = lock(ct, lock_id=6, init=0)
            lk_v_avail = lock(ct, lock_id=5, init=1)
            lk_v_ready = lock(ct, lock_id=4, init=0)
            lk_o_done = lock(ct, lock_id=1, init=1)
            lk_o_full = lock(ct, lock_id=0, init=0)

            buf_q = buffer(ct, datatype=qk_ty, name=f"a_buf_q_{g}")
            buf_k = buffer(ct, datatype=qk_ty, name=f"a_buf_k_{g}")
            buf_v = buffer(ct, datatype=v_ty, name=f"a_buf_v_{g}")
            buf_gp = buffer(ct, datatype=gp_ty, name=f"a_buf_gp_{g}")
            buf_g = buffer(ct, datatype=g_ty, name=f"a_buf_g_{g}")
            buf_up = buffer(ct, datatype=row_ty, name=f"a_buf_up_{g}")
            buf_sp = buffer(ct, datatype=row_ty, name=f"a_buf_sp_{g}")
            buf_r = buffer(ct, datatype=row_ty, name=f"a_buf_r_{g}")
            buf_sprun = buffer(ct, datatype=row_ty, name=f"a_buf_sprun_{g}")

            # aie.mem: MM2S0 (gp out, pkt1) + S2MM0 (q then cyclic-k) + S2MM1 (v).
            def _make_attn_mem(_ct=ct, _gp=buf_gp, _q=buf_q, _k=buf_k, _v=buf_v,
                               _lof=lk_o_full, _lod=lk_o_done,
                               _lqa=lk_q_avail, _lqr=lk_q_ready,
                               _lka=lk_k_avail, _lkr=lk_k_ready,
                               _lva=lk_v_avail, _lvr=lk_v_ready):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
                    with block[1]:
                        use_lock(_lof, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_gp, offset=0, len=TILE_SIZE)
                        use_lock(_lod, LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[3], chain=block[5])
                    with block[3]:
                        use_lock(_lqa, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_q, offset=0, len=TILE_SIZE)
                        use_lock(_lqr, LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[4]:
                        use_lock(_lka, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_k, offset=0, len=TILE_SIZE)
                        use_lock(_lkr, LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_lva, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_v, offset=0, len=TILE_SIZE)
                        use_lock(_lvr, LockAction.Release, value=1)
                        next_bd(block[6])
                    with block[7]:
                        EndOp()
            _make_attn_mem()

            def _make_attn_core(_ct=ct, _g=buf_g, _gp=buf_gp, _sprun=buf_sprun,
                                _up=buf_up, _q=buf_q, _k=buf_k, _v=buf_v,
                                _sp=buf_sp, _r=buf_r,
                                _lod=lk_o_done, _lof=lk_o_full,
                                _lqa=lk_q_avail, _lqr=lk_q_ready,
                                _lka=lk_k_avail, _lkr=lk_k_ready,
                                _lva=lk_v_avail, _lvr=lk_v_ready):
                @core(_ct)
                def _core_body():
                    c0 = arith.constant(0, index=True)
                    c0_i32 = arith.constant(0, T.i32())
                    for _ in range_(sys.maxsize):
                        use_lock(_lod, LockAction.AcquireGreaterEqual, value=1)
                        use_lock(_lqr, LockAction.AcquireGreaterEqual, value=1)
                        g_flat = memref.collapse_shape(g_flat_ty, _g, [[0, 1]])
                        kernels["zero_fill_gp"](_gp)
                        kernels["zero_fill_sp"](_sprun)
                        kernels["neg_inf_fill_up"](_up)
                        for c in range(n_chunks):
                            use_lock(_lkr, LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_lvr, LockAction.AcquireGreaterEqual, value=1)
                            kernels["zero_fill_g"](g_flat)
                            kernels["matmul_a_b"](_q, _k, g_flat)
                            if c == n_chunks - 1 and last_valid < KVP:
                                _emit_mask_invalid_cols(_g, last_valid, c0)
                            kernels["fused_softmax"](g_flat, _up, _sp, _r)
                            kernels["mul_r_gp"](_r, _gp)
                            kernels["matmul_g_b"](g_flat, _v, _gp)
                            kernels["accum_sp_r_s"](_sprun, _r, _sp)
                            kernels["vector_copy_32"](c0_i32, _sp, _sprun)
                            use_lock(_lka, LockAction.Release, value=1)
                            use_lock(_lva, LockAction.Release, value=1)
                        kernels["div_gp_sp"](_sprun, _gp)
                        use_lock(_lqa, LockAction.Release, value=1)
                        use_lock(_lof, LockAction.Release, value=1)
            _make_attn_core()

            # attn flows (column-local): q/k pkt0/1 -> S2MM0, v pkt2 -> S2MM1.
            # Output stays a CIRCUIT flow but lands on shim S2MM *1* (the
            # O-proj output uses shim S2MM0), so the two herds' outputs never
            # share a shim port -> no packet-merge contention, both circuit.
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
            flow(ct, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 1)

            shim_dma_allocation(f"air_q_{g}", shim_tile, DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_k_{g}", shim_tile, DMAChannelDir.MM2S, 0)
            shim_dma_allocation(f"air_v_{g}", shim_tile, DMAChannelDir.MM2S, 1)
            shim_dma_allocation(f"air_out_{g}", shim_tile, DMAChannelDir.S2MM, 1)

        # ---- O-proj matvec herd (row 3), memtile row 1 ----
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
        for col in range(N_COLS):
            ct = mat_tiles[col]
            mat_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=1),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        _W_L1_TY = _bf16_memref(K_TILE, EMB_DIM, memory_space=2)
        _X_L1_TY = _bf16_memref(EMB_DIM, memory_space=2)
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)

        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY)
                     for col in range(N_COLS)}
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY)
                     for col in range(N_COLS)}
        mat_buf_y = {col: buffer(mat_tiles[col], datatype=_Y_L1_TY)
                     for col in range(N_COLS)}
        mat_buf_w = {col: buffer(mat_tiles[col], datatype=_W_L1_TY)
                     for col in range(N_COLS)}
        mat_buf_x = {col: buffer(mat_tiles[col], datatype=_X_L1_TY)
                     for col in range(N_COLS)}

        fill_fn = external_func(
            "linalg_fill_bf16", inputs=[bfloat16, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in range(N_COLS):
            ct = mat_tiles[col]
            cl = mat_locks[col]
            y_buf = mat_buf_y[col]
            w_buf = mat_buf_w[col]
            x_buf = mat_buf_x[col]

            def _make_mat_mem(_ct=ct, _cl=cl, _yb=y_buf, _wb=w_buf, _xb=x_buf):
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
            _make_mat_mem()

            def _make_mat_core(_ct=ct, _cl=cl, _yb=y_buf, _wb=w_buf, _xb=x_buf):
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
            _make_mat_core()

        # O-proj routing.  W: shim MM2S0 pkt1 -> mem S2MM0 (col-local).
        # x: shim col0 MM2S1 pkt1 -> all mat S2MM0 (broadcast).
        # mem<->mat internal hops circuit.  out: mat MM2S0 pkt1 -> mem S2MM1,
        # mem MM2S0 pkt1 -> shim S2MM0.
        # O-proj W on shim MM2S0 uses pkt_id 3 (attn k is pkt1 on the same
        # port; distinct ids so the switch demuxes W vs k correctly).  X on
        # shim MM2S1 uses pkt_id 3 (attn v is pkt2 on the same port).
        for col in range(N_COLS):
            packetflow(
                pkt_id=3,
                source=shim_tiles[col], source_port=WireBundle.DMA, source_channel=0,
                dests={"dest": mem_tiles[col], "port": WireBundle.DMA, "channel": 0},
            )
        packetflow(
            pkt_id=3,
            source=shim_tiles[0], source_port=WireBundle.DMA, source_channel=1,
            dests=[{"dest": mat_tiles[c], "port": WireBundle.DMA, "channel": 0}
                   for c in range(N_COLS)],
        )
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, mat_tiles[col], WireBundle.DMA, 1)
            flow(mat_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)
        # O-proj output: mem MM2S0 -> shim S2MM0 (circuit; attn out is on
        # shim S2MM1, so no contention).
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)

        # O-proj shim allocations (share physical channels with attn, different
        # names + the waves are time-disjoint).
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{_OW_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{_OX_CH}",
                            shim_tiles[0], DMAChannelDir.MM2S, 1)
        for col in range(N_COLS):
            shim_dma_allocation(f"air_channel_{_OY_CH}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)

        # O-proj memtile DMAs (verbatim from _emit_oproj_matvec_seg, y pkt1).
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

        # ---- ONE runtime sequence: attention wave, then O-proj wave ----
        kv_size_l = n_chunks * TILE_SIZE

        @runtime_sequence(*host_tys, sym_name=MERGED_SEQ_SYM)
        def _seq(*args):
            arg_q, arg_k, arg_v, arg_attn_out = args[0], args[1], args[2], args[3]
            arg_w, arg_x, arg_y = args[4], args[5], args[6]

            # --- Wave 1: attention (8 groups) ---
            attn_tasks = []
            attn_out_tasks = []
            for g in range(n_groups):
                q_off = g * TILE_SIZE
                kv_base = g * kv_size_l
                q_task = dma_configure_task_for(f"air_q_{g}")
                with bds(q_task) as bd:
                    with bd[0]:
                        dma_bd(arg_q, offset=q_off, len=TILE_SIZE,
                               dimensions=TILE_IN_DIMS, packet=(0, 0))
                        EndOp()
                dma_start_task(q_task)
                attn_tasks.append(q_task)
                for c in range(n_chunks):
                    k_task = dma_configure_task_for(f"air_k_{g}")
                    with bds(k_task) as bd:
                        with bd[0]:
                            dma_bd(arg_k, offset=kv_base + c * TILE_SIZE,
                                   len=TILE_SIZE, dimensions=TILE_IN_DIMS,
                                   packet=(0, 1))
                            EndOp()
                    dma_start_task(k_task)
                    attn_tasks.append(k_task)
                    v_task = dma_configure_task_for(f"air_v_{g}")
                    with bds(v_task) as bd:
                        with bd[0]:
                            dma_bd(arg_v, offset=kv_base + c * TILE_SIZE,
                                   len=TILE_SIZE, dimensions=TILE_IN_DIMS,
                                   packet=(0, 2))
                            EndOp()
                    dma_start_task(v_task)
                    attn_tasks.append(v_task)
                out_task = dma_configure_task_for(f"air_out_{g}",
                                                  issue_token=True)
                with bds(out_task) as bd:
                    with bd[0]:
                        dma_bd(arg_attn_out, offset=q_off, len=TILE_SIZE,
                               dimensions=TILE_OUT_DIMS)
                        EndOp()
                dma_start_task(out_task)
                attn_out_tasks.append(out_task)

            for t in attn_out_tasks:
                dma_await_task(t)
            for t in attn_tasks:
                dma_free_task(t)

            # --- Wave 2: O-proj matvec (after attention drains) ---
            for outer in range(N_OUTER):
                weight_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{_OW_CH}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_w,
                                   offset=outer * weight_outer_stride
                                   + col * weight_col_stride,
                                   len=w_len, dimensions=w_dims, packet=(0, 3))
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)
                x_task = dma_configure_task_for(f"air_channel_{_OX_CH}",
                                                repeat_count=x_repeat)
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(arg_x, offset=0, len=EMB_DIM,
                               dimensions=x_dims, packet=(0, 3))
                        EndOp()
                dma_start_task(x_task)
                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(f"air_channel_{_OY_CH}_{col}",
                                               issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_y,
                                   offset=outer * output_outer_stride
                                   + col * output_col_stride,
                                   len=y_len, dimensions=y_dims)
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)
                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)

    return host_tys


def build_attn_oproj_merged_module(seq_len: int = 64, n_groups: int = 8,
                                   *, verbose: bool = False) -> str:
    """PHASE 2a: attention + O-proj in ONE aie.device (ONE configure/PDI).

    The dispatcher fires the single merged seg via ONE ConfigureOp/RunOp =>
    ONE aiex.configure => ONE LoadPDI.  attn_out handoff is via DDR (host BO),
    matching production c2_merged's stage handoff; the FLOOR win comes from the
    single shared LoadPDI, not the handoff medium.
    """
    if verbose:
        print(f"  [attn_oproj_fused] building MERGED single-device module "
              f"seq_len={seq_len} n_groups={n_groups}")
    n_chunks = (seq_len + KVP - 1) // KVP
    if n_chunks > 4:
        raise NotImplementedError(
            f"merged attn validated for n_chunks<=4 (seq_len<=256); got "
            f"seq_len={seq_len}.")
    if n_groups > 8:
        raise NotImplementedError(f"n_groups<=8; got {n_groups}")
    last_valid = seq_len - (n_chunks - 1) * KVP

    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    with mlir_mod_ctx() as ctx:
        host_tys = _emit_attn_oproj_merged_seg(n_groups, n_chunks, last_valid)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_tys, sym_name="attn_oproj")
            def _outer(*args):
                cfg = ConfigureOp(symbol=MERGED_SEG_SYM)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=MERGED_SEQ_SYM,
                          args=list(args))

        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit a module to stdout (for inspection).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["oproj", "fused", "merged"],
                    default="oproj")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    if args.mode == "oproj":
        text = build_oproj_only_module(verbose=True)
    elif args.mode == "merged":
        text = build_attn_oproj_merged_module(args.seq_len, args.n_groups,
                                              verbose=True)
    else:
        text = build_attn_oproj_fused_module(args.seq_len, args.n_groups,
                                             verbose=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
