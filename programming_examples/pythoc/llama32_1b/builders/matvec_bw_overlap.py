# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prefetch-overlap + compute-free matvec bandwidth variants (ADDITIVE).

Standalone copies of ``builders.matvec_bw_probe`` used to attribute the
~15 GB/s decode-GEMV weight-ingest ceiling.  Three modes, selected by
``mode`` in :func:`build_matvec_bw_overlap_module`:

  * ``"baseline"`` -- byte-identical to ``matvec_bw_probe`` (single-buffered
    L1 weight slot).  Sanity-check that this file reproduces the probe.

  * ``"nop"`` -- COMPUTE-FREE.  Same shim->memtile->core weight DMA and the
    SAME lock choreography (so the DMA ring keeps cycling at the same cadence),
    but the core SKIPS the ``matvec_vectorized_bf16_bf16`` MAC -- it only
    zero-fills the output tile.  Measures pure ingest bandwidth: if BW(nop) is
    also ~15 GB/s, the wall is the DMA fabric, not compute serialization.
    Output is garbage (zeros) -- NOT numerically gated.

  * ``"pingpong"`` -- PREFETCH-OVERLAPPED.  TWO L1 weight slots (wb0+wb1),
    the core MM2S->S2MM(1) weight DMA runs as a 2-BD ring, and ``w_avail``
    starts at init=2 so the weight DMA can fill slot N+1 while the core MACs
    slot N.  The core's infinite loop is unrolled by 2 (consume wb0, then
    wb1).  Numerically identical to baseline (proj = W @ x); gated.

The L2 (memtile) staging ring is single-slot in all modes here -- the L1
double-buffer is the first-order lever; an L2 ping-pong variant (``mode=
"pingpong_l2"``) doubles the memtile W buffer too, for the case the memtile
staging is the limit.

Everything else (topology, channels, DDR offsets, runtime sequence) is copied
verbatim from matvec_bw_probe so the bench harness can drop these in.
"""

from __future__ import annotations

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
from aie.extras import types as T
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.ir import InsertionPoint, IntegerAttr, IntegerType, MemRefType, UnitAttr

from ._emit import attach_loop_annotation_to_all_scf_for

K_FIXED = 2048
M_TILE = 8
K_TILE = 8
KERNEL_OBJECT = "mv_pythoc.o"

BW_SEG_SYM = "matvec_bwo_seg"
BW_SEQ_SYM = "matvec_bwo_seg_sequence"


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


def _bf16_memref(*shape, memory_space=None):
    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


def bw_host_arg_types(M: int, K: int):
    return [_bf16_np(M, K), _bf16_np(K), _bf16_np(M)]


def _shim_w_dims(rows_per_col: int, K: int):
    n = rows_per_col * K
    if n >= 512 and n % 512 == 0:
        return [(n // 512, 512), (512, 1)], n
    return [(n, 1)], n


def _emit_matvec_bwo_seg(M: int, K: int, n_cols: int, mode: str,
                         host_arg_types, weight_idx, input_idx, output_idx,
                         chan_base=60) -> None:
    if K != K_FIXED:
        raise ValueError(f"K must be {K_FIXED} (kernel-baked); got {K}")
    if M % (n_cols * M_TILE) != 0:
        raise ValueError(
            f"M={M} must be divisible by n_cols*M_TILE={n_cols*M_TILE}")
    if mode not in ("baseline", "nop", "pingpong", "pingpong_l2"):
        raise ValueError(f"unknown mode {mode!r}")

    pingpong_w = mode in ("pingpong", "pingpong_l2")
    pingpong_l2 = mode == "pingpong_l2"
    do_mac = mode != "nop"

    # Per-tile row count.  baseline/nop use M_TILE=8 (one 32 KB W L1 slot fits).
    # pingpong needs TWO W L1 slots, and 2 x 32 KB overflows the 64 KB aie2p
    # core data memory (+ X 4 KB + Y + stack), so it halves the tile to MT=4
    # rows (2 x 16 KB W slots).  The MAC kernel takes the row-count as its
    # first arg (m), so MT just flows through as m=MT; row_offset stays 0.
    MT = 4 if pingpong_w else M_TILE

    rows_per_col = M // n_cols
    if rows_per_col % MT != 0:
        raise ValueError(f"rows_per_col={rows_per_col} not divisible by MT={MT}")
    blocks_per_col = rows_per_col // MT

    MAX_REPEAT = 255
    n_outer = 1
    while (blocks_per_col // n_outer) > MAX_REPEAT or blocks_per_col % n_outer != 0:
        n_outer += 1
        if n_outer > blocks_per_col:
            raise ValueError(
                f"cannot chunk blocks_per_col={blocks_per_col} under repeat cap")
    blocks_per_outer = blocks_per_col // n_outer
    rows_per_outer = blocks_per_outer * MT
    x_repeat = blocks_per_outer

    # When pingpong, the core unrolls the steady loop by 2 (consume wb0 then
    # wb1), so the number of MT-row blocks per outer must be even.
    if pingpong_w and blocks_per_outer % 2 != 0:
        raise ValueError(
            f"pingpong needs even blocks_per_outer; got {blocks_per_outer} "
            f"(M={M}, n_cols={n_cols}, n_outer={n_outer})")

    out_base = chan_base
    weight_base = chan_base + 16
    input_chan = chan_base + 32

    weight_col_stride = rows_per_col * K
    weight_outer_stride = rows_per_outer * K
    output_col_stride = rows_per_col
    output_outer_stride = rows_per_outer
    w_dims, w_len = _shim_w_dims(rows_per_outer, K)
    x_dims = [(K // 512, 512), (512, 1)] if K % 512 == 0 else [(K, 1)]
    y_len = rows_per_outer
    y_dims = ([(rows_per_outer // 512, 512), (512, 1)]
              if rows_per_outer >= 512 and rows_per_outer % 512 == 0
              else [(rows_per_outer, 1)])

    @device(AIEDevice.npu2, sym_name=BW_SEG_SYM)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(n_cols)]
        mem_tiles = [tile(c, 1) for c in range(n_cols)]
        compute_tiles = [tile(c, 2) for c in range(n_cols)]

        _w_dma_done_init = 2 if pingpong_l2 else 1
        mem_locks = {}
        for col in reversed(range(n_cols)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=_w_dma_done_init),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        _w_avail_init = 2 if pingpong_w else 1
        core_locks = {}
        for col in range(n_cols):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=_w_avail_init),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        # W L1 slot holds MT rows of K (16 KB at MT=4, 32 KB at MT=8).  Y L1
        # stays at M_TILE(=8) elements so linalg_fill_bf16's fixed 16-lane
        # store keeps the same headroom as baseline regardless of MT.
        _W_L1_TY = _bf16_memref(MT, K, memory_space=2)
        _X_L1_TY = _bf16_memref(K, memory_space=2)
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _bf16_memref(1, MT, K, memory_space=1)
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)

        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY)
                     for col in range(n_cols)}
        mem_buf_w1 = ({col: buffer(mem_tiles[col], datatype=_W_L2_TY)
                       for col in range(n_cols)} if pingpong_l2 else {})
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY)
                     for col in range(n_cols)}
        core_buf_y = {col: buffer(compute_tiles[col], datatype=_Y_L1_TY)
                      for col in range(n_cols)}
        core_buf_w = {col: buffer(compute_tiles[col], datatype=_W_L1_TY)
                      for col in range(n_cols)}
        core_buf_w1 = ({col: buffer(compute_tiles[col], datatype=_W_L1_TY)
                        for col in range(n_cols)} if pingpong_w else {})
        core_buf_x = {col: buffer(compute_tiles[col], datatype=_X_L1_TY)
                      for col in range(n_cols)}

        external_buffer(_bf16_np(M, K), name="__bwo_ext_w")
        external_buffer(_bf16_np(K), name="__bwo_ext_x")
        external_buffer(_bf16_np(M), name="__bwo_ext_y")

        fill_fn = external_func(
            "linalg_fill_bf16", inputs=[bfloat16, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        fill_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KERNEL_OBJECT)
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        for col in range(n_cols):
            ct = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            w_buf1 = core_buf_w1.get(col)
            x_buf = core_buf_x[col]

            def _make_core_mem(_ct, _cl, _yb, _wb, _xb, _wb1):
                @mem(_ct)
                def _core_mem(block):
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                    with block[1]:
                        use_lock(_cl["y_full"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_yb, offset=0, len=MT)
                        use_lock(_cl["y_done"], LockAction.Release, value=1)
                        next_bd(block[1])
                    with block[2]:
                        EndOp()
                    with block[3]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_cl["x_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_xb, offset=0, len=K)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    if _wb1 is None:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=MT * K)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
                    else:
                        with block[6]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb, offset=0, len=MT * K)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[7])
                        with block[7]:
                            use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                            dma_bd(_wb1, offset=0, len=MT * K)
                            use_lock(_cl["w_ready"], LockAction.Release, value=1)
                            next_bd(block[6])
            _make_core_mem(ct, cl, y_buf, w_buf, x_buf, w_buf1)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb, _wb1):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(K, T.i32())
                    # m = row count for the MAC kernel (MT rows per tile).
                    k_tile_c = arith.constant(MT, T.i32())
                    zero_bf16 = arith.constant(0.0, T.bf16())
                    k_i32_0 = arith.constant(0, T.i32())

                    if _wb1 is None:
                        # Single-buffered (baseline / nop): one tile per cycle.
                        for _ in range_(_sys.maxsize):
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if do_mac:
                                matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
                    else:
                        # Ping-pong: unroll by 2 -- consume wb0, then wb1.
                        # w_avail init=2 lets the weight DMA fill slot N+1
                        # while the core MACs slot N.
                        for _ in range_(_sys.maxsize):
                            # --- slot 0 ---
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if do_mac:
                                matvec_fn(k_tile_c, k_total, k_i32_0, _wb, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
                            # --- slot 1 ---
                            use_lock(_cl["y_done"], LockAction.AcquireGreaterEqual, value=1)
                            fill_fn(zero_bf16, _yb)
                            use_lock(_cl["x_ready"], LockAction.AcquireGreaterEqual, value=1)
                            use_lock(_cl["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                            if do_mac:
                                matvec_fn(k_tile_c, k_total, k_i32_0, _wb1, _xb, _yb)
                            use_lock(_cl["x_avail"], LockAction.Release, value=1)
                            use_lock(_cl["w_avail"], LockAction.Release, value=1)
                            use_lock(_cl["y_full"], LockAction.Release, value=1)
            _make_core_body(ct, cl, y_buf, w_buf, x_buf, w_buf1)

        for col in range(n_cols):
            flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)
        for col in range(n_cols):
            flow(shim_tiles[0], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 0)
        for col in range(n_cols):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)
        for col in range(n_cols):
            flow(mem_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        for col in range(n_cols):
            flow(compute_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)

        def _make_memtile_dma(_col, _ml, _w, _y, _w1):
            @memtile_dma(mem_tiles[_col])
            def _mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(_ml["y_ready"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=MT)
                    use_lock(_ml["y_done"], LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()
                if _w1 is None:
                    # single-slot L2 staging ring
                    with block[3]:
                        dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[5])
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=MT * K)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                    with block[6]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=MT * K)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
                    with block[7]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                    with block[8]:
                        use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_y, offset=0, len=MT)
                        use_lock(_ml["y_ready"], LockAction.Release, value=1)
                        next_bd(block[8])
                else:
                    # 2-slot L2 staging ring (w_dma_done init=2).  MM2S(1) is
                    # the shim->memtile producer (alternates w/w1); S2MM(0) is
                    # the memtile->core consumer (alternates w/w1).
                    with block[3]:
                        dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6])
                    with block[4]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=MT * K)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[5])
                    with block[5]:
                        use_lock(_ml["w_ready"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=MT * K)
                        use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[6]:
                        dma_start(DMAChannelDir.S2MM, 0, dest=block[7], chain=block[9])
                    with block[7]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w, offset=0, len=MT * K)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[8])
                    with block[8]:
                        use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_w1, offset=0, len=MT * K)
                        use_lock(_ml["w_ready"], LockAction.Release, value=1)
                        next_bd(block[7])
                    with block[9]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[10], chain=block[2])
                    with block[10]:
                        use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_y, offset=0, len=MT)
                        use_lock(_ml["y_ready"], LockAction.Release, value=1)
                        next_bd(block[10])
        for col in range(n_cols):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col],
                              mem_buf_y[col], mem_buf_w1.get(col))

        for col in range(n_cols):
            shim_dma_allocation(f"air_channel_{out_base}_{col}",
                                shim_tiles[col], DMAChannelDir.S2MM, 0)
        for col in range(n_cols):
            shim_dma_allocation(f"air_channel_{weight_base}_{col}",
                                shim_tiles[col], DMAChannelDir.MM2S, 0)
        shim_dma_allocation(f"air_channel_{input_chan}",
                            shim_tiles[0], DMAChannelDir.MM2S, 1)

        @runtime_sequence(*host_arg_types, sym_name=BW_SEQ_SYM)
        def _seq(*args):
            arg_w = args[weight_idx]
            arg_x = args[input_idx]
            arg_y = args[output_idx]
            for outer in range(n_outer):
                weight_tasks = []
                for col in range(n_cols):
                    t = dma_configure_task_for(f"air_channel_{weight_base}_{col}")
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_w,
                                   offset=col * weight_col_stride
                                   + outer * weight_outer_stride,
                                   len=w_len, dimensions=w_dims)
                            EndOp()
                    dma_start_task(t)
                    weight_tasks.append(t)

                x_task = dma_configure_task_for(f"air_channel_{input_chan}",
                                                repeat_count=x_repeat)
                with bds(x_task) as bd:
                    with bd[0]:
                        dma_bd(arg_x, offset=0, len=K, dimensions=x_dims)
                        EndOp()
                dma_start_task(x_task)

                out_tasks = []
                for col in range(n_cols):
                    t = dma_configure_task_for(f"air_channel_{out_base}_{col}",
                                               issue_token=True)
                    with bds(t) as bd:
                        with bd[0]:
                            dma_bd(arg_y,
                                   offset=col * output_col_stride
                                   + outer * output_outer_stride,
                                   len=y_len, dimensions=y_dims)
                            EndOp()
                    dma_start_task(t)
                    out_tasks.append(t)

                for t in reversed(out_tasks):
                    dma_await_task(t)
                dma_free_task(x_task)
                for t in reversed(weight_tasks):
                    dma_free_task(t)


def build_matvec_bw_overlap_module(M: int, K: int = K_FIXED, n_cols: int = 8,
                                   *, mode: str = "pingpong",
                                   verbose: bool = False) -> str:
    """Column-parameterized matvec proj = W @ x with an ingest-mode knob.

    mode in {"baseline","nop","pingpong","pingpong_l2"}.
    Host args: arg0 W (M,K), arg1 x (K,), arg2 proj (M,).
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    host_tys = bw_host_arg_types(M, K)
    if verbose:
        print(f"  [matvec_bw_overlap] M={M} K={K} n_cols={n_cols} mode={mode} "
              f"(weight {M*K*2/1e6:.2f} MB)")
    with mlir_mod_ctx() as ctx:
        _emit_matvec_bwo_seg(M, K, n_cols, mode, host_tys, weight_idx=0,
                             input_idx=1, output_idx=2)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_tys, sym_name="matvec_bw")
            def _outer(*args):
                cfg = ConfigureOp(symbol=BW_SEG_SYM)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=BW_SEQ_SYM, args=list(args))

        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-M", type=int, default=2048)
    ap.add_argument("-K", type=int, default=2048)
    ap.add_argument("-c", "--n-cols", type=int, default=8)
    ap.add_argument("-m", "--mode", default="pingpong")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    text = build_matvec_bw_overlap_module(args.M, args.K, args.n_cols,
                                          mode=args.mode, verbose=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
