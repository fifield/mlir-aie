# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Column-parameterized matvec bandwidth probe (ADDITIVE -- standalone).

Measures the columns -> weight-streaming-bandwidth curve of the decode GEMV
matvec engine.  A slim, column-parameterized copy of
``attn_oproj_fused._emit_oproj_matvec_seg`` whose only knobs are:

    build_matvec_bw_module(M, K, n_cols)

Computes ``proj = W @ x`` with ``W`` shape (M, K) bf16, ``x`` shape (K,) bf16,
``proj`` shape (M,) bf16.  ``M`` rows are partitioned EVENLY across ``n_cols``
columns; each column owns ``M // n_cols`` rows.  The weight stream per dispatch
is ``M * K * 2`` bytes (bf16), distributed over ``n_cols`` shim MM2S0 channels
(one per column) -- exactly the production decode weight-streaming topology, but
with the column count parameterized.

Constraints (inherited from the matvec kernel ``mv_pythoc.o``):
  * K is FIXED at 2048 -- the kernel ``matvec_vectorized_bf16_bf16`` bakes
    ``loop_range(32)`` == K/64 == 32 into its inner loop.  Vary M to vary bytes.
  * M_TILE = 8 (kernel computes 8 output rows per call / per y_full cycle).
  * M must be divisible by (n_cols * M_TILE) so every column gets a whole
    number of M_TILE blocks.

Topology per column c (matches production _emit_matvec_seg_k2048):
    weight : shim(c) MM2S0 --(circuit)--> memtile(c,1) S2MM0 --> compute(c,2)
    x      : shim(0) MM2S1 --(broadcast circuit)--> all compute(c,2) S2MM0
    output : compute(c,2) MM2S0 --> memtile(c,1) --> shim(c) S2MM0

The weight BD per column streams that column's ENTIRE row-set in ONE shim BD
(N_OUTER=1), so the slope cleanly reflects steady weight streaming.
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

K_FIXED = 2048          # kernel-baked contraction dim (loop_range(32))
M_TILE = 8              # kernel computes 8 rows per call
K_TILE = 8              # matches attn_oproj_fused (single matvec call per block)
KERNEL_OBJECT = "mv_pythoc.o"

BW_SEG_SYM = "matvec_bw_seg"
BW_SEQ_SYM = "matvec_bw_seg_sequence"


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


def _bf16_memref(*shape, memory_space=None):
    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


def bw_host_arg_types(M: int, K: int):
    """(W (M,K), x (K,), proj (M,))."""
    return [_bf16_np(M, K), _bf16_np(K), _bf16_np(M)]


def _shim_w_dims(rows_per_col: int, K: int):
    """Shim-side BD strides for streaming ``rows_per_col`` rows of one column.

    The weights for column c are a contiguous (rows_per_col, K) block in DDR
    (because W is reshaped so each column owns a contiguous row range -- see the
    runtime_sequence offsets).  A flat contiguous 1D BD over rows_per_col*K
    elements is the natural pattern; we split into the AIR (n//512,512)+(512,1)
    2D form when divisible, else a single contiguous dim.
    """
    n = rows_per_col * K
    if n >= 512 and n % 512 == 0:
        return [(n // 512, 512), (512, 1)], n
    return [(n, 1)], n


def _emit_matvec_bw_seg(M: int, K: int, n_cols: int,
                        host_arg_types, weight_idx, input_idx, output_idx,
                        chan_base=60, x_repeat=None) -> None:
    """Emit one [n_cols,1] matvec herd computing proj = W @ x.

    Row partition: column c owns rows [c*rows_per_col, (c+1)*rows_per_col).
    """
    if K != K_FIXED:
        raise ValueError(f"K must be {K_FIXED} (kernel-baked); got {K}")
    if M % (n_cols * M_TILE) != 0:
        raise ValueError(
            f"M={M} must be divisible by n_cols*M_TILE={n_cols*M_TILE}")

    rows_per_col = M // n_cols          # rows owned by each column (total)
    blocks_per_col = rows_per_col // M_TILE  # M_TILE-row blocks per column

    # The x-broadcast shim task uses repeat_count, which hardware caps at 255.
    # When a column has more than 255 M_TILE-blocks, chunk the streaming into
    # N_OUTER outer iterations so each outer re-issues x with a repeat <= 255.
    # (This mirrors the production matvec's N_OUTER loop.)  We pick the smallest
    # N_OUTER dividing blocks_per_col such that blocks_per_outer <= 255.
    MAX_REPEAT = 255
    n_outer = 1
    while (blocks_per_col // n_outer) > MAX_REPEAT or blocks_per_col % n_outer != 0:
        n_outer += 1
        if n_outer > blocks_per_col:
            raise ValueError(
                f"cannot chunk blocks_per_col={blocks_per_col} under repeat cap")
    blocks_per_outer = blocks_per_col // n_outer
    rows_per_outer = blocks_per_outer * M_TILE  # rows per column per outer
    if x_repeat is None:
        x_repeat = blocks_per_outer

    out_base = chan_base          # S2MM output channels
    weight_base = chan_base + 16  # MM2S weight channels
    input_chan = chan_base + 32   # MM2S input vector channel

    # DDR offsets (W laid out row-major (M,K); column c gets a contiguous
    # rows_per_col x K slab; within a column, outer o gets rows_per_outer rows).
    weight_col_stride = rows_per_col * K       # elements between column slabs
    weight_outer_stride = rows_per_outer * K   # elements between outers (per col)
    output_col_stride = rows_per_col           # elements between column outputs
    output_outer_stride = rows_per_outer
    w_dims, w_len = _shim_w_dims(rows_per_outer, K)
    x_dims = [(K // 512, 512), (512, 1)] if K % 512 == 0 else [(K, 1)]
    y_len = rows_per_outer
    # output memtile->shim BD: simple contiguous.
    y_dims = ([(rows_per_outer // 512, 512), (512, 1)]
              if rows_per_outer >= 512 and rows_per_outer % 512 == 0
              else [(rows_per_outer, 1)])

    @device(AIEDevice.npu2, sym_name=BW_SEG_SYM)
    def _dev():
        shim_tiles = [tile(c, 0) for c in range(n_cols)]
        mem_tiles = [tile(c, 1) for c in range(n_cols)]
        compute_tiles = [tile(c, 2) for c in range(n_cols)]

        mem_locks = {}
        for col in reversed(range(n_cols)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        core_locks = {}
        for col in range(n_cols):
            ct = compute_tiles[col]
            core_locks[col] = {
                "w_avail": lock(ct, lock_id=5, init=1),
                "w_ready": lock(ct, lock_id=4, init=0),
                "x_avail": lock(ct, lock_id=3, init=1),
                "x_ready": lock(ct, lock_id=2, init=0),
                "y_done":  lock(ct, lock_id=1, init=1),
                "y_full":  lock(ct, lock_id=0, init=0),
            }

        _W_L1_TY = _bf16_memref(K_TILE, K, memory_space=2)
        _X_L1_TY = _bf16_memref(K, memory_space=2)
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)
        _W_L2_TY = _bf16_memref(1, M_TILE, K, memory_space=1)
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)

        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY)
                     for col in range(n_cols)}
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY)
                     for col in range(n_cols)}
        core_buf_y = {col: buffer(compute_tiles[col], datatype=_Y_L1_TY)
                      for col in range(n_cols)}
        core_buf_w = {col: buffer(compute_tiles[col], datatype=_W_L1_TY)
                      for col in range(n_cols)}
        core_buf_x = {col: buffer(compute_tiles[col], datatype=_X_L1_TY)
                      for col in range(n_cols)}

        external_buffer(_bf16_np(M, K), name="__bw_ext_w")
        external_buffer(_bf16_np(K), name="__bw_ext_x")
        external_buffer(_bf16_np(M), name="__bw_ext_y")

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
                        dma_bd(_xb, offset=0, len=K)
                        use_lock(_cl["x_ready"], LockAction.Release, value=1)
                        next_bd(block[4])
                    with block[5]:
                        dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[2])
                    with block[6]:
                        use_lock(_cl["w_avail"], LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(_wb, offset=0, len=K_TILE * K)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(ct, cl, y_buf, w_buf, x_buf)

            def _make_core_body(_ct, _cl, _yb, _wb, _xb):
                import sys as _sys
                from aie.extras.dialects.arith import index_cast

                @core(_ct)
                def _core_body():
                    k_total = arith.constant(K, T.i32())
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

        # Mem tile DMAs (chain M_TILE-row blocks; loops forever, fed by shim).
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
                    dma_bd(_w, offset=0, len=M_TILE * K)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * K)
                    use_lock(_ml["w_ready"], LockAction.Release, value=1)
                    next_bd(block[6])
                with block[7]:
                    dma_start(DMAChannelDir.S2MM, 1, dest=block[8], chain=block[2])
                with block[8]:
                    use_lock(_ml["y_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_y, offset=0, len=M_TILE)
                    use_lock(_ml["y_ready"], LockAction.Release, value=1)
                    next_bd(block[8])
        for col in range(n_cols):
            _make_memtile_dma(col, mem_locks[col], mem_buf_w[col], mem_buf_y[col])

        # Shim DMA allocations.
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


def build_matvec_bw_module(M: int, K: int = K_FIXED, n_cols: int = 8,
                           *, verbose: bool = False) -> str:
    """Column-parameterized matvec proj = W @ x, partitioned over n_cols cols.

    Host args: arg0 W (M,K), arg1 x (K,), arg2 proj (M,).
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    host_tys = bw_host_arg_types(M, K)
    if verbose:
        print(f"  [matvec_bw_probe] building M={M} K={K} n_cols={n_cols} "
              f"(weight {M*K*2/1e6:.2f} MB)")
    with mlir_mod_ctx() as ctx:
        _emit_matvec_bw_seg(M, K, n_cols, host_tys, weight_idx=0, input_idx=1,
                            output_idx=2)

        @device(AIEDevice.npu2)
        def _dispatcher():
            @runtime_sequence(*host_tys, sym_name="matvec_bw")
            def _outer(*args):
                cfg = ConfigureOp(symbol=BW_SEG_SYM)
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(runtime_sequence_symbol=BW_SEQ_SYM,
                          args=list(args))

        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)
    return str(module)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-M", type=int, default=2048)
    ap.add_argument("-K", type=int, default=2048)
    ap.add_argument("-c", "--n-cols", type=int, default=8)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    text = build_matvec_bw_module(args.M, args.K, args.n_cols, verbose=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
