# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b LM Head GEMV (AWQ uint4) kernel.

This is the AWQ uint4 counterpart of ``builders/lm_head_gemv.py``. It
handles the same 8-partition vocab=128256 LM head GEMV, but with
packed-uint4 weight buffers and the AWQ matvec kernel
(``awq_matvec_vectorized_u4_bf16`` + ``awq_linalg_fill_bf16`` from
``awq_mv_pythoc.o``). Topology, locks, flows, and channel-id assignments
are identical to the BF16 sibling -- the only thing that changes is the
weight buffer element type / shape (bf16 (K_TILE, EMB_DIM) becomes
ui8 (K_TILE, ROW_BYTES) where ROW_BYTES = EMB_DIM/2 + 4 *
(EMB_DIM/group_size) = 1088) and the weight DMA stride pattern
(borrowed verbatim from ``builders/o_gemv_ffn_awq.py``'s
``_emit_awq_matvec_seg_k2048``).

There is no cached AIR-stitched MLIR for ``lm_head_gemv_awq``; this
builder is invented fresh. We model it on the BF16 ``lm_head_gemv``
layout (8 partitions x 8 cols, runtime_sequence per partition, top-level
dispatcher with ``aiex.configure`` + ``aiex.run`` per partition) and on
the AWQ K=2048 matvec segment in ``o_gemv_ffn_awq.py``.

Module layout (identical structure to ``lm_head_gemv``):

    module {
        aie.device(npu2) @p7_awq_matvec_bf16_0 { ... }    # partition 7
        ...                                               # partitions 6..0
        aie.device(npu2) @p0_awq_matvec_bf16_0 { ... }    # partition 0
        aie.device(npu2) {                                # dispatcher
            aie.runtime_sequence @lm_head_gemv_awq(...) {
                aiex.configure @p0_awq_matvec_bf16_0 { aiex.run ... }
                ...
                aiex.configure @p7_awq_matvec_bf16_0 { aiex.run ... }
            }
        }
    }

References:
  * ``builders/lm_head_gemv.py`` -- BF16 sibling, structural template.
  * ``builders/o_gemv_ffn_awq.py::_emit_awq_matvec_seg_k2048`` -- AWQ
    K=2048 matvec pattern (weight DMA dims, ROW_BYTES math, ui8
    memref helper).
"""

from __future__ import annotations

from typing import Dict, List

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
from ._emit import attach_loop_annotation_to_all_scf_for
from aie.extras.dialects import arith
from aie.helpers.dialects.scf import _for as range_
from aie.ir import (
    Attribute,
    DictAttr,
    FlatSymbolRefAttr,
    IntegerAttr,
    IntegerType,
    InsertionPoint,
    StringAttr,
)


# ---------------------------------------------------------------------------
# Constants -- match the BF16 sibling for Llama-3.2-1B LM Head
# ---------------------------------------------------------------------------
N_PART = 16384       # rows of W per partition (vocab_size / 8, padded)
EMB_DIM = 2048       # K dimension (matches model config)
N_PARTITIONS = 8
N_COLS = 8           # 8 compute columns per partition device
ROWS_PER_CORE_PER_OUTER = 128   # rows handled per compute core per outer iter
ROWS_PER_OUTER = ROWS_PER_CORE_PER_OUTER * N_COLS  # = 1024
N_OUTER = N_PART // ROWS_PER_OUTER                 # = 16
K_TILE = 4           # inner K tiling factor for the matvec kernel
M_TILE = 8           # rows processed per matvec call
KERNEL_OBJECT = "awq_mv_pythoc.o"

# AWQ packed-uint4 row layout: K/2 packed nibbles + 4 bytes per group
# of (scale + zero) parameters.  At EMB_DIM=2048, group_size=128:
# 2048/2 + 4 * (2048/128) = 1024 + 64 = 1088 bytes per output row.
GROUP_SIZE = 128


def _combined_row_bytes(k: int, group_size: int = GROUP_SIZE) -> int:
    return k // 2 + 4 * (k // group_size)


ROW_BYTES = _combined_row_bytes(EMB_DIM)  # = 1088


# ---------------------------------------------------------------------------
# Channel name pools.  Reuse the BF16 sibling's `_CHANNEL_MAP` verbatim.
# These ids are local to each partition device's shim_dma_allocation
# symbols, so reusing them across the AWQ build is fine -- the
# dispatcher's `aiex.run` looks the partition sequence up by name.
# ---------------------------------------------------------------------------
_CHANNEL_MAP: Dict[int, Dict[str, int]] = {
    0: {"s2mm": 44, "mm2s": 54, "input": 1},
    1: {"s2mm": 45, "mm2s": 48, "input": 6},
    2: {"s2mm": 55, "mm2s": 46, "input": 11},
    3: {"s2mm": 42, "mm2s": 47, "input": 16},
    4: {"s2mm": 51, "mm2s": 50, "input": 21},
    5: {"s2mm": 53, "mm2s": 40, "input": 26},
    6: {"s2mm": 49, "mm2s": 52, "input": 31},
    7: {"s2mm": 41, "mm2s": 43, "input": 36},
}


def _bf16_np(*shape):
    return np.ndarray[shape, np.dtype[bfloat16]]


def _bf16_memref(*shape, memory_space=None):
    """Build a bf16 MemRefType with optional memory_space attribute (1=L2, 2=L1)."""
    from aie.extras import types as T
    from aie.ir import MemRefType, IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


# Host / L3 buffers.  Input vector and output stay bf16; the host
# weight matrix is packed-uint4 with the AWQ ROW_BYTES layout.
_HOST_X_TY = _bf16_np(EMB_DIM,)
_HOST_W_TY = np.ndarray[(N_PART, ROW_BYTES), np.dtype[np.uint8]]
_HOST_Y_TY = _bf16_np(N_PART,)


def _air_channel_names(idx: int) -> Dict[str, str | List[str]]:
    """Return shim_dma_allocation symbol names for partition `idx`.

    Layout (matches the BF16 sibling):
      out_<col>  -> "air_channel_<s2mm>_<col>"  S2MM, channel 0, col=0..7
      w_<col>    -> "air_channel_<mm2s>_<col>"  MM2S, channel 0, col=0..7
      input      -> "air_channel_<input>"      MM2S, channel 1, col=0
    """
    base = _CHANNEL_MAP[idx]
    return {
        "out": [f"air_channel_{base['s2mm']}_{c}" for c in range(N_COLS)],
        "w":   [f"air_channel_{base['mm2s']}_{c}" for c in range(N_COLS)],
        "input": f"air_channel_{base['input']}",
    }


# ---------------------------------------------------------------------------
# Partition device emitter.
# ---------------------------------------------------------------------------
def _emit_partition_device(part_idx: int) -> None:
    """Emit one `aie.device(npu2) @p<idx>_awq_matvec_bf16_0 { ... }` block.

    Must be called inside an `mlir_mod_ctx()` at the module insertion
    point.  Returns nothing; inserts the device op as a side effect.

    Each partition's runtime_sequence accepts 17 args (shared input
    vector + alternating weight/output pairs) and only DMAs from
    `arg[2*part_idx + 1]` (weight) and `arg[2*part_idx + 2]` (output).
    The other args are present so the dispatcher's `aiex.run` symbol
    type-check works.
    """
    sym = f"p{part_idx}_awq_matvec_bf16_0"
    chans = _air_channel_names(part_idx)
    weight_arg_idx = 2 * part_idx + 1
    out_arg_idx = 2 * part_idx + 2

    @device(AIEDevice.npu2, sym_name=sym)
    def part_device():
        # --- Tile declarations ----------------------------------------
        shim_tiles = [tile(c, 0) for c in range(N_COLS)]
        mem_tiles = [tile(c, 1) for c in range(N_COLS)]
        compute_tiles = [tile(c, 2) for c in range(N_COLS)]

        # --- Locks ----------------------------------------------------
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

        # --- Buffer types --------------------------------------------
        # Weight buffers are ui8 (packed AWQ rows); input/output stay bf16.
        from aie.ir import MemRefType, IntegerAttr, IntegerType
        from aie.extras import types as T

        def _ui8_memref(*shape, memory_space=None):
            ms = None
            if memory_space is not None:
                ms = IntegerAttr.get(
                    IntegerType.get_signless(32), memory_space)
            return MemRefType.get(list(shape), T.ui8(), None, ms)

        _W_L1_TY = _ui8_memref(K_TILE, ROW_BYTES, memory_space=2)        # 4 x 1088 ui8
        _X_L1_TY = _bf16_memref(EMB_DIM, memory_space=2)                  # 2048 bf16
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)                   # 8 bf16
        _W_L2_TY = _ui8_memref(1, M_TILE, ROW_BYTES, memory_space=1)      # 1 x 8 x 1088 ui8
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)                # 1 x 8 bf16

        # Mem tile buffers (one packed-weight buffer + one small output
        # buffer per column).
        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY) for col in range(N_COLS)}
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY) for col in range(N_COLS)}

        # Compute tile buffers (output row, weight tile, input vector).
        core_buf_y = {col: buffer(compute_tiles[col], datatype=_Y_L1_TY) for col in range(N_COLS)}
        core_buf_w = {col: buffer(compute_tiles[col], datatype=_W_L1_TY) for col in range(N_COLS)}
        core_buf_x = {col: buffer(compute_tiles[col], datatype=_X_L1_TY) for col in range(N_COLS)}

        # External buffers: weight (N_PART, ROW_BYTES) ui8, input
        # (EMB_DIM,) bf16, output (N_PART,) bf16.  Mirrors the BF16
        # sibling's three declarations but with the appropriate types.
        external_buffer(_HOST_W_TY, name="__air_external_buffer")
        external_buffer(_HOST_X_TY, name="__air_external_buffer_1")
        external_buffer(_HOST_Y_TY, name="__air_external_buffer_2")

        # --- External function declarations --------------------------
        from aie.ir import UnitAttr as _UnitAttr
        fill_fn = external_func(
            "awq_linalg_fill_bf16",
            inputs=[bfloat16, _Y_L1_TY],
            link_with=KERNEL_OBJECT,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = _UnitAttr.get()
        matvec_fn = external_func(
            "awq_matvec_vectorized_u4_bf16",
            inputs=[np.int32, np.int32, np.int32, _W_L1_TY, _X_L1_TY, _Y_L1_TY],
            link_with=KERNEL_OBJECT,
        )
        matvec_fn.operation.attributes["llvm.emit_c_interface"] = _UnitAttr.get()

        for col in range(N_COLS):
            ct = compute_tiles[col]
            cl = core_locks[col]
            y_buf = core_buf_y[col]
            w_buf = core_buf_w[col]
            x_buf = core_buf_x[col]

            def _make_core_mem(_cl, _yb, _wb, _xb):
                @mem(ct)
                def _core_mem(block):
                    # bb0: dma_start MM2S 0 -> bb1, fallthrough bb3
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
                        # AWQ: len is in ui8 elements -> K_TILE * ROW_BYTES bytes.
                        dma_bd(_wb, offset=0, len=K_TILE * ROW_BYTES)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(cl, y_buf, w_buf, x_buf)

            def _make_core_body(_cl, _yb, _wb, _xb):
                @core(ct)
                def _core_body():
                    import sys as _sys
                    from aie.extras import types as T
                    from aie.extras.dialects.arith import index_cast
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
            _make_core_body(cl, y_buf, w_buf, x_buf)

        # --- Flows ----------------------------------------------------
        # Shim->mem DMA0 fan-in (weights), shim col->mem col, channel 0
        for col in range(N_COLS):
            flow(shim_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 0)
        # Shim 0 broadcasts the input vector on DMA1 -> every compute tile DMA0
        for col in range(N_COLS):
            flow(shim_tiles[0], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 0)
        # Mem->shim DMA0 (outputs)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 0, shim_tiles[col], WireBundle.DMA, 0)
        # Mem->compute DMA1 (weights)
        for col in range(N_COLS):
            flow(mem_tiles[col], WireBundle.DMA, 1, compute_tiles[col], WireBundle.DMA, 1)
        # Compute->mem DMA0 (outputs)
        for col in range(N_COLS):
            flow(compute_tiles[col], WireBundle.DMA, 0, mem_tiles[col], WireBundle.DMA, 1)

        # --- Mem tile DMAs -------------------------------------------
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
                    # AWQ: len is in ui8 elements -> M_TILE * ROW_BYTES bytes.
                    dma_bd(_w, offset=0, len=M_TILE * ROW_BYTES)
                    use_lock(_ml["w_dma_done"], LockAction.Release, value=1)
                    next_bd(block[4])
                with block[5]:
                    dma_start(DMAChannelDir.S2MM, 0, dest=block[6], chain=block[7])
                with block[6]:
                    use_lock(_ml["w_dma_done"], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(_w, offset=0, len=M_TILE * ROW_BYTES)
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

        # --- Shim DMA allocations ------------------------------------
        for col in range(N_COLS):
            shim_dma_allocation(
                chans["out"][col],
                shim_tiles[col],
                DMAChannelDir.S2MM,
                0,
            )
        for col in range(N_COLS):
            shim_dma_allocation(
                chans["w"][col],
                shim_tiles[col],
                DMAChannelDir.MM2S,
                0,
            )
        shim_dma_allocation(
            chans["input"],
            shim_tiles[0],
            DMAChannelDir.MM2S,
            1,
        )

        # --- Runtime sequence ----------------------------------------
        @runtime_sequence(
            _HOST_X_TY,                                   # arg0
            *([_HOST_W_TY, _HOST_Y_TY] * N_PARTITIONS),   # arg1..arg16
            sym_name=f"{sym}_sequence",
        )
        def _seq(*args):
            arg_x = args[0]
            arg_w = args[weight_arg_idx]
            arg_y = args[out_arg_idx]

            # Weight DMA stride pattern -- copied verbatim from
            # `o_gemv_ffn_awq.py::_emit_awq_matvec_seg_k2048`:
            #   16 mini-rows of 16 chunks of 544 bytes = 139264 bytes
            #   = 8 rows * 1088 bytes * 16 mem-tile cycles.
            w_dims = [(16, 69632), (16, 544), (544, 1)]
            w_len = 16 * 16 * 544  # 139264

            # Input vector stride: 4 batches of 512 contig elements (bf16).
            x_dims = [(4, 512), (512, 1)]
            x_len = EMB_DIM  # 2048

            # Output stride: 16 chunks of 8 rows, each chunk strided
            # by 64 in the output dim.  Total = 128 elements per task.
            y_dims = [(16, 64), (8, 1)]
            y_len = 128

            # Per-outer-iteration constants ---------------------------
            # AWQ weight base offset increment per outer iteration:
            # 1024 rows of ROW_BYTES bytes -> 1024 * 1088 bytes.
            weight_outer_stride = ROWS_PER_OUTER * ROW_BYTES   # 1024 * 1088 = 1_114_112
            weight_col_stride = M_TILE * ROW_BYTES             # 8 * 1088 = 8704
            # Output base offset increment per outer iteration: 1024 bf16 elts.
            output_outer_stride = ROWS_PER_OUTER               # 1024
            output_col_stride = M_TILE                         # 8

            for outer in range(N_OUTER):
                weight_tasks = []
                # 8 weight MM2S tasks (one per column)
                for col in range(N_COLS):
                    t = dma_configure_task_for(chans["w"][col])
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

                # 1 input vector MM2S task (broadcast to 8 cores).
                x_task = dma_configure_task_for(chans["input"], repeat_count=31)
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

                # 8 output S2MM tasks (one per column), issue_token=True
                out_tasks = []
                for col in range(N_COLS):
                    t = dma_configure_task_for(chans["out"][col], issue_token=True)
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
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device() -> None:
    """Emit the unnamed top-level dispatcher device.

    The dispatcher carries the *outer* `aie.runtime_sequence
    @lm_head_gemv_awq` that hands the same 17 host args to each
    partition's runtime_sequence via `aiex.configure` + `aiex.run`.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            _HOST_X_TY,
            *([_HOST_W_TY, _HOST_Y_TY] * N_PARTITIONS),
            sym_name="lm_head_gemv_awq",
        )
        def _outer(*args):
            for idx in range(N_PARTITIONS):
                cfg = ConfigureOp(symbol=f"p{idx}_awq_matvec_bf16_0")
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(
                        runtime_sequence_symbol=f"p{idx}_awq_matvec_bf16_0_sequence",
                        args=list(args),
                    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_lm_head_gemv_awq_module(emb_dim: int = EMB_DIM) -> str:
    """Build the AWQ LM Head GEMV `aie/aiex`-dialect module.

    Args:
        emb_dim: K dimension.  Currently must be `EMB_DIM` (2048); the
            AWQ row layout (ROW_BYTES = 1088) and weight DMA stride
            pattern are shape-specialised for the Llama-3.2-1B model
            dimensions.  Other values raise `ValueError`.

    Returns:
        The MLIR module as text -- ready to hand to
        `kernel_builder/aie_compile.compile_aie_to_elf`.
    """
    if emb_dim != EMB_DIM:
        raise ValueError(
            f"lm_head_gemv_awq builder is currently fixed to emb_dim={EMB_DIM}; "
            f"got {emb_dim}."
        )

    with mlir_mod_ctx() as ctx:
        # Emit partitions 7..0 to mirror the BF16 sibling's order.
        for idx in range(N_PARTITIONS - 1, -1, -1):
            _emit_partition_device(idx)
        _emit_dispatcher_device()
        module = ctx.module
        attach_loop_annotation_to_all_scf_for(module)

    return str(module)


# ---------------------------------------------------------------------------
# CLI -- emit the module to stdout (useful for diffing vs the BF16 sibling).
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
    args = parser.parse_args()
    text = build_lm_head_gemv_awq_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
