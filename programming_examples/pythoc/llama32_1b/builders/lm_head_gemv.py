# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Placed-IRON builder for the llama32_1b LM Head GEMV kernel.

Replaces the cached AIR-stitched `lm_head_gemv.npu.air.mlir` with an
mlir-aie Python program that emits structurally equivalent
`aie/aiex`-dialect text using the dialect Python bindings directly
(no `aircc` in the loop).

Module layout (matches the cached reference structurally):

    module {
        aie.device(npu2) @p7_matvec_bf16_0 { ... }     # partition 7
        aie.device(npu2) @p6_matvec_bf16_0 { ... }     # partition 6
        ...                                            # partitions 5..1
        aie.device(npu2) @p0_matvec_bf16_0 { ... }     # partition 0
        aie.device(npu2) {                             # dispatcher
            aie.runtime_sequence @lm_head_gemv(...) {
                aiex.configure @p0_matvec_bf16_0 { aiex.run ... }
                ...
                aiex.configure @p7_matvec_bf16_0 { aiex.run ... }
            }
        }
    }

Each partition device computes one GEMV chunk of the LM Head:
weight tile (16384, 2048) x input vector (2048,) -> output tile (16384,).
With vocab_size = 128256 (Llama-3.2-1B) and N_PART = 16384, eight
partitions cover 8 * 16384 = 131072 >= 128256 (last partition is padded).

Per partition (matches the AIR-lowered cached MLIR):

  * 8 shim noc tiles  (col 0..7, row 0) for L3 <-> L2 DMA
  * 8 mem tiles       (col 0..7, row 1) for L2 staging
  * 8 compute tiles   (col 0..7, row 2) running the matvec_pythoc kernel
  * 4 locks per mem tile, 6 locks per compute tile
  * On each compute tile: 3 buffers (input vec, weight tile, 8-element
    output) and an `aie.mem` + `aie.core` block
  * On each mem tile: 2 buffers (input fan-out 1x8x2048, output collect
    1x8) and a `memtile_dma` block
  * 32 flows fanning input through mem tiles to compute tiles and
    collecting outputs back through the mem tiles
  * 17 shim_dma_allocations (8 S2MM outputs, 8 MM2S weights, 1 MM2S
    input vector) -- channel sym-names are arbitrary numeric ids
    assigned by AIR; the dispatcher device's `aiex.run` looks the
    partition sequence up by name, not by channel id
  * `aie.runtime_sequence @p<N>_matvec_bf16_0_sequence(arg0, arg1..arg16)`
    with 16 outer iterations over the K dimension (each iteration =
    8 weight DMAs + 1 input DMA + 8 output DMAs = 17 tasks).

References:
  * `reference_mlir/lm_head_gemv.npu.air.mlir` -- ground truth (19,567
    lines, produced by AIR's aircc).
  * `flash_attention/flash_attention.py` -- placed-IRON example using
    the same dialect bindings.
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
# Constants -- match the cached AIR-stitched IR for Llama-3.2-1B LM Head
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
KERNEL_OBJECT = "mv_pythoc.o"


# ---------------------------------------------------------------------------
# Channel name pools.  AIR assigns arbitrary numeric ids to its
# air_channels; what actually has to be unique is the resulting
# `aie.shim_dma_allocation` symbol within the *module*, since multiple
# `aie.device`s declare shim allocations side-by-side.  We reuse AIR's
# assignment so the dispatcher's `aiex.run` arg layout stays identical.
# ---------------------------------------------------------------------------
# Map partition index -> (S2MM_base, MM2S_base, input_id) channel numbers,
# extracted verbatim from `reference_mlir/lm_head_gemv.npu.air.mlir`.
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
    """Build a MemRefType with optional memory_space attribute (1=L2, 2=L1)."""
    from aie.extras import types as T
    from aie.ir import MemRefType, IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


# Host / L3 buffers (no memory space)
_HOST_X_TY = _bf16_np(EMB_DIM,)
_HOST_W_TY = _bf16_np(N_PART, EMB_DIM)
_HOST_Y_TY = _bf16_np(N_PART,)


def _air_channel_names(idx: int) -> Dict[str, str | List[str]]:
    """Return shim_dma_allocation symbol names for partition `idx`.

    Layout (matches AIR's allocator):
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
    """Emit one `aie.device(npu2) @p<idx>_matvec_bf16_0 { ... }` block.

    Must be called inside an `mlir_mod_ctx()` at the module insertion
    point.  This routine returns nothing; it inserts the device op as a
    side effect.

    Each partition's runtime_sequence accepts 17 args (shared input
    vector + alternating weight/output pairs) and only uses
    `arg[2*part_idx + 1]` (weight) and `arg[2*part_idx + 2]` (output).
    The other args are present so the dispatcher's `aiex.run` symbol
    type-check works.
    """
    sym = f"p{part_idx}_matvec_bf16_0"
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
        # Per-mem-tile: 4 locks (ids 3..0). lock_id 3 = w_dma_done (1),
        # lock_id 2 = w_ready (0), lock_id 1 = y_done (1), lock_id 0 = y_ready (0)
        # AIR initializes lock(3)=1, lock(2)=0, lock(1)=1, lock(0)=0
        mem_locks = {}
        for col in reversed(range(N_COLS)):
            mt = mem_tiles[col]
            mem_locks[col] = {
                "w_dma_done": lock(mt, lock_id=3, init=1),
                "w_ready":    lock(mt, lock_id=2, init=0),
                "y_done":     lock(mt, lock_id=1, init=1),
                "y_ready":    lock(mt, lock_id=0, init=0),
            }

        # Per-compute-tile: 6 locks (ids 5..0). The AIR ordering is:
        # lock_id 5 -> w_l1_avail (init 1)
        # lock_id 4 -> w_l1_ready (init 0)
        # lock_id 3 -> x_l1_avail (init 1)
        # lock_id 2 -> x_l1_ready (init 0)
        # lock_id 1 -> y_l1_done (init 1)
        # lock_id 0 -> y_l1_full (init 0)
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

        # --- Buffers --------------------------------------------------
        # Build per-region MemRefType (must be created inside the mlir_ctx
        # so the underlying mlir.Type registers in the right context).
        _W_L1_TY = _bf16_memref(K_TILE, EMB_DIM, memory_space=2)     # 4 x 2048
        _X_L1_TY = _bf16_memref(EMB_DIM, memory_space=2)             # 2048
        _Y_L1_TY = _bf16_memref(M_TILE, memory_space=2)              # 8
        _W_L2_TY = _bf16_memref(1, M_TILE, EMB_DIM, memory_space=1)  # 1 x 8 x 2048
        _Y_L2_TY = _bf16_memref(1, M_TILE, memory_space=1)           # 1 x 8

        # Mem tile buffers (one large weight buffer + one small output
        # buffer per column).  AIR names them buf319..buf304 in reverse;
        # we let MLIR auto-name and just remember the SSA handles.
        mem_buf_w = {col: buffer(mem_tiles[col], datatype=_W_L2_TY) for col in range(N_COLS)}
        mem_buf_y = {col: buffer(mem_tiles[col], datatype=_Y_L2_TY) for col in range(N_COLS)}

        # Compute tile buffers (output row, weight tile, input vector).
        core_buf_y = {col: buffer(compute_tiles[col], datatype=_Y_L1_TY) for col in range(N_COLS)}
        core_buf_w = {col: buffer(compute_tiles[col], datatype=_W_L1_TY) for col in range(N_COLS)}
        core_buf_x = {col: buffer(compute_tiles[col], datatype=_X_L1_TY) for col in range(N_COLS)}

        # External buffers for the three top-level arrays (matches AIR's
        # `aie.external_buffer` declarations in the cached IR -- present
        # mainly for symmetry; aiecc treats them as opaque references).
        external_buffer(_HOST_W_TY, name="__air_external_buffer")
        external_buffer(_HOST_X_TY, name="__air_external_buffer_1")
        external_buffer(_HOST_Y_TY, name="__air_external_buffer_2")

        # --- Compute tile mem DMAs + cores ---------------------------
        # Declare the external functions once.  Each gets link_with=...
        # so aie-assign-core-link-files routes them to mv_pythoc.o.
        # external_func declarations must live at device-region scope.
        # `llvm.emit_c_interface` is needed so the linker picks the C
        # ABI version of the function -- AIR sets the same attribute.
        from aie.ir import UnitAttr as _UnitAttr
        fill_fn = external_func(
            "linalg_fill_bf16",
            inputs=[bfloat16, _Y_L1_TY],
            link_with=KERNEL_OBJECT,
        )
        fill_fn.operation.attributes["llvm.emit_c_interface"] = _UnitAttr.get()
        matvec_fn = external_func(
            "matvec_vectorized_bf16_bf16",
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

            # aie.mem(...) block.  The decorator passes a single
            # AutoInitializingContextManagedBlockList; everything else
            # is closed over from the enclosing scope.
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
                        dma_bd(_wb, offset=0, len=K_TILE * EMB_DIM)
                        use_lock(_cl["w_ready"], LockAction.Release, value=1)
                        next_bd(block[6])
            _make_core_mem(cl, y_buf, w_buf, x_buf)

            # aie.core(...) block.  Body matches AIR exactly:
            #   loop forever {
            #     acquire y_done
            #     linalg_fill(y_buf, 0)
            #     for k in 0..M_TILE step K_TILE:
            #       acquire x_ready
            #       acquire w_ready
            #       matvec_vec(K_TILE, EMB_DIM, k, w_buf, x_buf, y_buf)
            #       release x_avail
            #       release w_avail
            #     release y_full
            #   }
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
                            # Cast loop index -> i32 for the matvec call,
                            # mirroring AIR's lowered scf.for with i32 type.
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
                # bb0: dma_start MM2S 0 -> bb1 (output forward), chain bb3
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

        # --- Shim DMA allocations ------------------------------------
        # 8 S2MM (output), 8 MM2S (weight), 1 MM2S (input vector)
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
        # Args: arg0 = input vector, then 8 x (weight, output).
        # We only DMA from the per-partition arg pair, but all 17 args
        # must be declared so the dispatcher's `aiex.run` symbol matches.
        @runtime_sequence(
            _HOST_X_TY,                                   # arg0
            *([_HOST_W_TY, _HOST_Y_TY] * N_PARTITIONS),   # arg1..arg16
            sym_name=f"{sym}_sequence",
        )
        def _seq(*args):
            arg_x = args[0]
            arg_w = args[weight_arg_idx]
            arg_y = args[out_arg_idx]

            # Stride pattern for the weight gather: (16, 131072) outer
            # selects one core's row band of 16 mini-rows; (32, 512)
            # iterates K in 32 chunks of 512; (512, 1) is the contig
            # part.  This is exactly AIR's `aie.dma_bd ... [<size=16,
            # stride=131072>, <size=32, stride=512>, <size=512,
            # stride=1>]` pattern.
            w_dims = [(16, 131072), (32, 512), (512, 1)]
            w_len = 16 * 32 * 512  # 262144

            # Input vector stride: 4 batches of 512 contig elements.
            x_dims = [(4, 512), (512, 1)]
            x_len = EMB_DIM  # 2048

            # Output stride: 16 chunks of 8 rows, each chunk strided
            # by 64 in the output dim.  Total = 128 elements per task.
            y_dims = [(16, 64), (8, 1)]
            y_len = 128

            # Per-outer-iteration constants ---------------------------------
            # Weight base offset increment per outer iteration:
            # 1024 rows of N=2048 cols, in bf16 -> 1024 * 2048 elements.
            weight_outer_stride = ROWS_PER_OUTER * EMB_DIM  # 2_097_152
            weight_col_stride = M_TILE * EMB_DIM            # 16_384
            # Output base offset increment per outer iteration:
            # 1024 elements (one row band).
            output_outer_stride = ROWS_PER_OUTER            # 1024
            output_col_stride = M_TILE                      # 8

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
                # repeat_count=31 -> 32 deliveries (matches AIR).
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

                # Await outputs in reverse order (matches AIR).
                for t in reversed(out_tasks):
                    dma_await_task(t)
                # Free input.
                dma_free_task(x_task)
                # Free weights in reverse order.
                for t in reversed(weight_tasks):
                    dma_free_task(t)


# ---------------------------------------------------------------------------
# Dispatcher device emitter.
# ---------------------------------------------------------------------------
def _emit_dispatcher_device() -> None:
    """Emit the unnamed top-level dispatcher device.

    The dispatcher carries the *outer* `aie.runtime_sequence @lm_head_gemv`
    that hands the same 17 host args to each partition's
    runtime_sequence via `aiex.configure` + `aiex.run`.
    """
    from aie.dialects._aiex_ops_gen import ConfigureOp, RunOp

    @device(AIEDevice.npu2)
    def _dispatcher():
        @runtime_sequence(
            _HOST_X_TY,
            *([_HOST_W_TY, _HOST_Y_TY] * N_PARTITIONS),
            sym_name="lm_head_gemv",
        )
        def _outer(*args):
            for idx in range(N_PARTITIONS):
                cfg = ConfigureOp(symbol=f"p{idx}_matvec_bf16_0")
                blk = cfg.body.blocks.append()
                with InsertionPoint(blk):
                    RunOp(
                        runtime_sequence_symbol=f"p{idx}_matvec_bf16_0_sequence",
                        args=list(args),
                    )
                    # No terminator needed for configure -- AIR's
                    # cached IR doesn't add one and aiecc accepts it.


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_lm_head_gemv_module(emb_dim: int = EMB_DIM) -> str:
    """Build the LM Head GEMV `aie/aiex`-dialect module.

    Args:
        emb_dim: K dimension.  Currently must be `EMB_DIM` (2048); the
            cached AIR layout is shape-specialised for the Llama-3.2-1B
            model dimensions.  Other values raise `ValueError`.

    Returns:
        The MLIR module as text -- ready to hand to
        `kernel_builder/aie_compile.compile_aie_to_elf`.
    """
    if emb_dim != EMB_DIM:
        raise ValueError(
            f"lm_head_gemv builder is currently fixed to emb_dim={EMB_DIM}; "
            f"got {emb_dim}."
        )

    with mlir_mod_ctx() as ctx:
        # AIR emits partition 7 first (highest first), then 6, ..., 0.
        for idx in range(N_PARTITIONS - 1, -1, -1):
            _emit_partition_device(idx)
        _emit_dispatcher_device()
        module = ctx.module

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
    args = parser.parse_args()
    text = build_lm_head_gemv_module()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)
