#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# High-performance GEMV (M=1) example with multi-row cascade reduction.
# Ported (and scaled to npu3 = 3 cols x 4 compute rows) from
#   pacifica-apps/kernels/llama2/llama_4096_512/iron.py
#
# This example demonstrates:
#   - 12-core spatial tiling (3 cols x 4 rows)
#   - Activation broadcast from a single shim to all 12 compute cores
#   - Per-column weight fan-out through the memtile (object_fifo_link)
#   - North->south cascade reduction within each column
#
# Problem: M=1, K=4096, N=192, INT8 x INT8 -> INT32.
#
# NPU3 shim DMA budget = 4 MM2S + 2 S2MM per shim, so the shim cannot
# afford one weight stream per (c, r) directly.  Instead each compute
# column's shim sends a single 256-KB blob of all 4 row weight slabs
# to its memtile, and the memtile splits that blob into 4 per-core
# FIFOs via object_fifo_link (the vector_add_multicol pattern).

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_


# Hardware partitioning
N_COLS         = 3
COMPUTE_ROWS   = [2, 3, 4, 5]
N_ROWS         = len(COMPUTE_ROWS)

# Problem geometry (must match constants in kernel.cc)
TILE_K         = 64
TILE_N         = 64
K_TOTAL        = 4096
N_TOTAL        = N_COLS * TILE_N            # 192
K_PER_ROW      = K_TOTAL // N_ROWS          # 1024
K_TILES_PER_ROW = K_PER_ROW // TILE_K       # 16
N_PER_COL      = TILE_N                     # 64

# Per-core data sizes.  Each core's 64-KB weight slab is split along
# the N dimension and streamed via TWO parallel shim MM2S channels
# (mirroring the source design's two weight streams).  W_lo carries N
# cols 0..31 of every K-tile; W_hi carries N cols 32..63.
#
# Compute tiles have only 2 input DMA channels.  Since each core needs
# A + W_lo + W_hi (= 3 streams), we embed this core's per-row 1024-byte
# A slice at the FRONT of its W_lo buffer:
#   AW_lo per core = [A_row_slice 1 KB | W_lo_slab 32 KB] = 33 KB
#   W_hi  per core = 32 KB
# Two FIFOs per core, fits the 2-input-channel budget.
A_SLICE_BYTES        = K_PER_ROW                              # 1024 B per row
W_HALF_BYTES         = K_PER_ROW * TILE_N // 2                # 32768 B
AW_LO_PER_CORE_BYTES = A_SLICE_BYTES + W_HALF_BYTES           # 33792 B
W_HI_PER_CORE_BYTES  = W_HALF_BYTES                           # 32768 B
AW_LO_COL_BYTES      = N_ROWS * AW_LO_PER_CORE_BYTES          # 135168 B per shim->memtile blob
W_HI_COL_BYTES       = N_ROWS * W_HI_PER_CORE_BYTES           # 131072 B per shim->memtile blob

BUF_DEPTH            = 2


def my_gemv(dev, iters_per_call=1, enable_trace=False, trace_size=8192):
    @device(dev)
    def device_body():
        # ---------------------------------------------------------------
        # Element types
        # ---------------------------------------------------------------
        aw_lo_ty       = np.ndarray[(AW_LO_PER_CORE_BYTES,), np.dtype[np.int8]]
        w_hi_ty        = np.ndarray[(W_HI_PER_CORE_BYTES,),  np.dtype[np.int8]]
        aw_lo_col_ty   = np.ndarray[(AW_LO_COL_BYTES,),      np.dtype[np.int8]]
        w_hi_col_ty    = np.ndarray[(W_HI_COL_BYTES,),       np.dtype[np.int8]]
        c_slab_ty      = np.ndarray[(N_PER_COL,),            np.dtype[np.int32]]

        # Host tensors: A is gone (embedded), W tensor is per-col
        # [AW_lo_blob | W_hi_blob] concatenated.
        W_TENSOR_BYTES = N_COLS * (AW_LO_COL_BYTES + W_HI_COL_BYTES)
        w_tensor_ty = np.ndarray[(W_TENSOR_BYTES,), np.dtype[np.int8]]
        c_tensor_ty = np.ndarray[(N_TOTAL,),       np.dtype[np.int32]]

        # ---------------------------------------------------------------
        # External kernels (one per cascade role).  All take the FULL A
        # vector plus a row index so each core can offset to its own
        # K_PER_ROW slice without a separate per-row FIFO on the shim.
        # ---------------------------------------------------------------
        gemv_core_send = external_func(
            "gemv_core_send",
            inputs=[aw_lo_ty, w_hi_ty],
            link_with="project/kernel.o",
        )
        gemv_core_pass = external_func(
            "gemv_core_pass",
            inputs=[aw_lo_ty, w_hi_ty],
            link_with="project/kernel.o",
        )
        gemv_core_sink = external_func(
            "gemv_core_sink",
            inputs=[aw_lo_ty, w_hi_ty, c_slab_ty],
            link_with="project/kernel.o",
        )

        # ---------------------------------------------------------------
        # Tiles
        # ---------------------------------------------------------------
        shims = [tile(c, 0) for c in range(N_COLS)]
        for c, sh in enumerate(shims):
            sh.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, c])
        mems  = [tile(c, 1) for c in range(N_COLS)]
        cores = [[tile(c, r) for r in COMPUTE_ROWS] for c in range(N_COLS)]

        # Required Control32 packetflow per shim (sibling examples do this).
        for c in range(N_COLS):
            packetflow(
                c,
                shims[c],
                WireBundle.Control32,
                0,
                [{"dest": shims[0], "port": WireBundle.Controller32, "channel": 0}],
            )

        # ---------------------------------------------------------------
        # Weights: each column's shim sends TWO blobs to its memtile via
        # two parallel MM2S channels:
        #   AW_lo blob (135 KB) = per-row [A_slice 1 KB | W_lo_slab 32 KB]
        #                         x N_ROWS rows
        #   W_hi  blob (128 KB) = per-row W_hi_slab 32 KB x N_ROWS rows
        # The memtile splits each blob into 4 per-(c,r) FIFOs via
        # object_fifo_link with row offsets.  The activation slice rides
        # along inside AW_lo so the per-core compute tile only consumes
        # 2 input FIFOs (within the 2-input-channel hard limit).
        #
        # Per-core L1 footprint: AW_lo (33 KB, depth 1) + W_hi (32 KB,
        # depth 1) + C (256 B, depth 2) + stack ~1 KB = ~66 KB.  Fits in
        # the 128-KB AIE4 core local memory.
        # ---------------------------------------------------------------
        aw_lo_shim_to_mem = []
        w_hi_shim_to_mem  = []
        aw_lo_per_core    = [[None] * N_ROWS for _ in range(N_COLS)]
        w_hi_per_core     = [[None] * N_ROWS for _ in range(N_COLS)]
        aw_lo_row_offsets = [r_idx * AW_LO_PER_CORE_BYTES for r_idx in range(N_ROWS)]
        w_hi_row_offsets  = [r_idx * W_HI_PER_CORE_BYTES  for r_idx in range(N_ROWS)]
        for c in range(N_COLS):
            aw_lo_shim_to_mem.append(
                object_fifo(f"AWlo_s{c}", shims[c], mems[c], BUF_DEPTH,
                            aw_lo_col_ty)
            )
            w_hi_shim_to_mem.append(
                object_fifo(f"Whi_s{c}",  shims[c], mems[c], BUF_DEPTH,
                            w_hi_col_ty)
            )
            aw_fifos, hi_fifos = [], []
            for r_idx in range(N_ROWS):
                aw_lo_per_core[c][r_idx] = object_fifo(
                    f"AWlo_m{c}_r{r_idx}", mems[c], cores[c][r_idx],
                    1, aw_lo_ty,
                )
                w_hi_per_core[c][r_idx] = object_fifo(
                    f"Whi_m{c}_r{r_idx}",  mems[c], cores[c][r_idx],
                    1, w_hi_ty,
                )
                aw_fifos.append(aw_lo_per_core[c][r_idx])
                hi_fifos.append(w_hi_per_core[c][r_idx])
            object_fifo_link(aw_lo_shim_to_mem[c], aw_fifos, [], aw_lo_row_offsets)
            object_fifo_link(w_hi_shim_to_mem[c],  hi_fifos, [], w_hi_row_offsets)

        # ---------------------------------------------------------------
        # Output: one S2MM per column from the cascade root (row index 0
        # = hw row 2) directly to its column's shim.  64 INT32 = 256 B.
        # ---------------------------------------------------------------
        c_fifos = [
            object_fifo(f"C_c{c}", cores[c][0], shims[c], BUF_DEPTH, c_slab_ty)
            for c in range(N_COLS)
        ]

        # ---------------------------------------------------------------
        # Cascade flows: N->S within each column.
        # ---------------------------------------------------------------
        for c in range(N_COLS):
            for r_idx in range(N_ROWS - 1, 0, -1):
                cascade_flow(cores[c][r_idx], cores[c][r_idx - 1])

        # ---------------------------------------------------------------
        # Optional trace setup.  Capture compute/memory events on the
        # cascade-root core (col 0, row 5) and cascade-sink core (col 0,
        # row 2), plus shim DMA events.  All trace packets route to
        # shim col 0 S2MM channel 1, draining into the host trace buffer.
        # ---------------------------------------------------------------
        if enable_trace:
            from aie.dialects.aie import trace, trace_mode, trace_packet, \
                trace_event, trace_start, trace_stop, \
                TraceMode, TracePacketType

            # Bisect: trace a CORE tile instead of the shim, matching
            # event_trace's working topology.  Pick the cascade-sink
            # core (col 0, row 2) which is the cascade-reduction root
            # for col 0's output.
            @trace(cores[0][0], "core_trace")
            def _():
                trace_mode(TraceMode.EventTime)
                trace_packet(10, TracePacketType.Core)
                trace_event("INSTR_EVENT_0")
                trace_event("INSTR_VECTOR")
                trace_event("MEMORY_STALL")
                trace_event("STREAM_STALL")
                trace_event("LOCK_STALL")
                trace_start(broadcast=0)
                trace_stop(broadcast=1)

            # Note: the trace lowering pass auto-generates the
            # Trace32 -> shim DMA packetflow from the trace.config op
            # above, so we do NOT declare an explicit packetflow here.
            # Declaring it manually causes a duplicate packet_flow(10)
            # in the lowered MLIR which deadlocks the run on npu3.

        # ---------------------------------------------------------------
        # Per-core bodies.
        # @core(...) runs the body immediately, so per-iteration FIFO
        # references are captured at decoration time.  Use factory helpers
        # so each call to @core sees its own closure cleanly.
        # ---------------------------------------------------------------
        def make_sink_core(tile_, awf, whf, cf):
            @core(tile_)
            def _():
                for _ in range_(sys.maxsize):
                    elem_aw  = awf.acquire(ObjectFifoPort.Consume, 1)
                    elem_whi = whf.acquire(ObjectFifoPort.Consume, 1)
                    elem_c   = cf.acquire(ObjectFifoPort.Produce, 1)
                    gemv_core_sink(elem_aw, elem_whi, elem_c)
                    awf.release(ObjectFifoPort.Consume, 1)
                    whf.release(ObjectFifoPort.Consume, 1)
                    cf.release(ObjectFifoPort.Produce, 1)

        def make_send_core(tile_, awf, whf):
            @core(tile_)
            def _():
                for _ in range_(sys.maxsize):
                    elem_aw  = awf.acquire(ObjectFifoPort.Consume, 1)
                    elem_whi = whf.acquire(ObjectFifoPort.Consume, 1)
                    gemv_core_send(elem_aw, elem_whi)
                    awf.release(ObjectFifoPort.Consume, 1)
                    whf.release(ObjectFifoPort.Consume, 1)

        def make_pass_core(tile_, awf, whf):
            @core(tile_)
            def _():
                for _ in range_(sys.maxsize):
                    elem_aw  = awf.acquire(ObjectFifoPort.Consume, 1)
                    elem_whi = whf.acquire(ObjectFifoPort.Consume, 1)
                    gemv_core_pass(elem_aw, elem_whi)
                    awf.release(ObjectFifoPort.Consume, 1)
                    whf.release(ObjectFifoPort.Consume, 1)

        for c in range(N_COLS):
            for r_idx in range(N_ROWS):
                t   = cores[c][r_idx]
                awf = aw_lo_per_core[c][r_idx]
                whf = w_hi_per_core[c][r_idx]
                if r_idx == 0:
                    make_sink_core(t, awf, whf, c_fifos[c])
                elif r_idx == N_ROWS - 1:
                    make_send_core(t, awf, whf)
                else:
                    make_pass_core(t, awf, whf)

        # ---------------------------------------------------------------
        # Host runtime sequence.
        #   W: per column, two bursts on two parallel shim MM2S channels:
        #        AW_lo blob (135 KB) -- per-row [A_slice | W_lo_slab]
        #        W_hi  blob (128 KB) -- per-row W_hi_slab
        #   C: per column, 256-byte drain (64 INT32)
        # Layout (whole W tensor, per col): [AW_lo blob | W_hi blob].
        #
        # When iters_per_call > 1 we issue the full task batch repeatedly
        # inside a single kernel invocation -- the cores' infinite outer
        # loop already keeps them ready to consume each new batch.  This
        # amortizes the XRT host launch overhead so the benchmark
        # measures device-side steady-state throughput.
        # ---------------------------------------------------------------
        col_stride = AW_LO_COL_BYTES + W_HI_COL_BYTES

        def emit_one_batch(W, C):
            tasks = []
            for c in range(N_COLS):
                col_base = c * col_stride
                tasks.append(
                    shim_dma_single_bd_task(
                        aw_lo_shim_to_mem[c], W,
                        offset=col_base,
                        sizes=[1, 1, 1, AW_LO_COL_BYTES],
                        issue_token=True,
                    )
                )
                tasks.append(
                    shim_dma_single_bd_task(
                        w_hi_shim_to_mem[c], W,
                        offset=col_base + AW_LO_COL_BYTES,
                        sizes=[1, 1, 1, W_HI_COL_BYTES],
                        issue_token=True,
                    )
                )
                tasks.append(
                    shim_dma_single_bd_task(
                        c_fifos[c], C,
                        offset=c * N_PER_COL,
                        sizes=[1, 1, 1, N_PER_COL],
                        issue_token=True,
                    )
                )
            # Per-task dma_start_task() serializes inside the runtime
            # sequence; batching them in one call lets the shims fire
            # in parallel across both MM2S channels.
            dma_start_task(*tasks)
            dma_await_task(*tasks)

        if enable_trace:
            trace_ty = np.ndarray[(trace_size,), np.dtype[np.int32]]

            @runtime_sequence(w_tensor_ty, c_tensor_ty, trace_ty)
            def sequence(W, C, T):
                trace_start_config("core_trace")
                trace_host_config(buffer_size=trace_size, arg_idx=2)
                for _ in range(iters_per_call):
                    emit_one_batch(W, C)
        else:
            @runtime_sequence(w_tensor_ty, c_tensor_ty)
            def sequence(W, C):
                for _ in range(iters_per_call):
                    emit_one_batch(W, C)


# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="High-perf GEMV (cascade) for npu3")
parser.add_argument("-d", "--dev", default="npu3", help="AIE device (npu3 only)")
parser.add_argument("-i", "--iters", type=int, default=1,
                    help="GEMV iterations per kernel call (batch factor)")
parser.add_argument("--trace", action="store_true",
                    help="Enable event trace (emits extra trace buffer arg)")
parser.add_argument("--trace_size", type=int, default=8192,
                    help="Trace buffer size in INT32 elements")
args = parser.parse_args()

if args.dev != "npu3":
    raise ValueError(f"This example targets npu3; got {args.dev}")

with mlir_mod_ctx() as ctx:
    my_gemv(AIEDevice.npu3,
            iters_per_call=args.iters,
            enable_trace=args.trace,
            trace_size=args.trace_size)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res, file=sys.stderr)
        sys.exit(1)
