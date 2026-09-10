#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# High-performance GEMV (M=1) example with multi-row cascade reduction.
# Ported (and scaled to npu3 = 3 cols x 4 compute rows) from a
# llama2 llama_4096_512 kernel example.
#
# This example demonstrates:
#   - Spatial tiling across a configurable 1..3 cols x 4 rows (--cols)
#   - Activation broadcast from a single shim to all compute cores
#   - Per-column weight fan-out through the memtile (object_fifo_link)
#   - Configurable 2 or 4 weight streams (shim MM2S) per column (--streams)
#   - North->south cascade reduction within each column
#
# Problem: M=1, K via --ktotal (default 4096), N = cols*64 (192 for 3 cols),
# INT8 x INT8 -> INT32.
#
# NPU3 shim DMA budget = 4 MM2S + 2 S2MM per shim, so the shim cannot
# afford one weight stream per (c, r) directly.  Instead each compute
# column's shim sends its row weight slabs to its memtile over `--streams`
# MM2S channels (2 or 4), and the memtile splits each stream into the
# per-core FIFOs via object_fifo_link (the vector_add_multicol pattern).
# More streams = more parallel shim DMA channels = higher weight-load BW.

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_


# Hardware partitioning.  N_COLS is configurable (1, 2 or 3) via --cols;
# COMPUTE_ROWS is fixed at the 4 compute rows of an npu3 column.
COMPUTE_ROWS   = [2, 3, 4, 5]
N_ROWS         = len(COMPUTE_ROWS)

# Problem geometry (must match constants in kernel.cc).  TILE_K/TILE_N are
# fixed by the 64x64 staging-matrix kernel; N_TOTAL scales with N_COLS and
# the K-derived byte sizes scale with K_TOTAL -- both computed in my_gemv.
TILE_K         = 64
TILE_N         = 64
N_PER_COL      = TILE_N                     # 64

BUF_DEPTH      = 2       # shim -> memtile stream FIFO depth (memtile SRAM)
AW_LO_DEPTH    = 1       # memtile -> core AW_lo FIFO depth (core L1)
W_HI_DEPTH     = 1       # memtile -> core W_hi  FIFO depth (core L1)

# The shim BD's outermost (iteration) size field is 6 bits, so one shim
# descriptor can repeat a transfer at most 64 times.
MAX_BD_REPEAT  = 64

# Approx. per-core L1 budget for the two weight buffers (AW_lo + W_hi).
# AIE4 core local memory is 128 KB; reserve headroom for the C buffer and
# stack.  Per-core weight bytes = depth * K_per_row * (1 + TILE_N).
L1_WEIGHT_BUDGET = 128 * 1024 - 3 * 1024    # ~125 KB


def my_gemv(dev, n_cols=3, weight_streams=2, k_total=4096,
            iters_per_call=1, enable_trace=False, trace_size=8192,
            buf_depth=BUF_DEPTH, aw_lo_depth=AW_LO_DEPTH,
            w_hi_depth=W_HI_DEPTH):
    # Column count (1..3) and per-column shim weight-stream count (2 or 4).
    N_COLS  = n_cols
    N_TOTAL = N_COLS * TILE_N

    # K (contraction length) is configurable via --ktotal.  It is split
    # evenly across the N_ROWS compute rows and each row's slice is tiled
    # into TILE_K chunks, so K_TOTAL must be a multiple of N_ROWS*TILE_K.
    K_TOTAL = k_total
    if K_TOTAL % (N_ROWS * TILE_K) != 0:
        raise ValueError(
            f"K_TOTAL must be a multiple of N_ROWS*TILE_K "
            f"(={N_ROWS * TILE_K}); got {K_TOTAL}")
    K_PER_ROW       = K_TOTAL // N_ROWS
    K_TILES_PER_ROW = K_PER_ROW // TILE_K

    # Per-core data sizes.  Each core's weight slab is split along N into two
    # halves streamed via parallel shim MM2S channels: W_lo carries N cols
    # 0..31 of every K-tile, W_hi cols 32..63.  Compute tiles have only 2
    # input DMA channels and each core needs A + W_lo + W_hi (3 streams), so
    # this core's per-row A slice (K_PER_ROW bytes) is embedded at the FRONT
    # of its AW_lo buffer -- two FIFOs per core, within the 2-channel budget.
    #   AW_lo per core = [A_slice | W_lo half-slab]
    #   W_hi  per core = W_hi half-slab
    A_SLICE_BYTES        = K_PER_ROW
    W_HALF_BYTES         = K_PER_ROW * TILE_N // 2
    AW_LO_PER_CORE_BYTES = A_SLICE_BYTES + W_HALF_BYTES
    W_HI_PER_CORE_BYTES  = W_HALF_BYTES
    AW_LO_COL_BYTES      = N_ROWS * AW_LO_PER_CORE_BYTES
    W_HI_COL_BYTES       = N_ROWS * W_HI_PER_CORE_BYTES

    # L1 footprint guard (approximate; the chess build is the real check).
    # The per-core FIFO depths multiply the L1 cost, so deepening the
    # memtile->core hop trades L1 for pipelining.
    per_core_weight_bytes = (aw_lo_depth * AW_LO_PER_CORE_BYTES
                             + w_hi_depth * W_HI_PER_CORE_BYTES)
    if per_core_weight_bytes > L1_WEIGHT_BUDGET:
        raise ValueError(
            f"K_TOTAL={K_TOTAL} with AW_lo depth {aw_lo_depth} / W_hi depth "
            f"{w_hi_depth} needs {per_core_weight_bytes} B of weight "
            f"buffers per core, over the ~{L1_WEIGHT_BUDGET} B L1 budget; "
            f"reduce K_TOTAL or the per-core depths "
            f"(K <= ~7680 for the default depth-1 4-row layout)")

    # Each column's weight payload has two "kinds": AW_lo (A slice + low
    # N-half) and W_hi (high N-half).  Each kind is split across
    # streams_per_kind = weight_streams // 2 shim MM2S channels along the
    # row dimension, rows_per_stream rows per channel.  weight_streams=2 ->
    # one channel per kind (all N_ROWS rows); weight_streams=4 -> two
    # channels per kind (N_ROWS/2 rows each).  Either way the memtile fans
    # each stream out to its row group's per-core FIFOs, so each compute
    # tile still consumes exactly 2 input FIFOs.
    if weight_streams not in (2, 4):
        raise ValueError(f"weight_streams must be 2 or 4, got {weight_streams}")
    streams_per_kind = weight_streams // 2
    if N_ROWS % streams_per_kind != 0:
        raise ValueError(
            f"N_ROWS={N_ROWS} not divisible by streams_per_kind={streams_per_kind}")
    rows_per_stream    = N_ROWS // streams_per_kind
    aw_lo_stream_bytes = rows_per_stream * AW_LO_PER_CORE_BYTES
    w_hi_stream_bytes  = rows_per_stream * W_HI_PER_CORE_BYTES

    @device(dev)
    def device_body():
        # ---------------------------------------------------------------
        # Element types
        # ---------------------------------------------------------------
        aw_lo_ty        = np.ndarray[(AW_LO_PER_CORE_BYTES,), np.dtype[np.int8]]
        w_hi_ty         = np.ndarray[(W_HI_PER_CORE_BYTES,),  np.dtype[np.int8]]
        aw_lo_stream_ty = np.ndarray[(aw_lo_stream_bytes,),   np.dtype[np.int8]]
        w_hi_stream_ty  = np.ndarray[(w_hi_stream_bytes,),    np.dtype[np.int8]]
        c_slab_ty       = np.ndarray[(N_PER_COL,),            np.dtype[np.int32]]

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

        # Required Control32 packetflows per shim.  On AIE4, shim DMA
        # completion tokens for MM2S channels 2/3 return on Control32:1, so
        # 4-stream designs need both control return ports routed to the
        # controller.
        control_return_channels = 2 if weight_streams > 2 else 1
        for ctrl_ch in range(control_return_channels):
            for c in range(N_COLS):
                packetflow(
                    c,
                    shims[c],
                    WireBundle.Control32,
                    ctrl_ch,
                    [
                        {
                            "dest": shims[0],
                            "port": WireBundle.Controller32,
                            "channel": 0,
                        }
                    ],
                )

        # ---------------------------------------------------------------
        # Weights: each column's shim drives `weight_streams` MM2S channels.
        # The per-column payload has two kinds -- AW_lo (per-row
        # [A_slice 1 KB | W_lo_slab 32 KB]) and W_hi (per-row W_hi_slab
        # 32 KB) -- and each kind is split across streams_per_kind channels
        # along the row dimension (rows_per_stream rows per channel):
        #   weight_streams=2 -> 1 channel/kind carrying all N_ROWS rows
        #                       (AW_lo blob 135 KB, W_hi blob 128 KB)
        #   weight_streams=4 -> 2 channels/kind carrying N_ROWS/2 rows each
        # The memtile splits each stream into its row group's per-(c,r)
        # FIFOs via object_fifo_link with per-core offsets.  The activation
        # slice rides along inside AW_lo so each compute tile only consumes
        # 2 input FIFOs (within the 2-input-channel hard limit), regardless
        # of how many shim streams feed the memtile.
        #
        # Per-core L1 footprint: AW_lo (33 KB, depth 1) + W_hi (32 KB,
        # depth 1) + C (256 B, depth 2) + stack ~1 KB = ~66 KB.  Fits in
        # the 128-KB AIE4 core local memory.
        # ---------------------------------------------------------------
        aw_lo_per_core = [[None] * N_ROWS for _ in range(N_COLS)]
        w_hi_per_core  = [[None] * N_ROWS for _ in range(N_COLS)]
        aw_lo_streams  = [[None] * streams_per_kind for _ in range(N_COLS)]
        w_hi_streams   = [[None] * streams_per_kind for _ in range(N_COLS)]
        for c in range(N_COLS):
            # Per-core consumer FIFOs (always 2 per core: AW_lo + W_hi).
            for r_idx in range(N_ROWS):
                aw_lo_per_core[c][r_idx] = object_fifo(
                    f"AWlo_m{c}_r{r_idx}", mems[c], cores[c][r_idx],
                    aw_lo_depth, aw_lo_ty,
                )
                w_hi_per_core[c][r_idx] = object_fifo(
                    f"Whi_m{c}_r{r_idx}",  mems[c], cores[c][r_idx],
                    w_hi_depth, w_hi_ty,
                )
            # One shim->memtile FIFO per (kind, row group), each linked to
            # the rows_per_stream cores of its group with per-core offsets.
            for g in range(streams_per_kind):
                rows = range(g * rows_per_stream, (g + 1) * rows_per_stream)
                aw_lo_streams[c][g] = object_fifo(
                    f"AWlo_s{c}_g{g}", shims[c], mems[c], buf_depth,
                    aw_lo_stream_ty,
                )
                w_hi_streams[c][g] = object_fifo(
                    f"Whi_s{c}_g{g}",  shims[c], mems[c], buf_depth,
                    w_hi_stream_ty,
                )
                object_fifo_link(
                    aw_lo_streams[c][g],
                    [aw_lo_per_core[c][r] for r in rows],
                    [],
                    [i * AW_LO_PER_CORE_BYTES for i in range(rows_per_stream)],
                )
                object_fifo_link(
                    w_hi_streams[c][g],
                    [w_hi_per_core[c][r] for r in rows],
                    [],
                    [i * W_HI_PER_CORE_BYTES for i in range(rows_per_stream)],
                )

        # ---------------------------------------------------------------
        # Output: one S2MM per column from the cascade root (row index 0
        # = hw row 2) directly to its column's shim.  64 INT32 = 256 B.
        # ---------------------------------------------------------------
        c_fifos = [
            object_fifo(f"C_c{c}", cores[c][0], shims[c], buf_depth, c_slab_ty)
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
        # When iters_per_call > 1 the batch is issued ONCE with the shim
        # BD's hardware repeat_count set to iters_per_call, not as an
        # unrolled Python loop of start/await pairs.
        #
        # PERF (measured on npu3, K=4096/COLS=3/STREAMS=2): the unrolled
        # form put a dma_start_task + dma_await_task barrier between every
        # iteration, which fully drains the shim->memtile->core->cascade
        # pipeline each time.  A two-point fit over K=2048 and K=4096
        # showed ~7.5 us/iter of iteration-invariant cost against only
        # ~7.5 ns/KB of marginal transfer cost (~133 GB/s marginal), i.e.
        # over half of each 13.5 us iteration was barrier, not data.  That
        # is why deepening the objectfifos does nothing here: with a
        # barrier every iteration there is nothing to prefetch across.
        # Folding the repeat into a single descriptor per channel lets the
        # shim stream continuously, back-pressured only by the consuming
        # objectfifo's locks (the same pattern shim_dma_memtile_bw uses to
        # reach its ~110-115 GB/s).
        # ---------------------------------------------------------------
        col_stride = AW_LO_COL_BYTES + W_HI_COL_BYTES

        def emit_one_batch(W, C, reps=1):
            # `reps` becomes the shim BD's hardware repeat_count (sizes[0]
            # with a zero outer stride): ONE descriptor per channel that
            # re-issues the same transfer `reps` times back to back.  The
            # consuming objectfifo sees `reps` objects and back-pressures
            # the shim through its own locks, so the shim streams
            # continuously instead of being drained and restarted per
            # iteration.  See the perf note above emit_one_batch's caller.
            tasks = []
            for c in range(N_COLS):
                col_base = c * col_stride
                # AW_lo streams: contiguous row groups at the front of the
                # column blob.
                for g in range(streams_per_kind):
                    tasks.append(
                        shim_dma_single_bd_task(
                            aw_lo_streams[c][g], W,
                            offset=col_base + g * aw_lo_stream_bytes,
                            sizes=[reps, 1, 1, aw_lo_stream_bytes],
                            strides=[0, 0, 0, 1],
                            issue_token=True,
                        )
                    )
                # W_hi streams: contiguous row groups immediately after the
                # AW_lo blob in the column blob.
                hi_base = col_base + AW_LO_COL_BYTES
                for g in range(streams_per_kind):
                    tasks.append(
                        shim_dma_single_bd_task(
                            w_hi_streams[c][g], W,
                            offset=hi_base + g * w_hi_stream_bytes,
                            sizes=[reps, 1, 1, w_hi_stream_bytes],
                            strides=[0, 0, 0, 1],
                            issue_token=True,
                        )
                    )
                tasks.append(
                    shim_dma_single_bd_task(
                        c_fifos[c], C,
                        offset=c * N_PER_COL,
                        sizes=[reps, 1, 1, N_PER_COL],
                        strides=[0, 0, 0, 1],
                        issue_token=True,
                    )
                )
            # Per-task dma_start_task() serializes inside the runtime
            # sequence; batching them in one call lets the shims fire
            # in parallel across both MM2S channels.
            dma_start_task(*tasks)
            dma_await_task(*tasks)

        def emit_iters(W, C, total):
            # The shim BD's outermost (iteration) size field is 6 bits, so a
            # single descriptor can repeat at most MAX_BD_REPEAT times.  Split
            # the requested iteration count into as few chunks as possible;
            # only the chunk boundaries pay the start/await pipeline drain.
            remaining = total
            while remaining > 0:
                reps = min(remaining, MAX_BD_REPEAT)
                emit_one_batch(W, C, reps)
                remaining -= reps

        if enable_trace:
            trace_ty = np.ndarray[(trace_size,), np.dtype[np.int32]]

            @runtime_sequence(w_tensor_ty, c_tensor_ty, trace_ty)
            def sequence(W, C, T):
                trace_start_config("core_trace")
                trace_host_config(buffer_size=trace_size, arg_idx=2)
                emit_iters(W, C, iters_per_call)
        else:
            @runtime_sequence(w_tensor_ty, c_tensor_ty)
            def sequence(W, C):
                emit_iters(W, C, iters_per_call)


# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="High-perf GEMV (cascade) for npu3")
parser.add_argument("-d", "--dev", default="npu3", help="AIE device (npu3 only)")
parser.add_argument("-c", "--cols", type=int, default=3, choices=(1, 2, 3),
                    help="Number of compute columns (N = cols * 64)")
parser.add_argument("-s", "--streams", type=int, default=2, choices=(2, 4),
                    help="Per-column shim weight-stream (MM2S) channels")
parser.add_argument("-k", "--ktotal", type=int, default=4096,
                    help="Contraction length K (multiple of N_ROWS*TILE_K=256; "
                         "<= ~7680 for the default layout)")
parser.add_argument("-i", "--iters", type=int, default=1,
                    help="GEMV iterations per kernel call (batch factor)")
parser.add_argument("--bufdepth", type=int, default=BUF_DEPTH,
                    help="shim->memtile weight stream FIFO depth (memtile SRAM)")
parser.add_argument("--awdepth", type=int, default=AW_LO_DEPTH,
                    help="memtile->core AW_lo FIFO depth (core L1)")
parser.add_argument("--whidepth", type=int, default=W_HI_DEPTH,
                    help="memtile->core W_hi FIFO depth (core L1)")
parser.add_argument("--trace", action="store_true",
                    help="Enable event trace (emits extra trace buffer arg)")
parser.add_argument("--trace_size", type=int, default=8192,
                    help="Trace buffer size in INT32 elements")
args = parser.parse_args()

if args.dev != "npu3":
    raise ValueError(f"This example targets npu3; got {args.dev}")

with mlir_mod_ctx() as ctx:
    my_gemv(AIEDevice.npu3,
            n_cols=args.cols,
            weight_streams=args.streams,
            k_total=args.ktotal,
            iters_per_call=args.iters,
            enable_trace=args.trace,
            trace_size=args.trace_size,
            buf_depth=args.bufdepth,
            aw_lo_depth=args.awdepth,
            w_hi_depth=args.whidepth)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res, file=sys.stderr)
        sys.exit(1)
