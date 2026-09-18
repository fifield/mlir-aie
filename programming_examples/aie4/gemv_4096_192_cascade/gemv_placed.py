#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# INT8 GEMV with configurable geometry and north-to-south cascade reduction.
# See README.md for configuration, host layout, and routing constraints.

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Hardware partitioning.  ALL_COMPUTE_ROWS lists the 4 compute rows of an
# npu3 column, row2 (index 0, adjacent to the memtile/shim) first; --rows
# selects how many of these (starting from row2) are active.
ALL_COMPUTE_ROWS = [2, 3, 4, 5]
MAX_ROWS = len(ALL_COMPUTE_ROWS)

# Problem geometry (must match constants in kernel.cc).  TILE_K/TILE_N are
# fixed by the 64x64 staging-matrix kernel; N_TOTAL scales with N_COLS and
# the K-derived byte sizes scale with K_TOTAL -- both computed in my_gemv.
TILE_K = 64
TILE_N = 64
N_PER_COL = TILE_N  # 64

BUF_DEPTH = 2  # shim -> memtile stream FIFO depth (memtile SRAM)
AW_LO_DEPTH = 1  # memtile -> core AW_lo FIFO depth (core L1)
W_HI_DEPTH = 1  # memtile -> core W_hi  FIFO depth (core L1)

# The shim BD's outermost (iteration) size field is 6 bits, so one shim
# descriptor can repeat a transfer at most 64 times.
MAX_BD_REPEAT = 64

# Approx. per-core L1 budget for the two weight buffers (AW_lo + W_hi).
# AIE4 core local memory is 128 KB; reserve headroom for the C buffer and
# stack. Residency depends on the chunk sizes and FIFO depths; see the
# per_core_weight_bytes check below.
L1_WEIGHT_BUDGET = 128 * 1024 - 3 * 1024  # ~125 KB


def my_gemv(
    dev,
    n_total=192,
    n_rows=MAX_ROWS,
    weight_streams=2,
    k_total=4096,
    iters_per_call=1,
    enable_trace=False,
    trace_size=8192,
    buf_depth=BUF_DEPTH,
    aw_lo_depth=AW_LO_DEPTH,
    w_hi_depth=W_HI_DEPTH,
    k_chunk_tiles=0,
):
    # Pad device geometry to TILE_N; the host checks only the requested lanes.
    if n_total < 1:
        raise ValueError(f"--n must be >= 1, got {n_total}")
    N_COLS = -(-n_total // TILE_N)  # ceil(n_total / TILE_N)
    if N_COLS > 3:
        raise ValueError(
            f"--n {n_total} needs {N_COLS} columns (64 each); npu3 caps at "
            f"3 columns (N <= 192)"
        )
    N_TOTAL = N_COLS * TILE_N

    # Rows: --rows (1..4) selects how many of the 4 npu3 compute rows are
    # active, starting from row2 (the sink, closest to the memtile/shim).
    if not (1 <= n_rows <= MAX_ROWS):
        raise ValueError(f"--rows must be 1..{MAX_ROWS}, got {n_rows}")
    COMPUTE_ROWS = ALL_COMPUTE_ROWS[:n_rows]
    N_ROWS = n_rows

    # K (contraction length) is configurable via --ktotal.  It is split
    # evenly across the N_ROWS compute rows and each row's slice is tiled
    # into TILE_K chunks, so K_TOTAL must be a multiple of N_ROWS*TILE_K.
    K_TOTAL = k_total
    if K_TOTAL % (N_ROWS * TILE_K) != 0:
        raise ValueError(
            f"K_TOTAL must be a multiple of N_ROWS*TILE_K "
            f"(={N_ROWS * TILE_K}); got {K_TOTAL}"
        )
    K_PER_ROW = K_TOTAL // N_ROWS
    K_TILES_PER_ROW = K_PER_ROW // TILE_K

    # A chunk is one memtile->core FIFO item; zero selects a whole row.
    K_CHUNK_TILES = K_TILES_PER_ROW if k_chunk_tiles == 0 else k_chunk_tiles
    if K_TILES_PER_ROW % K_CHUNK_TILES != 0:
        raise ValueError(
            f"--kchunk must divide K_TILES_PER_ROW (={K_TILES_PER_ROW}); "
            f"got {K_CHUNK_TILES}"
        )
    if K_CHUNK_TILES == 1 and K_TILES_PER_ROW > 1:
        raise ValueError(
            "--kchunk 1 (one K-tile per chunk) produces incorrect output "
            "on npu3 (a per-chunk indexing bug, not yet root-caused -- see "
            "README 'K-chunking' section); use 2 or more (or 0 for no "
            "chunking)"
        )
    NUM_CHUNKS = K_TILES_PER_ROW // K_CHUNK_TILES

    # AW_lo embeds A to keep each core within its two input DMA channels.
    # See test.cpp's pack_a_byte/pack_weight_byte for the chunked host layout.
    A_SLICE_BYTES = K_PER_ROW
    W_HALF_BYTES = K_PER_ROW * TILE_N // 2
    AW_LO_PER_CORE_BYTES = A_SLICE_BYTES + W_HALF_BYTES
    W_HI_PER_CORE_BYTES = W_HALF_BYTES
    AW_LO_COL_BYTES = N_ROWS * AW_LO_PER_CORE_BYTES
    W_HI_COL_BYTES = N_ROWS * W_HI_PER_CORE_BYTES

    CHUNK_A_BYTES = K_CHUNK_TILES * TILE_K
    CHUNK_WLO_BYTES = K_CHUNK_TILES * TILE_K * TILE_N // 2
    AW_LO_CHUNK_BYTES = CHUNK_A_BYTES + CHUNK_WLO_BYTES
    W_HI_CHUNK_BYTES = CHUNK_WLO_BYTES
    assert AW_LO_CHUNK_BYTES * NUM_CHUNKS == AW_LO_PER_CORE_BYTES
    assert W_HI_CHUNK_BYTES * NUM_CHUNKS == W_HI_PER_CORE_BYTES

    # With --awdepth/--whidepth > 1, ring slot k of a chunk-sized object FIFO
    # starts at k*AW_LO_CHUNK_BYTES (resp. W_HI_CHUNK_BYTES); the kernel
    # loads 128-byte v128int8 vectors from these buffers, so a chunk size
    # that isn't a 128-byte multiple would silently misalign every
    # odd-numbered ring slot's vector loads.  W_HI_CHUNK_BYTES is always a
    # multiple of 128 (K_CHUNK_TILES*2048); only AW_LO_CHUNK_BYTES (which
    # also carries the 64B/tile A slice) can fail this.
    if aw_lo_depth > 1 and AW_LO_CHUNK_BYTES % 128 != 0:
        raise ValueError(
            f"--kchunk={K_CHUNK_TILES} gives a {AW_LO_CHUNK_BYTES} B AW_lo "
            f"chunk, not a multiple of 128; --awdepth > 1 would misalign "
            f"v128int8 loads on odd ring slots. Use an even --kchunk "
            f"with --awdepth > 1, or --awdepth 1"
        )

    # Only `depth` chunks per FIFO are resident in L1. This estimate leaves
    # headroom for other allocations; the kernel build checks the full layout.
    per_core_weight_bytes = (
        aw_lo_depth * AW_LO_CHUNK_BYTES + w_hi_depth * W_HI_CHUNK_BYTES
    )
    if per_core_weight_bytes > L1_WEIGHT_BUDGET:
        raise ValueError(
            f"K_TOTAL={K_TOTAL} with AW_lo depth {aw_lo_depth} / W_hi depth "
            f"{w_hi_depth} and --kchunk={K_CHUNK_TILES} needs "
            f"{per_core_weight_bytes} B of weight buffers per core, over "
            f"the ~{L1_WEIGHT_BUDGET} B L1 budget; reduce K_TOTAL, the "
            f"per-core depths, or increase chunking (lower --kchunk) "
            f"(K <= ~7680 for the default depth-1 4-row unchunked layout)"
        )

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
            f"--streams {weight_streams} requires --rows to be divisible by "
            f"{streams_per_kind}; got --rows {N_ROWS}"
        )
    rows_per_stream = N_ROWS // streams_per_kind
    # Group spans locate host row groups; stream sizes describe one chunk
    # across a group. See emit_one_batch for the descriptor traversal.
    aw_lo_group_span = rows_per_stream * AW_LO_PER_CORE_BYTES
    w_hi_group_span = rows_per_stream * W_HI_PER_CORE_BYTES
    aw_lo_stream_bytes = rows_per_stream * AW_LO_CHUNK_BYTES
    w_hi_stream_bytes = rows_per_stream * W_HI_CHUNK_BYTES

    @device(dev)
    def device_body():
        # ---------------------------------------------------------------
        # Element types
        # ---------------------------------------------------------------
        aw_lo_chunk_ty = np.ndarray[(AW_LO_CHUNK_BYTES,), np.dtype[np.int8]]
        w_hi_chunk_ty = np.ndarray[(W_HI_CHUNK_BYTES,), np.dtype[np.int8]]
        aw_lo_stream_ty = np.ndarray[(aw_lo_stream_bytes,), np.dtype[np.int8]]
        w_hi_stream_ty = np.ndarray[(w_hi_stream_bytes,), np.dtype[np.int8]]
        c_slab_ty = np.ndarray[(N_PER_COL,), np.dtype[np.int32]]
        partial_ty = np.ndarray[(N_PER_COL,), np.dtype[np.int32]]

        # W concatenates per-column [AW_lo_blob | W_hi_blob], with A embedded.
        W_TENSOR_BYTES = N_COLS * (AW_LO_COL_BYTES + W_HI_COL_BYTES)
        w_tensor_ty = np.ndarray[(W_TENSOR_BYTES,), np.dtype[np.int8]]
        c_tensor_ty = np.ndarray[(N_TOTAL,), np.dtype[np.int32]]

        # ---------------------------------------------------------------
        # External kernels.  gemv_chunk_accumulate is called once per chunk
        # (num_chunks times per core per iteration); gemv_finish_* is
        # called once per core per iteration after all chunks land.
        # ---------------------------------------------------------------
        gemv_zero_partial = external_func(
            "gemv_zero_partial",
            inputs=[partial_ty],
            link_with="project/kernel.o",
        )
        gemv_chunk_accumulate = external_func(
            "gemv_chunk_accumulate",
            inputs=[aw_lo_chunk_ty, w_hi_chunk_ty, partial_ty],
            link_with="project/kernel.o",
        )
        gemv_finish_send = external_func(
            "gemv_finish_send",
            inputs=[partial_ty],
            link_with="project/kernel.o",
        )
        gemv_finish_pass = external_func(
            "gemv_finish_pass",
            inputs=[partial_ty],
            link_with="project/kernel.o",
        )
        gemv_finish_sink = external_func(
            "gemv_finish_sink",
            inputs=[partial_ty, c_slab_ty],
            link_with="project/kernel.o",
        )
        gemv_finish_solo = external_func(
            "gemv_finish_solo",
            inputs=[partial_ty, c_slab_ty],
            link_with="project/kernel.o",
        )

        # ---------------------------------------------------------------
        # Tiles
        # ---------------------------------------------------------------
        shims = [tile(c, 0) for c in range(N_COLS)]
        for c, sh in enumerate(shims):
            sh.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, c])
        mems = [tile(c, 1) for c in range(N_COLS)]
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
        # Distribute each shim stream's chunk across its row group's cores.
        # Keep one FIFO per core and payload kind; allocating a FIFO for
        # every chunk would exceed the core's input-DMA-channel budget.
        # ---------------------------------------------------------------
        aw_lo_per_core = [[None] * N_ROWS for _ in range(N_COLS)]
        w_hi_per_core = [[None] * N_ROWS for _ in range(N_COLS)]
        partial_bufs = [[None] * N_ROWS for _ in range(N_COLS)]
        aw_lo_streams = [[None] * streams_per_kind for _ in range(N_COLS)]
        w_hi_streams = [[None] * streams_per_kind for _ in range(N_COLS)]
        for c in range(N_COLS):
            # One consumer FIFO per core per kind (chunk-sized item).
            for r_idx in range(N_ROWS):
                aw_lo_per_core[c][r_idx] = object_fifo(
                    f"AWlo_m{c}_r{r_idx}",
                    mems[c],
                    cores[c][r_idx],
                    aw_lo_depth,
                    aw_lo_chunk_ty,
                )
                w_hi_per_core[c][r_idx] = object_fifo(
                    f"Whi_m{c}_r{r_idx}",
                    mems[c],
                    cores[c][r_idx],
                    w_hi_depth,
                    w_hi_chunk_ty,
                )
                partial_bufs[c][r_idx] = buffer(
                    cores[c][r_idx],
                    partial_ty,
                    f"partial_c{c}_r{r_idx}",
                )
            # Link one group-sized stream item to the per-core chunk items.
            for g in range(streams_per_kind):
                rows = range(g * rows_per_stream, (g + 1) * rows_per_stream)
                aw_lo_streams[c][g] = object_fifo(
                    f"AWlo_s{c}_g{g}",
                    shims[c],
                    mems[c],
                    buf_depth,
                    aw_lo_stream_ty,
                )
                w_hi_streams[c][g] = object_fifo(
                    f"Whi_s{c}_g{g}",
                    shims[c],
                    mems[c],
                    buf_depth,
                    w_hi_stream_ty,
                )
                object_fifo_link(
                    aw_lo_streams[c][g],
                    [aw_lo_per_core[c][r] for r in rows],
                    [],
                    [i * AW_LO_CHUNK_BYTES for i in range(rows_per_stream)],
                )
                object_fifo_link(
                    w_hi_streams[c][g],
                    [w_hi_per_core[c][r] for r in rows],
                    [],
                    [i * W_HI_CHUNK_BYTES for i in range(rows_per_stream)],
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
        # Cascade flows: N->S within each column.  With N_ROWS==1 there is
        # no cascade at all (the single core is both root and sink).
        # ---------------------------------------------------------------
        for c in range(N_COLS):
            for r_idx in range(N_ROWS - 1, 0, -1):
                cascade_flow(cores[c][r_idx], cores[c][r_idx - 1])

        # ---------------------------------------------------------------
        # Optional trace setup.  Capture compute/memory events on all
        # active column-0 cascade cores, so the row-to-row cascade handoff
        # can be seen end to end.  Each core gets its own trace unit +
        # distinct packet id; all share the same start/stop broadcast
        # pulse so they're aligned on a common timeline.  All trace packets
        # route to shim col 0 S2MM channel 1, draining into the host trace
        # buffer.
        # ---------------------------------------------------------------
        trace_config_names = []
        if enable_trace:
            from aie.dialects.aie import (
                trace,
                trace_mode,
                trace_packet,
                trace_event,
                trace_start,
                trace_stop,
                TraceMode,
                TracePacketType,
            )

            def _make_core_trace_body(hw_row):
                def _():
                    trace_mode(TraceMode.EventTime)
                    # Offset packet ids away from 0..2, which the
                    # Control32 return packetflow above already uses for
                    # columns 0..2 (weight_streams>2 needs both ids 0/1 on
                    # every column) -- colliding ids on the col-0 shim
                    # switchbox would garble trace packets and/or stall
                    # control-token returns.
                    trace_packet(8 + (hw_row - 2), TracePacketType.Core)
                    trace_event("INSTR_EVENT_0")
                    trace_event("INSTR_MATRIX")
                    trace_event("MEMORY_STALL")
                    trace_event("STREAM_STALL")
                    trace_event("LOCK_STALL")
                    trace_start(broadcast=3)
                    trace_stop(broadcast=4)

                return _

            for r_idx in range(N_ROWS):
                hw_row = COMPUTE_ROWS[r_idx]
                cfg_name = f"core_trace_r{hw_row}"
                trace_config_names.append(cfg_name)
                trace(cores[0][r_idx], cfg_name)(_make_core_trace_body(hw_row))

            # Note: the trace lowering pass auto-generates the
            # Trace32 -> shim DMA packetflow from each trace.config op
            # above, so we do NOT declare an explicit packetflow here.
            # Declaring it manually causes a duplicate packet_flow in the
            # lowered MLIR which deadlocks the run on npu3.

        # ---------------------------------------------------------------
        # Per-core bodies.
        # @core(...) runs the body immediately, so per-iteration FIFO
        # references are captured at decoration time.  Use factory helpers
        # so each call to @core sees its own closure cleanly.  Every role
        # loops over NUM_CHUNKS acquiring/releasing one chunk pair at a
        # time and accumulating into that core's persistent `partial`
        # buffer, then calls the appropriate finish/cascade entry once the
        # whole row has landed.
        # ---------------------------------------------------------------
        def make_role_core(tile_, awf, whf, partial_buf, finish):
            @core(tile_)
            def _():
                for _ in range_(sys.maxsize):
                    gemv_zero_partial(partial_buf)
                    for _ in range_(NUM_CHUNKS):
                        elem_aw = awf.acquire(ObjectFifoPort.Consume, 1)
                        elem_whi = whf.acquire(ObjectFifoPort.Consume, 1)
                        gemv_chunk_accumulate(elem_aw, elem_whi, partial_buf)
                        awf.release(ObjectFifoPort.Consume, 1)
                        whf.release(ObjectFifoPort.Consume, 1)
                    finish(partial_buf)

        def make_sink_or_solo_core(tile_, awf, whf, partial_buf, cf, finish_kernel):
            @core(tile_)
            def _():
                for _ in range_(sys.maxsize):
                    gemv_zero_partial(partial_buf)
                    for _ in range_(NUM_CHUNKS):
                        elem_aw = awf.acquire(ObjectFifoPort.Consume, 1)
                        elem_whi = whf.acquire(ObjectFifoPort.Consume, 1)
                        gemv_chunk_accumulate(elem_aw, elem_whi, partial_buf)
                        awf.release(ObjectFifoPort.Consume, 1)
                        whf.release(ObjectFifoPort.Consume, 1)
                    elem_c = cf.acquire(ObjectFifoPort.Produce, 1)
                    finish_kernel(partial_buf, elem_c)
                    cf.release(ObjectFifoPort.Produce, 1)

        for c in range(N_COLS):
            for r_idx in range(N_ROWS):
                t = cores[c][r_idx]
                awf = aw_lo_per_core[c][r_idx]
                whf = w_hi_per_core[c][r_idx]
                pbuf = partial_bufs[c][r_idx]
                if N_ROWS == 1:
                    make_sink_or_solo_core(
                        t,
                        awf,
                        whf,
                        pbuf,
                        c_fifos[c],
                        gemv_finish_solo,
                    )
                elif r_idx == 0:
                    make_sink_or_solo_core(
                        t,
                        awf,
                        whf,
                        pbuf,
                        c_fifos[c],
                        gemv_finish_sink,
                    )
                elif r_idx == N_ROWS - 1:
                    make_role_core(t, awf, whf, pbuf, gemv_finish_send)
                else:
                    make_role_core(t, awf, whf, pbuf, gemv_finish_pass)

        # ---------------------------------------------------------------
        # Host runtime sequence; see emit_one_batch for DMA traversal.
        # ---------------------------------------------------------------
        col_stride = AW_LO_COL_BYTES + W_HI_COL_BYTES

        def emit_one_batch(W, C, reps=1):
            # Traverse [repeat, chunk, row, byte]. The zero repeat stride
            # reuses host data without a start/await barrier per iteration.
            # Chunk and row strides gather one group-sized FIFO item,
            # producing reps * NUM_CHUNKS items per stream. FIFO locks
            # provide backpressure while the descriptor runs.
            tasks = []
            for c in range(N_COLS):
                col_base = c * col_stride
                # AW_lo streams: contiguous row groups at the front of the
                # column blob.
                for g in range(streams_per_kind):
                    tasks.append(
                        shim_dma_single_bd_task(
                            aw_lo_streams[c][g],
                            W,
                            offset=col_base + g * aw_lo_group_span,
                            sizes=[
                                reps,
                                NUM_CHUNKS,
                                rows_per_stream,
                                AW_LO_CHUNK_BYTES,
                            ],
                            strides=[0, AW_LO_CHUNK_BYTES, AW_LO_PER_CORE_BYTES, 1],
                            issue_token=True,
                        )
                    )
                # W_hi streams: contiguous row groups immediately after the
                # AW_lo blob in the column blob.
                hi_base = col_base + AW_LO_COL_BYTES
                for g in range(streams_per_kind):
                    tasks.append(
                        shim_dma_single_bd_task(
                            w_hi_streams[c][g],
                            W,
                            offset=hi_base + g * w_hi_group_span,
                            sizes=[reps, NUM_CHUNKS, rows_per_stream, W_HI_CHUNK_BYTES],
                            strides=[0, W_HI_CHUNK_BYTES, W_HI_PER_CORE_BYTES, 1],
                            issue_token=True,
                        )
                    )
                tasks.append(
                    shim_dma_single_bd_task(
                        c_fifos[c],
                        C,
                        offset=c * N_PER_COL,
                        sizes=[reps, 1, 1, N_PER_COL],
                        strides=[0, 0, 0, 1],
                        issue_token=True,
                    )
                )
            # Start all channels before awaiting completion.
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
            # The trace buffer argument is appended to the runtime sequence
            # by trace_host_config lowering, so it is not declared here.
            @runtime_sequence(w_tensor_ty, c_tensor_ty)
            def sequence(W, C):
                for cfg_name in trace_config_names:
                    trace_start_config(cfg_name)
                # trace_host_config takes a byte count; --trace_size (and
                # test.cpp's matching trace_elems argv) are in INT32
                # elements, so convert here to keep one consistent
                # "elements" unit at the CLI/host boundary.
                trace_host_config(trace_size * 4)
                emit_iters(W, C, iters_per_call)

        else:

            @runtime_sequence(w_tensor_ty, c_tensor_ty)
            def sequence(W, C):
                emit_iters(W, C, iters_per_call)


# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="High-perf GEMV (cascade) for npu3")
parser.add_argument("-d", "--dev", default="npu3", help="AIE device (npu3 only)")
parser.add_argument(
    "-n",
    "--ntotal",
    type=int,
    default=192,
    help="Total N (output width); padded up to a multiple "
    "of 64. npu3 caps at N=192 (3 columns)",
)
parser.add_argument(
    "-r",
    "--rows",
    type=int,
    default=MAX_ROWS,
    choices=tuple(range(1, MAX_ROWS + 1)),
    help="Active cascade rows per column (1..4); rows=1 " "skips the cascade entirely",
)
parser.add_argument(
    "-s",
    "--streams",
    type=int,
    default=2,
    choices=(2, 4),
    help="Per-column shim weight-stream (MM2S) channels; "
    "4 streams requires an even --rows value",
)
parser.add_argument(
    "-k",
    "--ktotal",
    type=int,
    default=4096,
    help="Contraction length K (multiple of ROWS*TILE_K; "
    "<= ~7680 for the default layout)",
)
parser.add_argument(
    "-i",
    "--iters",
    type=int,
    default=1,
    help="GEMV iterations per kernel call (batch factor)",
)
parser.add_argument(
    "--kchunk",
    type=int,
    default=0,
    help="K-tiles per chunk (must divide K_TILES_PER_ROW); "
    "0 = whole row (no chunking, default)",
)
parser.add_argument(
    "--bufdepth",
    type=int,
    default=BUF_DEPTH,
    help="shim->memtile weight stream FIFO depth (memtile SRAM)",
)
parser.add_argument(
    "--awdepth",
    type=int,
    default=AW_LO_DEPTH,
    help="memtile->core AW_lo FIFO depth (core L1, per chunk)",
)
parser.add_argument(
    "--whidepth",
    type=int,
    default=W_HI_DEPTH,
    help="memtile->core W_hi FIFO depth (core L1, per chunk)",
)
parser.add_argument(
    "--trace",
    action="store_true",
    help="Enable event trace (emits extra trace buffer arg)",
)
parser.add_argument(
    "--trace_size", type=int, default=8192, help="Trace buffer size in INT32 elements"
)
args = parser.parse_args()

if args.dev != "npu3":
    raise ValueError(f"This example targets npu3; got {args.dev}")

with mlir_mod_ctx() as ctx:
    my_gemv(
        AIEDevice.npu3,
        n_total=args.ntotal,
        n_rows=args.rows,
        weight_streams=args.streams,
        k_total=args.ktotal,
        iters_per_call=args.iters,
        enable_trace=args.trace,
        trace_size=args.trace_size,
        buf_depth=args.bufdepth,
        aw_lo_depth=args.awdepth,
        w_hi_depth=args.whidepth,
        k_chunk_tiles=args.kchunk,
    )
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res, file=sys.stderr)
        sys.exit(1)
