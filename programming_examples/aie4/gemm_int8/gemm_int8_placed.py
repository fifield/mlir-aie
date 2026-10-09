# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate npu3 int8 GEMM: C = A @ W.T, with int8 or int32 output.

A stays in L1; W streams from the memtile into MMAC staging. See README.md
for data layouts, configuration constraints and usage.
"""

import argparse
import contextlib
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

N_COLS = 3  # active columns, overridden by --cols
N_ROWS = 4
TILE = 64  # MMAC tile edge (int8: 1x64 . 64x64)
W_STREAM_PORT = 1  # must match W_STREAM in kernel.cc
MT_BYTES = 3 << 20  # memtile data memory
MT_MM2S_W = 6  # memtile MM2S: W replay (hand-written)
MT_MM2S_C = 7  # memtile MM2S: C drain to shim S2MM 0 (c_reasm)
MT_S2MM_W = 7  # memtile S2MM: W fill (hand-written)
MT_S2MM_C = [2, 3, 4, 5]  # memtile S2MM per core row: C in (c_reasm)


def build(
    M,
    N,
    K,
    out_i8,
    iters,
    a_bcast,
    ct,
    wbuf,
    trace_size=0,
    a_split=2,
    c_reasm=1,
    c_run=1024,
    kernel="extern",
    shift=12,
):
    assert K % TILE == 0 and K >= 2 * TILE, "K must be a multiple of 64, >= 128"
    assert M % (N_ROWS * TILE) == 0, "M must be a multiple of 256"
    assert N % (N_COLS * TILE) == 0, "N must be a multiple of 192"
    MB = M // (N_ROWS * TILE)  # M-blocks
    assert kernel in ("extern", "aievec")
    # aievec: a C objectfifo acquire inside the group loop would make
    # objectfifo lowering unroll the loop and copy the tile body.
    assert kernel == "extern" or c_reasm, "--kernel aievec requires --c-reasm 1"
    if wbuf > 1 and 2 * MB > 16:
        wbuf = 1  # replay chain would exceed the memtile channel's 16 BDs
    NT = N // (N_COLS * TILE)  # output tiles per core per M-block
    assert NT % ct == 0, "N / (cols*64) must be a multiple of CT"
    NG = NT // ct  # C buffers per core per M-block (CT tiles each)
    out_bytes = 1 if out_i8 else 4
    # FIFO channels must stay below the reserved W replay and C drain channels.
    a_rows_per_src = -(-N_ROWS // N_COLS) if a_bcast else N_ROWS
    fifo_mm2s = a_rows_per_src * a_split + (0 if c_reasm else 2)
    assert fifo_mm2s <= MT_MM2S_W, (
        f"needs {fifo_mm2s} object-FIFO memtile MM2S channels below the "
        f"hand-written W replay channel {MT_MM2S_W}: use --a-split 1"
        + ("" if a_bcast or N_COLS == 1 else " or --a-bcast 1")
    )

    # RW divides the column strip into whole core-buffer rows, up to c_run
    # bytes (or one core-buffer row if that is wider).
    grp_b = ct * TILE * out_bytes  # one core buffer row, bytes
    S = NT * TILE * out_bytes  # one column strip row, bytes
    RW = max(w for w in range(grp_b, max(grp_b, c_run) + 1, grp_b) if S % w == 0)
    SB = S // RW  # sub-blocks per M-block
    NGS = RW // grp_b  # core buffers per sub-block

    # Memtile capacity (the column sourcing the most A row-blocks).
    a_mt_b = 2 * a_rows_per_src * TILE * K  # A_mt, depth 2
    w_b = wbuf * NT * TILE * K  # W strip buffers
    c_b = 2 * N_ROWS * TILE * (RW if c_reasm else grp_b)  # C ping/pong
    mt_b = a_mt_b + w_b + c_b
    assert mt_b <= MT_BYTES, (
        f"memtile needs {mt_b // 1024} KiB (A {a_mt_b // 1024} + W "
        f"{w_b // 1024} + C {c_b // 1024}) of {MT_BYTES // 1024}: use "
        "--wbuf 1, a smaller N, or a smaller --c-run"
    )

    # All host buffers and FIFO element types are i32 words; the kernel
    # reinterprets them as int8 (or int32 for C).
    a_core_w = TILE * K // 4  # one core's A slice (64 x K int8)
    w_strip_w = NT * TILE * K // 4  # one column's W strip
    c_core_w = TILE * ct * TILE * out_bytes // 4  # 64 rows x CT tiles

    @device(AIEDevice.npu3)
    def device_body():
        a_part_rows = TILE // a_split  # A slice rows per core DMA channel
        a_part_ty = np.ndarray[(a_core_w // a_split,), np.dtype[np.int32]]
        w_strip_ty = np.ndarray[(w_strip_w,), np.dtype[np.int32]]
        c_core_ty = np.ndarray[(c_core_w,), np.dtype[np.int32]]
        a_host_ty = np.ndarray[(M * K // 4,), np.dtype[np.int32]]
        w_host_ty = np.ndarray[(N * K // 4,), np.dtype[np.int32]]
        c_host_ty = np.ndarray[(M * N * out_bytes // 4,), np.dtype[np.int32]]

        if kernel == "extern":
            gemm_stage_b = external_func(
                "gemm_stage_b", inputs=[], link_with="project/kernel.o"
            )
            gemm_tile = external_func(
                "gemm_tile",
                inputs=[a_part_ty, a_part_ty, c_core_ty, np.int32],
                link_with="project/kernel.o",
            )
            gemm_tile_last = external_func(
                "gemm_tile_last",
                inputs=[a_part_ty, a_part_ty, c_core_ty, np.int32],
                link_with="project/kernel.o",
            )

        shims = [tile(c, 0) for c in range(N_COLS)]
        mems = [tile(c, 1) for c in range(N_COLS)]
        cores = [[tile(c, 2 + r) for r in range(N_ROWS)] for c in range(N_COLS)]

        # All DMA completion tokens return to the sequence controller in
        # column 0. The upper shim BD pool uses Control32 channel 1.
        for c in range(N_COLS):
            shims[c].attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")(
                [0, c]
            )
        for ctrl_ch in range(2):
            for c in range(N_COLS):
                packetflow(
                    c,
                    shims[c],
                    WireBundle.Control32,
                    ctrl_ch,
                    [{"dest": shims[0], "port": WireBundle.Controller32, "channel": 0}],
                )

        def w_chains(bufs, empty, full):
            # One buffer uses MB lock credits; two buffers chain MB replay
            # BDs each so the next strip can fill while the current one replays.
            nb = len(bufs)
            reps = MB if nb > 1 else 1  # MM2S BDs per buffer
            per = 1 if nb > 1 else MB  # lock tokens per MM2S BD
            # Arrives row-major [NT*64][K]; scatter once into the staging
            # order [n-tile][kt][col][64 k] so every replay is a linear read.
            fill = [
                (
                    bufs[b],
                    (empty[b], per * reps),
                    (full[b], per * reps),
                    dict(
                        sizes=[NT, TILE, K // TILE, TILE // 4],
                        strides=[TILE * K // 4, TILE // 4, TILE * TILE // 4, 1],
                    ),
                )
                for b in range(nb)
            ]
            replay = [
                (bufs[b], (full[b], 1), (empty[b], 1), {})
                for b in range(nb)
                for _ in range(reps)
            ]
            return [
                (DMAChannelDir.S2MM, MT_S2MM_W, fill),
                (DMAChannelDir.MM2S, MT_MM2S_W, replay),
            ]

        def make_dma(dma_region, chains):
            # Each (direction, channel, BDs) chain loops forever. A BD is
            # (buffer, acquire lock/count, release lock/count, dma_bd kwargs).
            # Start blocks precede all BD blocks and the final end block.
            n = len(chains)
            first = [n + sum(len(c[2]) for c in chains[:j]) for j in range(n)]
            end = first[-1] + len(chains[-1][2])

            @dma_region
            def _(block):
                for j, (d, ch, bds) in enumerate(chains):
                    with block[j] if j else contextlib.nullcontext():
                        dma_start(
                            d,
                            ch,
                            dest=block[first[j]],
                            chain=block[j + 1 if j + 1 < n else end],
                        )
                    for k, (buf, acq, rel, kw) in enumerate(bds):
                        with block[first[j] + k]:
                            use_lock(acq[0], LockAction.AcquireGreaterEqual, acq[1])
                            dma_bd(buf, **kw)
                            use_lock(rel[0], LockAction.Release, rel[1])
                            next_bd(block[first[j] + (k + 1) % len(bds)])
                with block[end]:
                    EndOp()

        # Retile A from [row][K] to [kt][row][64]. Broadcast mode distributes
        # DDR reads across shims and sends each row group to all columns.
        # a_split=2 feeds two 32-row halves through separate core S2MM channels.
        a_dims = [(K // TILE, TILE // 4), (a_part_rows, K // 4), (TILE // 4, 1)]
        if a_bcast:
            bounds = [round(i * N_ROWS / N_COLS) for i in range(N_COLS + 1)]
            a_srcs = [
                (g, list(range(bounds[g], bounds[g + 1])), list(range(N_COLS)))
                for g in range(N_COLS)
                if bounds[g + 1] > bounds[g]
            ]
        else:
            a_srcs = [(c, list(range(N_ROWS)), [c]) for c in range(N_COLS)]
        a_mt, a_rows = {}, {}
        a_core = [[None] * N_ROWS for _ in range(N_COLS)]  # [c][r] -> parts
        for ac, rows, dests in a_srcs:
            ty = np.ndarray[(len(rows) * a_core_w,), np.dtype[np.int32]]
            a_mt[ac] = object_fifo(f"A_mt{ac}", shims[ac], mems[ac], 2, ty)
            a_rows[ac] = rows
            fifos, offsets = [], []
            for i, r in enumerate(rows):
                parts = []
                for h in range(a_split):
                    f = object_fifo(
                        f"A_c{ac}_r{r}_{h}",
                        mems[ac],
                        [cores[c][r] for c in dests],
                        1,
                        a_part_ty,
                        dimensionsToStream=a_dims,
                    )
                    fifos.append(f)
                    offsets.append(i * a_core_w + h * a_part_rows * K // 4)
                    parts.append(f)
                for c in dests:
                    a_core[c][r] = parts
            object_fifo_link(a_mt[ac], fifos, [], offsets)

        # Reassembly scatters each producer into its 64-row slice of a shared
        # [256][RW-byte] block. Manual DMAs allow this shared-buffer layout;
        # locks 14/15 on cores and 40-55 on memtiles avoid FIFO and W locks.
        # The alternative joins pairs of rows onto the two shim S2MM channels.
        C_SPLIT = 2
        c_rows = N_ROWS // C_SPLIT
        c_half_ty = np.ndarray[(c_rows * c_core_w,), np.dtype[np.int32]]
        c_core, c_mt, c_l1 = [], [], []
        mt_chains = [[] for _ in range(N_COLS)]  # extra memtile DMA chains
        rw_w = RW // 4  # one sub-block row, in words
        for c in range(N_COLS):
            if not c_reasm:
                c_core.append(
                    [
                        object_fifo(f"C_c{c}_r{r}", cores[c][r], mems[c], 2, c_core_ty)
                        for r in range(N_ROWS)
                    ]
                )
                halves = []
                for h in range(C_SPLIT):
                    f = object_fifo(f"C_mt{c}_{h}", mems[c], shims[c], 2, c_half_ty)
                    object_fifo_link(
                        c_core[c][h * c_rows : (h + 1) * c_rows],
                        f,
                        [i * c_core_w for i in range(c_rows)],
                        [],
                    )
                    halves.append(f)
                c_mt.append(halves)
                continue

            blk_ty = np.ndarray[(N_ROWS * TILE * rw_w,), np.dtype[np.int32]]
            blk = [buffer(mems[c], blk_ty, name=f"C_blk{c}_{b}") for b in range(2)]
            c_empty = [
                [
                    lock(
                        mems[c],
                        lock_id=40 + 2 * (b * N_ROWS + r),
                        init=1,
                        sym_name=f"C_empty{c}_{b}_{r}",
                    )
                    for r in range(N_ROWS)
                ]
                for b in range(2)
            ]
            c_full = [
                [
                    lock(
                        mems[c],
                        lock_id=41 + 2 * (b * N_ROWS + r),
                        init=0,
                        sym_name=f"C_full{c}_{b}_{r}",
                    )
                    for r in range(N_ROWS)
                ]
                for b in range(2)
            ]
            l1 = []
            for r in range(N_ROWS):
                t = cores[c][r]
                bufs = [
                    buffer(t, c_core_ty, name=f"C_l1_{c}_{r}_{b}") for b in range(2)
                ]
                prod = lock(t, lock_id=14, init=2, sym_name=f"C_prod{c}_{r}")
                cons = lock(t, lock_id=15, init=0, sym_name=f"C_cons{c}_{r}")
                l1.append((bufs, prod, cons))
                make_dma(
                    mem(t),
                    [
                        (
                            DMAChannelDir.MM2S,
                            0,
                            [(bufs[b], (cons, 1), (prod, 1), {}) for b in range(2)],
                        )
                    ],
                )
                flow(t, WireBundle.DMA, 0, mems[c], WireBundle.DMA, MT_S2MM_C[r])
                # NGS core buffers [64 rows][CT*64] side by side, in rows
                # r*64.. of the sub-block.
                scatter = dict(
                    offset=r * TILE * rw_w,
                    sizes=[NGS, TILE, c_core_w // TILE],
                    strides=[c_core_w // TILE, rw_w, 1],
                    transfer_len=NGS * c_core_w,
                )
                mt_chains[c].append(
                    (
                        DMAChannelDir.S2MM,
                        MT_S2MM_C[r],
                        [
                            (blk[b], (c_empty[b][r], 1), (c_full[b][r], 1), scatter)
                            for b in range(2)
                        ],
                    )
                )
            c_l1.append(l1)
            # Per-row credits prevent a fast producer from reserving a delayed
            # row's next buffer. Eight BDs drain ping rows then pong rows in
            # order, returning each credit only after its slice is read.
            mt_chains[c].append(
                (
                    DMAChannelDir.MM2S,
                    MT_MM2S_C,
                    [
                        (
                            blk[b],
                            (c_full[b][r], 1),
                            (c_empty[b][r], 1),
                            dict(offset=r * TILE * rw_w, transfer_len=TILE * rw_w),
                        )
                        for b in range(2)
                        for r in range(N_ROWS)
                    ],
                )
            )
            flow(mems[c], WireBundle.DMA, MT_MM2S_C, shims[c], WireBundle.DMA, 0)
            shim_dma_allocation(f"C_out{c}", shims[c], DMAChannelDir.S2MM, 0)

        for c in range(N_COLS):
            # W: shim -> memtile strip buffer (once) -> replayed MB times,
            # broadcast to core stream port W_STREAM_PORT of all 4 cores.
            flow(shims[c], WireBundle.DMA, 1, mems[c], WireBundle.DMA, MT_S2MM_W)
            for r in range(N_ROWS):
                flow(
                    mems[c],
                    WireBundle.DMA,
                    MT_MM2S_W,
                    cores[c][r],
                    WireBundle.Core,
                    W_STREAM_PORT,
                )
            shim_dma_allocation(f"W_in{c}", shims[c], DMAChannelDir.MM2S, 1)
            w_bufs = [
                buffer(mems[c], w_strip_ty, name=f"W_strip{c}_{b}") for b in range(wbuf)
            ]
            w_empty = [
                lock(
                    mems[c],
                    lock_id=56 + 2 * b,
                    init=MB,
                    sym_name=f"W_empty{c}_{b}",
                )
                for b in range(wbuf)
            ]
            w_full = [
                lock(mems[c], lock_id=57 + 2 * b, init=0, sym_name=f"W_full{c}_{b}")
                for b in range(wbuf)
            ]
            make_dma(
                memtile_dma(mems[c]), w_chains(w_bufs, w_empty, w_full) + mt_chains[c]
            )

        # c_reasm: the core alternates its two L1 C buffers in step with its
        # MM2S BD ping/pong, so the group loop is unrolled in Python; with an
        # odd NG, two M-blocks per loop iteration keep the parity aligned.
        mb_per_iter = 1 if (not c_reasm or NG % 2 == 0) else 2

        def make_core(c, r):
            @core(cores[c][r], stack_size=0x800)
            def _():
                for _ in range_(sys.maxsize):
                    for u in range(mb_per_iter):
                        a = [f.acquire(ObjectFifoPort.Consume, 1) for f in a_core[c][r]]
                        a_lo, a_hi = a[0], a[-1]  # same buffer when a_split == 1
                        gemm_stage_b()
                        if c_reasm:
                            bufs, prod, cons = c_l1[c][r]
                            for g in range(NG):
                                use_lock(prod, LockAction.AcquireGreaterEqual, 1)
                                for j in range(ct):
                                    last = g == NG - 1 and j == ct - 1
                                    (gemm_tile_last if last else gemm_tile)(
                                        a_lo, a_hi, bufs[(u * NG + g) % 2], j
                                    )
                                use_lock(cons, LockAction.Release, 1)
                        else:
                            for _ in range_(NG - 1):
                                o = c_core[c][r].acquire(ObjectFifoPort.Produce, 1)
                                for j in range(ct):
                                    gemm_tile(a_lo, a_hi, o, j)
                                c_core[c][r].release(ObjectFifoPort.Produce, 1)
                            o = c_core[c][r].acquire(ObjectFifoPort.Produce, 1)
                            for j in range(ct - 1):
                                gemm_tile(a_lo, a_hi, o, j)
                            gemm_tile_last(a_lo, a_hi, o, ct - 1)
                            c_core[c][r].release(ObjectFifoPort.Produce, 1)
                        for f in a_core[c][r]:
                            f.release(ObjectFifoPort.Consume, 1)

        for c in range(N_COLS):
            for r in range(N_ROWS):
                if kernel == "aievec":
                    aievec_core(
                        cores[c][r],
                        a_core[c][r],
                        c_l1[c][r],
                        K=K,
                        NG=NG,
                        ct=ct,
                        a_split=a_split,
                        out_i8=out_i8,
                        shift=shift,
                        trace=bool(trace_size),
                    )
                else:
                    make_core(c, r)

        # Trace column 0 on one broadcast timeline; packet IDs >=8 avoid
        # the DMA completion return flows (IDs 0..N_COLS-1).
        trace_names = []
        if trace_size:

            def core_trace(r):
                @trace(cores[0][r], f"core_trace_r{r}")
                def _():
                    trace_mode(TraceMode.EventTime)
                    trace_packet(8 + r, TracePacketType.Core)
                    trace_event("INSTR_EVENT_0")  # tile start (kernel event0)
                    trace_event("INSTR_EVENT_1")  # tile end (kernel event1)
                    trace_event("INSTR_MATRIX")  # MMAC issued
                    trace_event("STREAM_STALL")  # waiting on W stream
                    trace_event("LOCK_STALL")  # waiting on A / C buffers
                    trace_event("MEMORY_STALL")
                    # Use starvation/backpressure: looping BDs stay RUNNING.
                    trace_event("DMA_S2MM_0_STREAM_STARVATION")  # A not arriving
                    trace_event("DMA_MM2S_0_STREAM_BACKPRESSURE")  # C drain blocked
                    trace_start(broadcast=3)
                    trace_stop(broadcast=4)

                trace_names.append(f"core_trace_r{r}")

            for r in range(N_ROWS):
                core_trace(r)

            # Select DMA channels directly; trace_port is unavailable on npu3.
            if c_reasm:
                mt_sel = dict(
                    mm2s0=MT_MM2S_W, mm2s1=MT_MM2S_C, s2mm0=MT_S2MM_W, s2mm1=0
                )
                mt_events = [
                    "DMA_MM2S_SEL0_RUNNING",  # W replay
                    "DMA_MM2S_SEL0_STREAM_BACKPRESSURE",
                    "DMA_MM2S_SEL0_STALLED_LOCK",
                    "DMA_MM2S_SEL1_RUNNING",  # C drain
                    "DMA_MM2S_SEL1_STREAM_BACKPRESSURE",  # DDR write limit
                    "DMA_MM2S_SEL1_STALLED_LOCK",  # C row slice not complete
                    "DMA_S2MM_SEL1_RUNNING",  # A from shim
                    "DMA_S2MM_SEL1_STALLED_LOCK",
                ]
            else:
                mt_sel = dict(mm2s0=MT_MM2S_W, mm2s1=0, s2mm0=MT_S2MM_W, s2mm1=0)
                mt_events = [
                    "DMA_MM2S_SEL0_RUNNING",  # W replay
                    "DMA_MM2S_SEL0_STREAM_BACKPRESSURE",
                    "DMA_MM2S_SEL0_STALLED_LOCK",
                    "DMA_S2MM_SEL0_RUNNING",  # W fill
                    "DMA_MM2S_SEL1_RUNNING",  # A to row 2
                    "DMA_MM2S_SEL1_STALLED_LOCK",
                    "DMA_S2MM_SEL1_RUNNING",  # A from shim
                    "DMA_S2MM_SEL1_STALLED_LOCK",
                ]

            @trace(mems[0], "memtile_trace")
            def _():
                trace_mode(TraceMode.EventTime)
                trace_packet(12, TracePacketType.MemTile)
                for e in mt_events:
                    trace_event(e)
                trace_start(broadcast=3)
                trace_stop(broadcast=4)

            trace_names.append("memtile_trace")

            @trace(shims[0], "shim_trace")
            def _():
                trace_packet(13, TracePacketType.ShimTile)
                trace_event("DMA_MM2S_0_RUNNING")  # A from DDR
                trace_event("DMA_MM2S_1_RUNNING")  # W from DDR
                trace_event("DMA_S2MM_0_RUNNING")  # C to DDR (both C paths)
                trace_event("DMA_MM2S_0_MEMORY_STARVATION")
                trace_event("DMA_MM2S_1_MEMORY_STARVATION")
                trace_event("DMA_S2MM_0_MEMORY_BACKPRESSURE")
                trace_event("DMA_S2MM_0_STREAM_STARVATION")
                trace_event("DMA_MM2S_1_STREAM_BACKPRESSURE")
                trace_start(event="TRUE")
                trace_stop(event="NONE")

            trace_names.append("shim_trace")

        @runtime_sequence(a_host_ty, w_host_ty, c_host_ty)
        def sequence(A, W, C):
            row_w = N * out_bytes // 4  # one C row, in words
            tile_w = TILE * out_bytes // 4  # one C tile row, in words

            if trace_size:
                # DMA_Event_Channel_Selection_A uses raw DMA channel numbers.
                npu_write32(
                    0x9F40,
                    (mt_sel["mm2s1"] << 24)
                    | (mt_sel["mm2s0"] << 16)
                    | (mt_sel["s2mm1"] << 8)
                    | mt_sel["s2mm0"],
                    column=0,
                    row=1,
                )
                # The trace buffer becomes an extra (4th) runtime argument.
                for name in trace_names:
                    trace_start_config(name)
                trace_host_config(trace_size)

            def issue_inputs():
                tasks = []
                for c in range(N_COLS):
                    # Column c's strip of W rows (contiguous in row-major W).
                    tasks.append(
                        shim_dma_single_bd_task(
                            f"W_in{c}",
                            W,
                            offset=c * w_strip_w,
                            sizes=[1, NT, TILE, K // 4],
                            strides=[0, TILE * K // 4, K // 4, 1],
                            issue_token=True,
                        )
                    )
                    if c in a_mt:  # this column's A row-blocks, every M-block
                        rows = a_rows[c]
                        tasks.append(
                            shim_dma_single_bd_task(
                                a_mt[c],
                                A,
                                offset=rows[0] * TILE * K // 4,
                                sizes=[1, MB, len(rows) * TILE, K // 4],
                                strides=[0, N_ROWS * TILE * K // 4, K // 4, 1],
                                issue_token=True,
                            )
                        )
                dma_start_task(*tasks)
                return tasks

            # Issue the next inputs early to overlap W loading with compute.
            inputs = issue_inputs()
            for it in range(iters):
                next_inputs = issue_inputs() if (it + 1 < iters and wbuf > 1) else None
                # C: one task per M-block and column (c_reasm: SB sub-blocks
                # of 256 rows x RW bytes; else one per half, NG buffers of
                # 128 rows x CT*64 cols), scattered into row-major C.
                prev = []
                for mb in range(MB):
                    if c_reasm:
                        cur = [
                            shim_dma_single_bd_task(
                                f"C_out{c}",
                                C,
                                offset=mb * N_ROWS * TILE * row_w + c * NT * tile_w,
                                sizes=[1, SB, N_ROWS * TILE, rw_w],
                                strides=[0, rw_w if SB > 1 else 0, row_w, 1],
                                issue_token=True,
                            )
                            for c in range(N_COLS)
                        ]
                    else:
                        cur = [
                            shim_dma_single_bd_task(
                                c_mt[c][h],
                                C,
                                offset=(mb * N_ROWS + h * c_rows) * TILE * row_w
                                + c * NT * tile_w,
                                sizes=[1, NG, c_rows * TILE, ct * tile_w],
                                strides=[0, ct * tile_w, row_w, 1],
                                issue_token=True,
                            )
                            for c in range(N_COLS)
                            for h in range(C_SPLIT)
                        ]
                    dma_start_task(*cur)
                    if prev:
                        dma_await_task(*prev)
                    prev = cur
                dma_await_task(*prev)
                dma_await_task(*inputs)
                if next_inputs is None and it + 1 < iters:
                    next_inputs = issue_inputs()
                inputs = next_inputs


def aievec_core(core_tile, a_fifos, c_l1, *, K, NG, ct, a_split, out_i8, shift, trace):
    """Emit core (c, r)'s body in flat AIEVec (KERNEL=aievec).

    Same schedule as kernel.cc: prime 32 W pops per strip, then each 64x64
    output tile runs KT K-tiles of commit + 64 row MMACs, popping the next
    K-tile's W after every odd row. The K-loop covers K-tiles 0..KT-2 with a
    dynamic zero_if(kt == 0); the last K-tile is peeled and pre-stages the
    next output tile unless it ends the strip. The tile body is emitted
    twice: in the runtime output-tile loop and as the peeled strip end. The
    C buffer parity is computed at run time from the M-block count.
    """
    from aie.dialects import arith as arith_d, scf as scf_d, ub as ub_d
    from aie.dialects import vector as vector_d
    from aie.dialects.aie import event as aie_event
    from aie.helpers.dialects.scf import if_, yield_
    from aie.helpers.dialects.aievec_aie4 import (
        INT8,
        POS_INF,
        SAT,
        commit,
        mmac_active,
        srs,
        stage_stream,
    )
    from aie.ir import (
        Attribute,
        IndexType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        MemRefType,
        Operation,
        Type,
        VectorType,
    )

    KT = K // TILE
    # KT == 1 would make the peeled K-tile accumulate onto the poison init.
    assert KT >= 2, "--kernel aievec needs K >= 128"
    out_bytes = 1 if out_i8 else 4
    i32 = IntegerType.get_signless(32)
    V64I8 = VectorType.get([64], IntegerType.get_signless(8))
    V16I32 = VectorType.get([16], i32)
    V64I32 = VectorType.get([64], i32)
    KT_WORDS = (TILE // a_split) * TILE // 4  # one K-tile of one A part
    C_ROW_WORDS = ct * TILE * out_bytes // 4  # C buffer row stride
    C_TILE_WORDS = TILE * out_bytes // 4  # C tile j step
    bufs, prod, cons = c_l1

    def idx(v):
        return arith_d.constant(IndexType.get(), v)

    def c32(v):
        return arith_d.constant(i32, v)

    # A rows are addressed from one pointer per A part (ptr dialect), carried
    # across K-tiles as a K-loop iter_arg and advanced by one K-tile per
    # iteration. Peano then keeps it in a pointer register and walks the rows
    # with post-increment loads, as the C kernel's pLo/pHi do. Indexing the
    # buffer by kt instead makes Peano rebuild the K-tile address from the
    # loop counter at the top of every iteration.
    PTR = Type.parse("!ptr.ptr<#ptr.generic_space>")
    GEN_SPACE = Attribute.parse("#ptr.generic_space")

    def to_ptr(buf):
        """(pointer to buf, its generic-space memref type, its metadata)"""
        mt = MemRefType(buf.type)
        gty = MemRefType.get(mt.shape, mt.element_type, memory_space=GEN_SPACE)
        # ptr.to_ptr needs a #ptr.generic_space memref. memref.memory_space_cast
        # is not a drop-in (memref-to-LLVM cannot map that space); this cast
        # folds away in reconcile-unrealized-casts after the LLVM lowering.
        g = Operation.create(
            "builtin.unrealized_conversion_cast", results=[gty], operands=[buf]
        ).result
        p = Operation.create("ptr.to_ptr", results=[PTR], operands=[g]).result
        md_ty = Type.parse(f"!ptr.ptr_metadata<{gty}>")
        md = Operation.create("ptr.get_metadata", results=[md_ty], operands=[g])
        return p, gty, md.result

    def ptr_add(p, nbytes):
        return Operation.create(
            "ptr.ptr_add",
            results=[PTR],
            operands=[p, idx(nbytes)],
            # The flags property is a plain i32 in the generic form (no Python
            # enum binding for the ptr dialect); 3 is what `ptr_add inbounds`
            # parses to.
            attributes={"flags": IntegerAttr.get(i32, 3)},
        ).result

    def a_row(ptrs, meta, r):
        # A part layout [kt][rows][64 int8]; a_split == 1 keeps all 64 rows
        # in a_lo (= a_hi).
        rows = TILE // a_split
        part, rr = (1, r - rows) if r >= rows else (0, r)
        if a_split == 1:
            part = 0
        gty, md = meta[part]
        q = ptr_add(ptrs[part], rr * 64) if rr else ptrs[part]
        m = Operation.create("ptr.from_ptr", results=[gty], operands=[q, md]).result
        w = vector_d.load(V16I32, m, [idx(0)], alignment=64)
        return vector_d.bitcast(V64I8, w)

    def ktile(ptrs, meta, accs, *, zero_if, prefetch):
        commit(INT8, rhs_signed=True)
        out = []
        for r in range(TILE):
            out.append(
                mmac_active(
                    INT8,
                    a_row(ptrs, meta, r),
                    accs[r],
                    zero_if=zero_if,
                    lhs_signed=True,
                    rhs_signed=True,
                )
            )
            if prefetch and r & 1:
                stage_stream(INT8, W_STREAM_PORT, r // 2, rhs_signed=True)
        return out

    def emit_tile(a_lo, a_hi, c_buf, j_words, *, prefetch_last):
        if trace:
            aie_event(0)
        parts = [to_ptr(a_lo)] + ([] if a_split == 1 else [to_ptr(a_hi)])
        meta = [(gty, md) for _, gty, md in parts]
        # zero_if(kt == 0) ignores the accumulator, so the init is never read.
        init = [ub_d.poison(V64I32)] * TILE + [p for p, _, _ in parts]
        for kt, args, res in range_(0, KT - 1, iter_args=init, insert_yield=False):
            accs, ptrs = list(args[:TILE]), list(args[TILE:])
            yield_(
                ktile(
                    ptrs,
                    meta,
                    accs,
                    zero_if=arith_d.cmpi(arith_d.CmpIPredicate.eq, kt, idx(0)),
                    prefetch=True,
                )
                + [ptr_add(p, KT_WORDS * 4) for p in ptrs]
            )
        last = ktile(
            list(res[TILE:]),
            meta,
            list(res[:TILE]),
            zero_if=None,
            prefetch=prefetch_last,
        )
        for r in range(TILE):
            off = arith_d.addi(j_words, idx(r * C_ROW_WORDS))
            if out_i8:
                y = srs(
                    last[r], shift=shift, signed=True, rounding=POS_INF, saturation=SAT
                )
                vector_d.store(vector_d.bitcast(V16I32, y), c_buf, [off], alignment=64)
            else:
                vector_d.store(last[r], c_buf, [off], alignment=64)
        if trace:
            aie_event(1)

    def c_select(sel):
        # sel: a Python int, or an i32 Value (nonzero selects bufs[1]).
        if isinstance(sel, int):
            return bufs[sel].result
        op = scf_d.IfOp(
            arith_d.cmpi(arith_d.CmpIPredicate.ne, sel, c32(0)),
            [bufs[0].result.type],
            has_else=True,
        )
        with InsertionPoint(op.then_block):
            scf_d.yield_([bufs[1].result])
        with InsertionPoint(op.else_block):
            scf_d.yield_([bufs[0].result])
        return op.results[0]

    def xor_parity(a, b):
        # Each operand: None (= 0), a Python int, or an i32 Value.
        a = 0 if a is None else a
        b = 0 if b is None else b
        if isinstance(a, int) and isinstance(b, int):
            return (a ^ b) & 1
        if isinstance(a, int):
            a, b = b, a
        if isinstance(b, int):
            return arith_d.xori(a, c32(1)) if b & 1 else a
        return arith_d.xori(a, b)

    def eq(v, n):
        return arith_d.cmpi(arith_d.CmpIPredicate.eq, v, idx(n))

    # With KT == 2 both K-tiles are straight-line code and Peano spills
    # accumulators: the spill slots fill the default 2 KiB stack exactly.
    stack = 0x1000 if KT == 2 else 0x800

    # Lock values are plain constants (use_lock's wrapper), so objectfifo
    # lowering never peels these loops.
    @core(core_tile, stack_size=stack)
    def _():
        for m in range_(sys.maxsize):
            a = [f.acquire(ObjectFifoPort.Consume, 1) for f in a_fifos]
            a_lo, a_hi = a[0], a[-1]  # same buffer when a_split == 1
            for i in range(32):  # prime the strip's first K-tile
                stage_stream(INT8, W_STREAM_PORT, i, rhs_signed=True)
            # Buffer of group g in M-block m: (m*NG + g) & 1.
            base = arith_d.andi(arith_d.index_cast(i32, m), c32(1)) if NG % 2 else None
            n_loop = NG * ct - 1  # every tile but the strip's last
            if n_loop:
                for t in range_(0, n_loop):
                    g = arith_d.divui(t, idx(ct))
                    j = arith_d.remui(t, idx(ct))
                    if ct > 1:
                        with if_(eq(j, 0), hasElse=False):
                            use_lock(prod, LockAction.AcquireGreaterEqual, 1)
                    else:
                        use_lock(prod, LockAction.AcquireGreaterEqual, 1)
                    g_par = (
                        None
                        if NG == 1
                        else arith_d.andi(arith_d.index_cast(i32, g), c32(1))
                    )
                    emit_tile(
                        a_lo,
                        a_hi,
                        c_select(xor_parity(base, g_par)),
                        arith_d.muli(j, idx(C_TILE_WORDS)),
                        prefetch_last=True,
                    )
                    if ct > 1:
                        with if_(eq(j, ct - 1), hasElse=False):
                            use_lock(cons, LockAction.Release, 1)
                    else:
                        use_lock(cons, LockAction.Release, 1)
            if ct == 1:  # else the last group was acquired at j == 0
                use_lock(prod, LockAction.AcquireGreaterEqual, 1)
            emit_tile(
                a_lo,
                a_hi,
                c_select(xor_parity(base, (NG - 1) & 1)),
                idx((ct - 1) * C_TILE_WORDS),
                prefetch_last=False,
            )
            use_lock(cons, LockAction.Release, 1)
            for f in a_fifos:
                f.release(ObjectFifoPort.Consume, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="int8 GEMM for npu3")
    p.add_argument("-M", type=int, default=2048)
    p.add_argument("-N", type=int, default=3072)
    p.add_argument("-K", type=int, default=1024)
    p.add_argument("--out", choices=["i8", "i32"], default="i8")
    p.add_argument("--iters", type=int, default=1, help="GEMMs per launch")
    p.add_argument("--cols", type=int, default=3, help="array columns (1-3)")
    p.add_argument(
        "--trace-size", type=int, default=0, help="event trace bytes (0: off)"
    )
    p.add_argument(
        "--a-split", type=int, default=2, help="A DMA channels per core (1 or 2)"
    )
    p.add_argument(
        "--wbuf", type=int, default=2, help="memtile W strip buffers (1 or 2)"
    )
    p.add_argument("--ct", type=int, default=4, help="C tiles per core output buffer")
    p.add_argument(
        "--c-reasm",
        type=int,
        default=1,
        help="1: memtile reassembles C into long DDR write runs; 0: 2 joins",
    )
    p.add_argument(
        "--c-run", type=int, default=1024, help="max C DDR write run (c_reasm), bytes"
    )
    p.add_argument(
        "--a-bcast",
        type=int,
        default=1,
        help="1: fetch A once and broadcast across columns; 0: per column",
    )
    p.add_argument(
        "--kernel",
        choices=["extern", "aievec"],
        default="extern",
        help="extern: link kernel.o; aievec: emit the core in flat AIEVec",
    )
    p.add_argument(
        "--shift",
        type=int,
        default=12,
        choices=range(60),
        metavar="0..59",
        help="OUT=i8 SRS right shift (--kernel aievec only)",
    )
    args = p.parse_args()
    if args.kernel == "aievec":
        # A C acquire in the group loop would be unrolled with the tile body.
        assert args.c_reasm, "--kernel aievec requires --c-reasm 1"
    N_COLS = args.cols
    with mlir_mod_ctx() as ctx:
        build(
            args.M,
            args.N,
            args.K,
            args.out == "i8",
            args.iters,
            bool(args.a_bcast),
            args.ct,
            args.wbuf,
            args.trace_size,
            args.a_split,
            args.c_reasm,
            args.c_run,
            args.kernel,
            args.shift,
        )
        res = ctx.module.operation.verify()
        if res is not True:
            print(res, file=sys.stderr)
            sys.exit(1)
        print(ctx.module)
