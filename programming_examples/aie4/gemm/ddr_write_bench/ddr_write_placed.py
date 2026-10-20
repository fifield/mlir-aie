# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measure npu3 DDR writes in L-byte runs, with optional concurrent reads.

Memtile DMA loops feed each column's S=WB/COLS byte strip. Each block's
rows are split across CH shim channels; tasks use [S/L chunks, rows/CH, L/4]
word dimensions. Start each block before awaiting the previous one, as in GEMM.
REPS repeats the matrix; --rd sets read bytes per write byte, and
--block-rows sets rows per task group. No cores participate.
"""

import argparse
import contextlib
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

BLOCK_ROWS = 256  # rows per shim task group (gemm: one M-block)
SRC_WORDS = 16384  # 64 KiB memtile source buffer per channel
FILL = 0x5A5A5A5A


def geometry(L, cols, total_bytes=6 * 1024 * 1024, wb=3072):
    wb = max(wb, L * cols)
    assert (wb // cols) % L == 0, "strip width must be a multiple of L"
    rows = total_bytes // wb
    rows -= rows % BLOCK_ROWS
    return wb, rows


def build(L, cols, ch, reps, burst, wb_min, rd_ratio=0.0):
    assert L % 64 == 0 and ch in (1, 2) and 1 <= cols <= 3
    WB, ROWS = geometry(L, cols, wb=wb_min)
    S = WB // cols
    nblk = ROWS // BLOCK_ROWS
    rows_per_ch = BLOCK_ROWS // ch

    @device(AIEDevice.npu3)
    def device_body():
        host_ty = np.ndarray[(ROWS * WB // 4,), np.dtype[np.int32]]
        src_ty = np.ndarray[(SRC_WORDS,), np.dtype[np.int32]]
        shims = [tile(c, 0) for c in range(3)]
        mems = [tile(c, 1) for c in range(cols)]
        for c in range(3):
            shims[c].attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")(
                [0, c]
            )
        for ctrl_ch in range(2):
            for c in range(3):
                packetflow(
                    c,
                    shims[c],
                    WireBundle.Control32,
                    ctrl_ch,
                    [{"dest": shims[0], "port": WireBundle.Controller32, "channel": 0}],
                )

        for c in range(cols):
            loops = []  # (direction, memtile channel, buffer): endless BD loops
            for h in range(ch):
                src = buffer(
                    mems[c],
                    src_ty,
                    name=f"src{c}_{h}",
                    initial_value=np.full(SRC_WORDS, FILL, dtype=np.int32),
                )
                flow(mems[c], WireBundle.DMA, h, shims[c], WireBundle.DMA, h)
                loops.append((DMAChannelDir.MM2S, h, src))
            if rd_ratio:
                sink = buffer(mems[c], src_ty, name=f"sink{c}")
                flow(shims[c], WireBundle.DMA, 0, mems[c], WireBundle.DMA, 0)
                loops.append((DMAChannelDir.S2MM, 0, sink))

            def make_dma(mt, loops):
                @memtile_dma(mt)
                def _(block):
                    n = len(loops)
                    # block 2i+1: channel i's BD (loops to itself); block
                    # 2i+2: start of channel i+1 (or the end block).
                    for i, (d, chn, buf) in enumerate(loops):
                        with block[2 * i] if i else contextlib.nullcontext():
                            dma_start(
                                d, chn, dest=block[2 * i + 1], chain=block[2 * i + 2]
                            )
                        with block[2 * i + 1]:
                            dma_bd(buf)
                            next_bd(block[2 * i + 1])
                    with block[2 * n]:
                        EndOp()

            make_dma(mems[c], loops)

        @runtime_sequence(host_ty, host_ty)
        def sequence(C, R):
            # Round concurrent reads down to a whole 64-byte transfer.
            rd_w = int(rd_ratio * S * BLOCK_ROWS / 64) * 16
            prev = []
            for _ in range(reps):
                for b in range(nblk):
                    cur = []
                    for c in range(cols):
                        for h in range(ch):
                            t = dma_configure_task(
                                shims[c], DMAChannelDir.S2MM, h, issue_token=True
                            )
                            with bds(t) as bd:
                                with bd[0]:
                                    shim_dma_bd(
                                        C,
                                        offset=(b * BLOCK_ROWS + h * rows_per_ch)
                                        * WB
                                        // 4
                                        + c * S // 4,
                                        sizes=[1, S // L, rows_per_ch, L // 4],
                                        strides=[0, L // 4, WB // 4, 1],
                                        transfer_len=S * rows_per_ch // 4,
                                        burst_length=burst,
                                    )
                                    EndOp()
                            cur.append(t)
                    for c in range(cols if rd_w else 0):
                        t = dma_configure_task(
                            shims[c], DMAChannelDir.MM2S, 0, issue_token=True
                        )
                        with bds(t) as bd:
                            with bd[0]:
                                shim_dma_bd(
                                    R,
                                    offset=((b * cols + c) * rd_w)
                                    % (ROWS * WB // 4 - rd_w),
                                    sizes=[1, 1, 1, rd_w],
                                    strides=[0, 0, 0, 1],
                                    transfer_len=rd_w,
                                )
                                EndOp()
                        cur.append(t)
                    dma_start_task(*cur)
                    if prev:
                        dma_await_task(*prev)
                    prev = cur
            dma_await_task(*prev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=256, help="contiguous run bytes")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--ch", type=int, default=2, help="S2MM channels per column")
    p.add_argument("--reps", type=int, default=8, help="C matrix writes/launch")
    p.add_argument("--burst", type=int, default=0, help="0 = max (512 B)")
    p.add_argument("--wb", type=int, default=3072, help="min row stride bytes")
    p.add_argument(
        "--rd", type=float, default=0.0, help="concurrent read bytes / write bytes"
    )
    p.add_argument("--geometry", action="store_true", help="print WB ROWS")
    p.add_argument("--block-rows", type=int, default=256, help="rows per task")
    a = p.parse_args()
    BLOCK_ROWS = a.block_rows
    if a.geometry:
        print(*geometry(a.L, a.cols, wb=a.wb))
        sys.exit(0)
    with mlir_mod_ctx() as ctx:
        build(a.L, a.cols, a.ch, a.reps, a.burst, a.wb, a.rd)
        res = ctx.module.operation.verify()
        if res is not True:
            print(res, file=sys.stderr)
            sys.exit(1)
        print(ctx.module)
