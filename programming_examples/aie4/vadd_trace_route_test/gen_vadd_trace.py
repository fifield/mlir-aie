#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# Generate MLIR for a 3-column vector add with trace on a configurable tile.
#
# Usage:
#   python3 gen_vadd_trace.py --row 2 --col 0              # print to stdout
#   python3 gen_vadd_trace.py --row 3 --col 0 -o out.mlir  # write to file
#
# npu3 layout:
#   row 0: shim tiles (DMA/NOC interface)
#   row 1: memtile tiles (staging SRAM)
#   rows 2-5: compute tiles (AIE cores)
#
# This exercises trace paths from every compute row through the 32-bit network
# and across a hard-wired 32-bit/512-bit bridge to the shim trace DMA.

import argparse
import sys

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU3, Tile
from aie.utils.trace.events import CoreEventAIE4 as CoreEvent

NCOLS = 3
COL_ELEMS = 4096
CHUNK = 1024
TRACE_BYTES = 8192


def gen(row: int, trace_col: int):
    assert 2 <= row <= 5, f"row must be 2-5, got {row}"
    assert 0 <= trace_col < NCOLS, f"col must be 0-{NCOLS-1}, got {trace_col}"

    column_ty = np.ndarray[(COL_ELEMS,), np.dtype[np.int32]]
    chunk_ty = np.ndarray[(CHUNK,), np.dtype[np.int32]]
    host_ty = np.ndarray[(NCOLS * COL_ELEMS,), np.dtype[np.int32]]

    def core_fn(in_a, in_b, out_c):
        for _ in range_(COL_ELEMS // CHUNK):
            elem_out = out_c.acquire(1)
            elem_a = in_a.acquire(1)
            elem_b = in_b.acquire(1)
            for k in range_(CHUNK):
                elem_out[k] = elem_a[k] + elem_b[k]
            in_a.release(1)
            in_b.release(1)
            out_c.release(1)

    workers = []
    inputs_a = []
    inputs_b = []
    outputs_c = []
    for col in range(NCOLS):
        shim = Tile(col, 0)
        mem = Tile(col, 1)

        in_a = ObjectFifo(column_ty, name=f"inA_s{col}")
        in_b = ObjectFifo(column_ty, name=f"inB_s{col}")
        out_c = ObjectFifo(column_ty, name=f"outC_s{col}")

        core_a = in_a.cons().forward(tile=mem, obj_type=chunk_ty, name=f"inA_m{col}")
        core_b = in_b.cons().forward(tile=mem, obj_type=chunk_ty, name=f"inB_m{col}")
        core_c = out_c.prod().join(
            [0], tile=mem, obj_types=[chunk_ty], names=[f"outC_m{col}"]
        )[0]

        workers.append(
            Worker(
                core_fn,
                fn_args=[core_a.cons(), core_b.cons(), core_c.prod()],
                tile=Tile(col, row),
            )
        )
        inputs_a.append(in_a.prod(tile=shim))
        inputs_b.append(in_b.prod(tile=shim))
        outputs_c.append(out_c.cons(tile=shim))

    def sequence(a, b, c, inputs_a, inputs_b, outputs_c):
        for col in range(NCOLS):
            offset = col * COL_ELEMS
            inputs_a[col].fill(a, offset=offset, sizes=[COL_ELEMS], wait=True)
            inputs_b[col].fill(b, offset=offset, sizes=[COL_ELEMS], wait=True)
            outputs_c[col].drain(c, offset=offset, sizes=[COL_ELEMS], wait=True)

    rt = Runtime(sequence, [host_ty, host_ty, host_ty, inputs_a, inputs_b, outputs_c])
    prog = Program(NPU3(), rt, workers=workers)
    prog.enable_trace(
        trace_size=TRACE_BYTES,
        workers=[workers[trace_col]],
        coretile_events=[
            CoreEvent.INSTR_EVENT_0,
            CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_VECTOR,
            CoreEvent.MEMORY_STALL,
            CoreEvent.STREAM_STALL,
            CoreEvent.LOCK_STALL,
            CoreEvent.DMA_S2MM_0_RUNNING,
            CoreEvent.DMA_MM2S_0_RUNNING,
        ],
    )
    return prog.resolve_program()


def main():
    p = argparse.ArgumentParser(
        description="Generate vadd_trace MLIR for npu3 with configurable trace tile."
    )
    p.add_argument(
        "--row",
        "-r",
        type=int,
        default=2,
        choices=range(2, 6),
        help="Compute row to trace (2-5, default: 2)",
    )
    p.add_argument(
        "--col",
        "-c",
        type=int,
        default=0,
        choices=range(NCOLS),
        help="Compute column to trace (default: 0)",
    )
    p.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = p.parse_args()

    mlir = gen(args.row, args.col)

    if args.output:
        with open(args.output, "w") as f:
            print(mlir, file=f)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(mlir)


if __name__ == "__main__":
    main()
