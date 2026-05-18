# dispatch_micro/generate.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""
Parameterized placed-IRON MLIR generator for the dispatch_micro benchmark suite.

The same Python source covers all four mechanisms; the mechanism switch only
controls whether a top-level `aiex.npu.load_pdi` is emitted (firmware path).
All other mechanism differences are handled by aiecc flags at build time.
"""
import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


DEVICES = {
    "npu2_1col": (AIEDevice.npu2_1col, 1),
    "npu2_4col": (AIEDevice.npu2_4col, 4),
    "npu2":      (AIEDevice.npu2,      8),
}

LINE_LEN = 1024  # int32 elements per BD


def _emit_device(name, dev_enum, cols, rows_per_col, bds_per_task, topology,
                 with_load_pdi_self):
    """Emit one @device region named `name`.

    `cols` is the number of shim columns used; `rows_per_col` is the number of
    compute tiles per column (rows 2..2+rows_per_col-1). Total compute tiles
    is cols * rows_per_col. When rows_per_col > 1, a memtile per column fans
    data out to each compute row and joins the outputs back. The kernel sees
    only two BOs (one in, one out) regardless of total tile count.
    """

    # Single in / single out runtime buffer shared across tiles. We're capped
    # at 5 BO slots by aiecc.cpp:3558, so packing per-tile data into one
    # global tensor is the only way to scale across the array.
    n_compute = cols * rows_per_col
    total_words = LINE_LEN * bds_per_task * n_compute

    @device(dev_enum, sym_name=name)
    def device_body():
        line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int32]]
        col_chunk_ty = np.ndarray[
            (LINE_LEN * bds_per_task * rows_per_col,), np.dtype[np.int32]
        ]
        tensor_ty = np.ndarray[(total_words,), np.dtype[np.int32]]

        pass_thru = external_func(
            "passThroughLine",
            inputs=[line_ty, line_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        compute_tiles = []
        of_ins = []    # shim-side input fifo per column (what the runtime
                       # sequence pushes into)
        of_outs = []   # shim-side output fifo per column
        _per_core_in_fifos = []   # per-compute-tile input fifo (what the
                                   # core acquires from)
        _per_core_out_fifos = []  # per-compute-tile output fifo

        def _make_column(c, shim_col):
            """Build the per-column fifo structure.
            Returns (comp_tiles, shim_in, shim_out, comp_in_fifos, comp_out_fifos).
            shim_in/out are the fifos the runtime sequence pushes/pulls; the
            comp_*_fifos are what each core acquires/releases on."""
            ShimT = tile(shim_col, 0)
            comp_tiles_local = [tile(c, 2 + r) for r in range(rows_per_col)]

            if rows_per_col == 1:
                in_fifo = object_fifo(
                    f"in_c{c}", ShimT, comp_tiles_local[0], 2, line_ty
                )
                out_fifo = object_fifo(
                    f"out_c{c}", comp_tiles_local[0], ShimT, 2, line_ty
                )
                return (comp_tiles_local, in_fifo, out_fifo,
                        [in_fifo], [out_fifo])

            # rows_per_col > 1: fan out via memtile in column c.
            MemT = tile(c, 1)
            shim_to_mem = object_fifo(
                f"shimmem_c{c}", ShimT, MemT, 2, col_chunk_ty
            )
            mem_to_comps = [
                object_fifo(
                    f"in_c{c}_r{r}", MemT, comp_tiles_local[r], 2, line_ty
                )
                for r in range(rows_per_col)
            ]
            object_fifo_link(
                shim_to_mem, mem_to_comps,
                srcOffsets=[],
                dstOffsets=[
                    r * LINE_LEN * bds_per_task for r in range(rows_per_col)
                ],
            )
            comp_to_mems = [
                object_fifo(
                    f"out_c{c}_r{r}", comp_tiles_local[r], MemT, 2, line_ty
                )
                for r in range(rows_per_col)
            ]
            mem_to_shim = object_fifo(
                f"memshim_c{c}", MemT, ShimT, 2, col_chunk_ty
            )
            object_fifo_link(
                comp_to_mems, mem_to_shim,
                srcOffsets=[
                    r * LINE_LEN * bds_per_task for r in range(rows_per_col)
                ],
                dstOffsets=[],
            )
            return (comp_tiles_local, shim_to_mem, mem_to_shim,
                    mem_to_comps, comp_to_mems)

        # Build columns; topology decides which shim each column uses.
        for c in range(cols):
            shim_col = 0 if topology == "branch" else c
            comps, shim_in, shim_out, comp_ins, comp_outs = _make_column(c, shim_col)
            compute_tiles.extend(comps)
            of_ins.append(shim_in)
            of_outs.append(shim_out)
            # Track per-compute-tile fifos for the core bodies.
            for r in range(rows_per_col):
                _per_core_in_fifos.append(comp_ins[r])
                _per_core_out_fifos.append(comp_outs[r])

        if topology not in ("linear", "hop", "branch"):
            raise ValueError(f"Unknown topology: {topology}")

        for idx, CompT in enumerate(compute_tiles):
            in_fifo = _per_core_in_fifos[idx]
            out_fifo = _per_core_out_fifos[idx]

            @core(CompT)
            def _():
                for _i in range_(sys.maxsize):
                    elem_in  = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                    pass_thru(elem_in, elem_out, LINE_LEN * 4)
                    in_fifo.release(ObjectFifoPort.Consume, 1)
                    out_fifo.release(ObjectFifoPort.Produce, 1)

        # Per-column shim-DMA chunk size: with rows_per_col > 1 we push a
        # wide chunk per BD that the memtile then splits across rows.
        chunk_words = LINE_LEN * rows_per_col

        @runtime_sequence(tensor_ty, tensor_ty)
        def seq(in_buf, out_buf):
            if with_load_pdi_self:
                npu_load_pdi(device_ref=name)

            # BD-count axis: `bds_per_task` independent dma_configure_task_for
            # ops per direction per column, each with one shim BD reading its
            # own slice of the shared in/out buffer. Each BD moves a chunk of
            # size `chunk_words` that covers all rows in the column (the
            # memtile splits it via object_fifo_link).
            last_out_tasks = []
            for c in range(cols):
                in_fifo = of_ins[c]
                out_fifo = of_outs[c]
                col_base = c * bds_per_task * chunk_words

                for k in range(bds_per_task):
                    issue_out = (k == bds_per_task - 1)
                    bd_offset = col_base + k * chunk_words
                    t_in = shim_dma_single_bd_task(
                        in_fifo, in_buf,
                        offset=bd_offset,
                        sizes=[1, 1, 1, chunk_words],
                        issue_token=False,
                    )
                    t_out = shim_dma_single_bd_task(
                        out_fifo, out_buf,
                        offset=bd_offset,
                        sizes=[1, 1, 1, chunk_words],
                        issue_token=issue_out,
                    )
                    dma_start_task(t_in)
                    dma_start_task(t_out)
                    if issue_out:
                        last_out_tasks.append(t_out)

            for t_out in last_out_tasks:
                dma_await_task(t_out)


def emit(mechanism, device_name, cols, rows_per_col, bds_per_task, topology, ab):
    dev_enum, max_cols = DEVICES[device_name]
    if cols > max_cols:
        sys.stderr.write(
            f"[generate.py] tiles={cols} exceeds device column count "
            f"{max_cols} for {device_name}\n"
        )
        sys.exit(2)
    if rows_per_col < 1 or rows_per_col > 4:
        sys.stderr.write(
            f"[generate.py] rows-per-col={rows_per_col} out of range [1,4]\n"
        )
        sys.exit(2)

    with mlir_mod_ctx() as ctx:
        if ab:
            _emit_device("cfg_a", dev_enum, cols, rows_per_col,
                         bds_per_task, topology, False)
            _emit_device("cfg_b", dev_enum, cols, rows_per_col,
                         bds_per_task, topology, False)

            total_words = LINE_LEN * bds_per_task * cols * rows_per_col

            @device(dev_enum, sym_name="ab_orch")
            def main_body():
                tensor_ty = np.ndarray[(total_words,), np.dtype[np.int32]]

                @runtime_sequence(tensor_ty, tensor_ty)
                def seq(in_buf, out_buf):
                    npu_load_pdi(device_ref="cfg_a")
                    npu_load_pdi(device_ref="cfg_b")
        else:
            with_self = mechanism in ("load_pdi_fw", "load_pdi_expanded")
            _emit_device("main", dev_enum, cols, rows_per_col,
                         bds_per_task, topology, with_self)

    print(ctx.module)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mechanism", required=True,
                   choices=["baseline", "load_pdi_fw",
                            "load_pdi_expanded", "ctrlpkt"])
    p.add_argument("--device", required=True, choices=list(DEVICES.keys()))
    p.add_argument("--tiles", type=int, required=True,
                   help="Number of shim columns to use (also the column-tile count).")
    p.add_argument("--rows-per-col", type=int, default=1,
                   help="Compute tiles per column (1..4). Whole-array = 4.")
    p.add_argument("--bds", type=int, required=True)
    p.add_argument("--topology", required=True,
                   choices=["linear", "branch", "hop"])
    p.add_argument("--ab", action="store_true",
                   help="Emit two-config orchestrator (load_pdi only).")
    args = p.parse_args()
    emit(args.mechanism, args.device, args.tiles, args.rows_per_col,
         args.bds, args.topology, args.ab)


if __name__ == "__main__":
    main()
