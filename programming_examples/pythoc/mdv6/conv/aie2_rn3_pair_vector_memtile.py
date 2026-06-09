#!/usr/bin/env python3
"""MemTile column-fanout vector rn3-pair prototype.

This is the next recoverable-regimes topology after direct shim-fed lanes hit
ShimNOCTile DMA capacity at 16 lanes.

Topology:

* one column input FIFO receives a packed block for up to four workers;
* the input FIFO splits to per-worker arena objects;
* one column output FIFO joins per-worker arena outputs;
* one per-column weight FIFO broadcasts the same 12-slot rn3-pair weight stream
  to the workers in that column;
* each worker computes one 8x8 rn3-pair output patch using the specialized
  vector bf16 `rn3_pair_vector_stage_bf16` kernel.

This mirrors the proven `aie2_multicore_ocb.py` split/join pattern while keeping
the math kernel specialized. It intentionally avoids repeated output drains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import (  # noqa: E402
    KERNEL_EXTRA_GLOBALS,
    _MMUL_HELPERS,
    rn3_pair_vector_stage_bf16,
)
from conv.aie2_rn3_pair_vector_ocb import (  # noqa: E402
    ARENA_SIZE,
    FINAL_OFFSET,
    FINAL_SIZE,
    IC,
    INPUT_SIZE,
    MASK_OFFSET,
    MASK_SIZE,
    MID_BLOCK,
    N_MID_BLOCKS,
    N_OC_BLOCKS,
    N_WEIGHT_SLOTS,
    OC_BLOCK,
    SCRATCH_SIZE,
    TILE_H,
    TILE_W,
    W1_SIZE,
    W2_SIZE,
    WEIGHT_SLOT_SIZE,
)


def rn3_pair_vector_memtile(
    dev=None,
    n_cores: int = 16,
    stack_size: int = 4096,
):
    """Return resolved IRON program for one-patch-per-worker memtile fanout."""
    if n_cores <= 0 or n_cores > 32:
        raise ValueError("n_cores must be in 1..32")
    dev = NPU2() if dev is None else dev
    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    arena_ty = np.ndarray[(ARENA_SIZE,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    host_input_ty = np.ndarray[(n_cores * ARENA_SIZE,), np.dtype[np.uint16]]
    host_weight_ty = np.ndarray[(N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(n_cores * ARENA_SIZE,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        rn3_pair_vector_stage_bf16,
        [arena_ty, weight_ty, arena_ty, np.int32, np.int32, np.int32],
        extra_globals=KERNEL_EXTRA_GLOBALS,
        helpers=_MMUL_HELPERS,
    )

    workers = []
    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []

    def core_fn(a, w, c, kern):
        ein = a.acquire(1)
        eout = c.acquire(1)

        mb = 0
        while mb < 3:
            ew = w.acquire(1)
            kern(ein, ew, eout, 0, mb, 0)
            w.release(1)
            mb = mb + 1
        a.release(1)

        ob = 0
        while ob < 3:
            ew = w.acquire(1)
            kern(eout, ew, eout, 1, ob, 0)
            w.release(1)
            ob = ob + 1
        c.release(1)

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_size = cores_this_col * ARENA_SIZE
        col_out_size = cores_this_col * ARENA_SIZE
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        col_in = ObjectFifo(col_in_ty, depth=1, name=f"rn3vm_col_in_{col}")
        in_splits = col_in.cons().split(
            offsets=[ARENA_SIZE * i for i in range(cores_this_col)],
            obj_types=[arena_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"rn3vm_in_{col}_{i}" for i in range(cores_this_col)],
        )
        col_out = ObjectFifo(col_out_ty, depth=1, name=f"rn3vm_col_out_{col}")
        out_joins = col_out.prod().join(
            offsets=[ARENA_SIZE * i for i in range(cores_this_col)],
            obj_types=[arena_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"rn3vm_out_{col}_{i}" for i in range(cores_this_col)],
        )
        wt = ObjectFifo(weight_ty, depth=1, name=f"rn3vm_wt_{col}")

        col_in_fifos.append(col_in)
        col_out_fifos.append(col_out)
        wt_fifos.append(wt)

        for i in range(cores_this_col):
            workers.append(
                Worker(
                    core_fn,
                    [in_splits[i].cons(), wt.cons(), out_joins[i].prod(), kernel],
                    stack_size=stack_size,
                )
            )

    rt = Runtime()
    with rt.sequence(host_input_ty, host_weight_ty, host_output_ty) as (I, W, O):
        rt.start(*workers)
        tg = rt.task_group()
        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            col_size = cores_this_col * ARENA_SIZE
            col_offset = col * cores_per_col * ARENA_SIZE
            rt.fill(
                col_in_fifos[col].prod(),
                I,
                TensorAccessPattern(
                    (n_cores * ARENA_SIZE,),
                    offset=col_offset,
                    sizes=[1, col_size],
                    strides=[0, 1],
                ),
                task_group=tg,
            )
            rt.fill(
                wt_fifos[col].prod(),
                W,
                TensorAccessPattern(
                    (N_WEIGHT_SLOTS * WEIGHT_SLOT_SIZE,),
                    offset=0,
                    sizes=[N_WEIGHT_SLOTS, WEIGHT_SLOT_SIZE],
                    strides=[WEIGHT_SLOT_SIZE, 1],
                ),
                task_group=tg,
            )
            rt.drain(
                col_out_fifos[col].cons(),
                O,
                TensorAccessPattern(
                    (n_cores * ARENA_SIZE,),
                    offset=col_offset,
                    sizes=[1, col_size],
                    strides=[0, 1],
                ),
                task_group=tg,
                wait=True,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-cores", type=int, default=16)
    p.add_argument("--stack-size", type=int, default=4096)
    args = p.parse_args(argv)
    print(rn3_pair_vector_memtile(n_cores=args.n_cores, stack_size=args.stack_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
