#!/usr/bin/env python3
"""Chained rn3-pair iterations in ONE launch — DDR-bounced halo redistribution.

See docs/halo_chain_scope.md. Topology (re6, 40x40x48, 25 tiles):

* 5 NPU columns; column c owns image grid column c (tiles tr=0..4).
* 4 workers/col; worker 0 computes 2 tiles, the rest 1 each. All workers
  run the same 2-tile loop (skipping compute for absent tiles) so the
  broadcast weight FIFO is consumed uniformly: 12 slot acquires per
  worker per iteration (6 slots streamed twice).
* Patches fill from a padded 44x44x48 HWC image in DDR (zero borders);
  HWC finals drain back to the image interior. One fill + one drain per
  column per iteration — single 4D BDs.
* Per-worker scratch is an iron Buffer; finals are 3072-u16 FIFO elems.
* Residual: finals += patch center, i.e. out_{i+1} = pair(x_i) + x_i.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import KERNEL_EXTRA_GLOBALS, _MMUL_HELPERS
from kernels.rn3_chain_pythoc import (
    chain_conv1_bf16,
    chain_mask_bf16,
    chain_conv2_bf16,
    chain_residual_bf16,
)

IC = 48
TILE = 8
GRID = 5
PAD = 2
IMG = GRID * TILE + 2 * PAD          # 44 (width)
IMG_H = 8 * TILE + 2 * PAD           # 68 rows: tiles 5..7 are junk rows below the interior
PATCH = 12 * 12 * IC                 # 6912 u16
MASK = 100
FINAL = TILE * TILE * IC             # 3072 u16, HWC
SCRATCH = 4800
WSLOT = 16 * 48 * 9 + 32             # 6944 u16
N_WSLOTS = 6                         # per tile-iteration (3 conv1 + 3 conv2)
IMG_ELEMS = IMG_H * IMG * IC

TILES_PER_COL = 8
WORKER_TILES = (2, 2, 2, 2)


def rn3_pair_vector_chain(dev=None, n_iters: int = 2, stack_size: int = 4096, n_cols: int = GRID, stages: int = 4, linear: int = 0):
    dev = NPU2() if dev is None else dev

    def patch_ty(n):
        # combined patch for n vertically-adjacent tiles: (8n+4) rows x 12 x 48
        return np.ndarray[((8 * n + 4) * 12 * IC,), np.dtype[np.uint16]]

    def mask_ty(n):
        return np.ndarray[(n * MASK,), np.dtype[np.uint16]]

    def final_ty(n):
        return np.ndarray[(n * FINAL,), np.dtype[np.uint16]]

    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(SCRATCH,), np.dtype[np.uint16]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    wt_host_ty = np.ndarray[(n_iters * 2 * N_WSLOTS * WSLOT,), np.dtype[np.uint16]]

    def kernels_for(nt):
        return (
            PythocKernel(chain_conv1_bf16, [patch_ty(nt), wslot_ty, scratch_ty, np.int32, np.int32],
                         extra_globals=KERNEL_EXTRA_GLOBALS, helpers=_MMUL_HELPERS),
            PythocKernel(chain_mask_bf16, [scratch_ty, np.int32, np.int32],
                         extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[]),
            PythocKernel(chain_conv2_bf16, [scratch_ty, wslot_ty, final_ty(nt), np.int32, np.int32],
                         extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[]),
            PythocKernel(chain_residual_bf16, [patch_ty(nt), wslot_ty, final_ty(nt), np.int32, np.int32],
                         extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[]),
        )

    kernel_sets = {2: kernels_for(2)}

    workers = []
    col_in, col_out, col_wt = [], [], []

    def core_fn(a, w, o, scratch, kc1, km, kc2, kr, n_tiles, iters, base_row, gcol):
        it = 0
        while it < iters:
            ein = a.acquire(1)
            eout = o.acquire(1)
            t = 0
            while t < 2:
                mb = 0
                while mb < 3:
                    ew = w.acquire(1)
                    if t < n_tiles and stages >= 1 and stages < 5:
                        kc1(ein, ew, scratch, mb, t)
                    w.release(1)
                    mb = mb + 1
                if t < n_tiles and stages >= 2 and stages < 5:
                    km(scratch, (base_row + t) * 8, gcol)
                ob = 0
                while ob < 3:
                    ew = w.acquire(1)
                    if t < n_tiles and stages >= 3 and stages < 5:
                        kc2(scratch, ew, eout, ob, t)
                    if t < n_tiles and stages >= 4 and ob == 2:
                        kr(ein, ew, eout, 0, t)
                    w.release(1)
                    ob = ob + 1
                t = t + 1
            a.release(1)
            o.release(1)
            it = it + 1

    for c in range(n_cols):
        fin = ObjectFifo(np.ndarray[(4 * 20 * 12 * IC,), np.dtype[np.uint16]], depth=1, name=f"ch_in_{c}")
        fout = ObjectFifo(final_ty(TILES_PER_COL), depth=1, name=f"ch_out_{c}")
        fwt = ObjectFifo(wslot_ty, depth=1, name=f"ch_wt_{c}")

        p_off, f_off, base_rows = [], [], []
        acc = 0
        for nt in WORKER_TILES:
            p_off.append((acc // 2) * 20 * 12 * IC)
            f_off.append(acc * FINAL)
            base_rows.append(acc)
            acc += nt
        in_sp = fin.cons().split(offsets=p_off, obj_types=[patch_ty(n) for n in WORKER_TILES],
                                 depths=[1] * 4, names=[f"ch_in_{c}_{i}" for i in range(4)])
        out_j = fout.prod().join(offsets=f_off, obj_types=[final_ty(n) for n in WORKER_TILES],
                                 depths=[1] * 4, names=[f"ch_out_{c}_{i}" for i in range(4)])

        col_in.append(fin)
        col_out.append(fout)
        col_wt.append(fwt)

        for i, nt in enumerate(WORKER_TILES):
            scratch = Buffer(scratch_ty, name=f"ch_scr_{c}_{i}")
            kc1, km, kc2, kr = kernel_sets[nt]
            workers.append(Worker(
                core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(),
                 scratch, kc1, km, kc2, kr, nt, n_iters, base_rows[i], 8 * c],
                stack_size=stack_size,
            ))

    rt = Runtime()
    with rt.sequence(img_ty, wt_host_ty) as (IMGB, WT):
        rt.start(*workers)
        for it in range(n_iters):
            tg = rt.task_group()
            for c in range(n_cols):
                in_tap = (TensorAccessPattern((IMG_ELEMS,), offset=0,
                          sizes=[1, 4 * 20 * 12 * IC], strides=[0, 1]) if linear else
                          TensorAccessPattern((IMG_ELEMS,), offset=8 * c * IC,
                          sizes=[TILES_PER_COL // 2, 20, 12, IC],
                          strides=[16 * IMG * IC, IMG * IC, IC, 1]))
                rt.fill(col_in[c].prod(), IMGB, in_tap, task_group=tg)
                # weights: 6 slots streamed twice (both worker tile passes)
                rt.fill(col_wt[c].prod(), WT, TensorAccessPattern(
                    (n_iters * 2 * N_WSLOTS * WSLOT,), offset=it * 2 * N_WSLOTS * WSLOT,
                    sizes=[2 * N_WSLOTS, WSLOT], strides=[WSLOT, 1]), task_group=tg)
                out_tap = (TensorAccessPattern((IMG_ELEMS,), offset=0,
                           sizes=[1, TILES_PER_COL * TILE * TILE * IC], strides=[0, 1]) if linear else
                           TensorAccessPattern((IMG_ELEMS,), offset=(PAD * IMG + PAD + 8 * c) * IC,
                           # leading size-1 dim: shim S2MM BDs with <4 dims hang
                           # (see test_dualtap_micro_hw.py)
                           sizes=[1, TILES_PER_COL, TILE, TILE * IC],
                           strides=[0, 8 * IMG * IC, IMG * IC, 1]))
                rt.drain(col_out[c].cons(), IMGB, out_tap, task_group=tg, wait=True)
            rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


if __name__ == "__main__":
    print(rn3_pair_vector_chain())
