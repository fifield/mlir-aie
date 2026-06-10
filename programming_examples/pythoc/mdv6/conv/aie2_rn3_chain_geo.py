#!/usr/bin/env python3
"""Geometry-generic chained rn3-pair launch (re4 80x80x32, re8 20x20x64).

Same DDR-bounced halo chain as aie2_rn3_pair_vector_chain (re6), built on
the ic-parametric kernels. Differences:

* GEOS table drives tiles/columns/passes; 8x8 tiles, 12-wide patches always.
* Wide grids run PASSES column passes per iteration (NPU col c covers grid
  cols c*PASSES..c*PASSES+PASSES-1).
* Fills and drains ping-pong between two image BOs per iteration: pass p+1
  fills overlap pass p's drained rows within the same iteration, so x_i must
  stay intact while x_{i+1} is written. Even iters read BO A and drain BO B.
* Weight stream: 2*(ic/16) slots per pair, repeated for every (pass, tile).
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
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import KERNEL_EXTRA_GLOBALS, _MMUL_HELPERS
from kernels.rn3_chain_pythoc import (
    chain_conv1_bf16,
    chain_mask_bf16,
    chain_conv2res_bf16,
    _store_bn_silu_res_4x8_rows,
)

TILE = 8
PAD = 2

GEOS = {
    "re4": dict(IC=32, GBOUND=80, COLS=5, PASSES=2, WORKER_TILES=(3, 3, 3, 3)),
    "re8": dict(IC=64, GBOUND=20, COLS=3, PASSES=1, WORKER_TILES=(1, 1, 1)),
}


def geo_params(geo: str):
    g = GEOS[geo]
    ic = g["IC"]
    n_blk = ic // 16
    wslot = 16 * ic * 9 + 32
    tiles_per_col = sum(g["WORKER_TILES"])
    img_w = g["COLS"] * g["PASSES"] * TILE + 2 * PAD
    img_h = tiles_per_col * TILE + 2 * PAD
    return dict(g, N_BLK=n_blk, WSLOT=wslot, TILES_PER_COL=tiles_per_col,
                IMG=img_w, IMG_H=img_h, IMG_ELEMS=img_h * img_w * ic,
                FINAL=TILE * TILE * ic, SCRATCH=n_blk * 1600,
                GRID_ROWS=(g["GBOUND"] + TILE - 1) // TILE)


def rn3_chain_geo(geo: str, n_iters: int = 2, stack_size: int = 4096, compute: int = 1):
    p = geo_params(geo)
    IC, COLS, PASSES = p["IC"], p["COLS"], p["PASSES"]
    WORKER_TILES, N_BLK, WSLOT = p["WORKER_TILES"], p["N_BLK"], p["WSLOT"]
    IMG, IMG_H, IMG_ELEMS = p["IMG"], p["IMG_H"], p["IMG_ELEMS"]
    FINAL, SCRATCH, GRID_ROWS, GBOUND = p["FINAL"], p["SCRATCH"], p["GRID_ROWS"], p["GBOUND"]
    TPC = p["TILES_PER_COL"]
    NW = len(WORKER_TILES)
    NT = WORKER_TILES[0]
    CHUNK_ROWS = 8 * NT + 4
    CHUNK = CHUNK_ROWS * 12 * IC
    SLOTS_PER_PAIR = 2 * N_BLK

    dev = NPU2()
    patch_ty = np.ndarray[(CHUNK,), np.dtype[np.uint16]]
    final_ty = np.ndarray[(NT * FINAL,), np.dtype[np.uint16]]
    col_out_ty = np.ndarray[(TPC * FINAL,), np.dtype[np.uint16]]
    col_in_ty = np.ndarray[(NW * CHUNK,), np.dtype[np.uint16]]
    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(SCRATCH,), np.dtype[np.uint16]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    wt_host_ty = np.ndarray[(n_iters * NT * SLOTS_PER_PAIR * WSLOT,), np.dtype[np.uint16]]

    kc1 = PythocKernel(chain_conv1_bf16, [patch_ty, wslot_ty, scratch_ty, np.int32, np.int32, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=_MMUL_HELPERS)
    km = PythocKernel(chain_mask_bf16, [scratch_ty, np.int32, np.int32, np.int32, np.int32],
                      extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[])
    kc2r = PythocKernel(chain_conv2res_bf16, [scratch_ty, wslot_ty, final_ty, patch_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
                        extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[_store_bn_silu_res_4x8_rows])

    workers = []
    col_in, col_out, col_wt = [], [], []

    def core_fn(a, w, o, scratch, c1, m, c2r, iters, base_row, col):
        for it in range_(iters):
            ps = 0
            while ps < PASSES:
                ein = a.acquire(1)
                eout = o.acquire(1)
                gcol = (col * PASSES + ps) * 8
                t = 0
                while t < NT:
                    real = base_row + t < GRID_ROWS
                    mb = 0
                    while mb < N_BLK:
                        ew = w.acquire(1)
                        if real and compute:
                            c1(ein, ew, scratch, mb, t, IC)
                        w.release(1)
                        mb = mb + 1
                    if real and compute:
                        m(scratch, (base_row + t) * 8, gcol, GBOUND, N_BLK)
                    ob = 0
                    while ob < N_BLK:
                        ew = w.acquire(1)
                        if real and compute:
                            c2r(scratch, ew, eout, ein, ob, t, IC, (base_row + t) * 8, gcol, GBOUND)
                        w.release(1)
                        ob = ob + 1
                    t = t + 1
                a.release(1)
                o.release(1)
                ps = ps + 1

    for c in range(COLS):
        fin = ObjectFifo(col_in_ty, depth=1, name=f"cg_in_{c}")
        fout = ObjectFifo(col_out_ty, depth=1, name=f"cg_out_{c}")
        fwt = ObjectFifo(wslot_ty, depth=1, name=f"cg_wt_{c}")

        p_off = [w * CHUNK for w in range(NW)]
        f_off = [w * NT * FINAL for w in range(NW)]
        in_sp = fin.cons().split(offsets=p_off, obj_types=[patch_ty] * NW,
                                 depths=[1] * NW, names=[f"cg_in_{c}_{i}" for i in range(NW)])
        out_j = fout.prod().join(offsets=f_off, obj_types=[final_ty] * NW,
                                 depths=[1] * NW, names=[f"cg_out_{c}_{i}" for i in range(NW)])
        col_in.append(fin)
        col_out.append(fout)
        col_wt.append(fwt)
        for i in range(NW):
            scratch = Buffer(scratch_ty, name=f"cg_scr_{c}_{i}")
            workers.append(Worker(core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(),
                 scratch, kc1, km, kc2r, n_iters, i * NT, c],
                stack_size=stack_size))

    rt = Runtime()
    with rt.sequence(img_ty, wt_host_ty, img_ty) as (A, WT, B):
        rt.start(*workers)
        for it in range(n_iters):
            src, dst = (A, B) if it % 2 == 0 else (B, A)
            for ps in range(PASSES):
                tg = rt.task_group()
                for c in range(COLS):
                    gx = (c * PASSES + ps) * 8
                    rt.fill(col_in[c].prod(), src, TensorAccessPattern(
                        (IMG_ELEMS,), offset=gx * IC,
                        sizes=[NW, CHUNK_ROWS, 12, IC],
                        strides=[8 * NT * IMG * IC, IMG * IC, IC, 1]), task_group=tg)
                    # 2*N_BLK slots per pair, streamed once per tile pass
                    # (broadcast: every worker consumes the same elems)
                    # linear, host duplicates the slot block NT times per iter
                    rt.fill(col_wt[c].prod(), WT, TensorAccessPattern(
                        (n_iters * NT * SLOTS_PER_PAIR * WSLOT,), offset=it * NT * SLOTS_PER_PAIR * WSLOT,
                        sizes=[1, NT * SLOTS_PER_PAIR * WSLOT],
                        strides=[0, 1]), task_group=tg)
                    rt.drain(col_out[c].cons(), dst, TensorAccessPattern(
                        (IMG_ELEMS,), offset=(PAD * IMG + PAD + gx) * IC,
                        # leading size-1 dim: <4-dim shim S2MM BDs hang
                        sizes=[1, TPC, TILE, TILE * IC],
                        strides=[0, 8 * IMG * IC, IMG * IC, 1]), task_group=tg, wait=True)
                rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


if __name__ == "__main__":
    import os
    print(rn3_chain_geo(os.environ.get("GEO", "re8"), int(os.environ.get("N_ITERS", "2"))))
