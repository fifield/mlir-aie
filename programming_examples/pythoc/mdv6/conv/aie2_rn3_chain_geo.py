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
from aie.iron.device import NPU2, Tile
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.rep_elan_bf16_pythoc import KERNEL_EXTRA_GLOBALS, _MMUL_HELPERS
from kernels.rn3_chain_pythoc import (
    chain_wt_arm,
    chain_wt_arm_nq,
    chain_wt_arm_tok,
    chain_wt_wait,
    chain_conv1_bf16,
    chain_mask_bf16,
    chain_conv2res_bf16,
    chain_copy_bf16,
    chain_gemm_bf16,
    _store_bn_silu_res_4x8_rows,
)

TILE = 8
PAD = 2

GEOS = {
    "re4": dict(IC=32, GBOUND=80, COLS=5, PASSES=2, WORKER_TILES=(3, 3, 3, 3)),
    "re6": dict(IC=48, GBOUND=40, COLS=5, PASSES=1, WORKER_TILES=(2, 2, 2)),
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
    # rnm epilogue: x2 lives in the SAME image BO below the chain region
    # (shims have only 2 MM2S channels — a separate x2 BO needs a third)
    x2_row0 = img_h
    # epilogue fills read full 20-row chunks per tile; the last tile's chunk
    # starts at x2_row0 + PAD + 8*(tiles-1) and extends 20 rows. The rnm
    # output region follows: it lives in the SAME image BOs because shims
    # only have 2 S2MM channels (drains to a third BO won't allocate).
    img_h_rnm = x2_row0 + PAD + 8 * (tiles_per_col - 1) + 20
    out_off = img_h_rnm * img_w * ic
    out_elems = tiles_per_col * TILE * g["GBOUND"] * 2 * ic
    return dict(g, N_BLK=n_blk, WSLOT=wslot, TILES_PER_COL=tiles_per_col,
                IMG=img_w, IMG_H=img_h, IMG_ELEMS=img_h * img_w * ic,
                X2_ROW0=x2_row0, IMG_H_RNM=img_h_rnm,
                OUT_OFF=out_off, OUT_ELEMS=out_elems,
                IMG_ELEMS_RNM=out_off + out_elems,
                FINAL=TILE * TILE * ic, SCRATCH=n_blk * 1600,
                GRID_ROWS=(g["GBOUND"] + TILE - 1) // TILE)


def rn3_chain_geo(geo: str, n_iters: int = 2, stack_size: int = 4096, compute: int = 1,
                  rnm: int = 0):
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

    # rnm epilogue: 1x1 GEMM concat(cur, x2) -> 2*IC channels in EPASS
    # oc-passes of IC each. Each epilogue FIFO element = cur 8x12xIC +
    # x2 8x12xIC + pad to one chain CHUNK.
    EPASS = 2
    OC2 = 2 * IC
    EPI_W0 = n_iters * NT * SLOTS_PER_PAIR * WSLOT
    EPI_WT = EPASS * NT * N_BLK * WSLOT
    OUT_OFF = p["OUT_OFF"]
    wt_elems = EPI_W0 + (EPI_WT if rnm else 0)
    X2_ROW0 = p["X2_ROW0"]
    img_elems = p["IMG_ELEMS_RNM"] if rnm else IMG_ELEMS
    if rnm:
        # one x2 tile parked in scratch during the epilogue (8 rows x 12 x IC);
        # chain conv planes are dead by then, so no L1 growth
        SCRATCH = max(SCRATCH, 96 * IC)

    dev = NPU2()
    patch_ty = np.ndarray[(CHUNK,), np.dtype[np.uint16]]
    final_ty = np.ndarray[(NT * FINAL,), np.dtype[np.uint16]]
    col_out_ty = np.ndarray[(TPC * FINAL,), np.dtype[np.uint16]]
    col_in_ty = np.ndarray[(NW * CHUNK,), np.dtype[np.uint16]]
    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(SCRATCH,), np.dtype[np.uint16]]
    img_ty = np.ndarray[(img_elems,), np.dtype[np.uint16]]
    wt_host_ty = np.ndarray[(wt_elems,), np.dtype[np.uint16]]

    kc1 = PythocKernel(chain_conv1_bf16, [patch_ty, wslot_ty, scratch_ty, np.int32, np.int32, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=_MMUL_HELPERS)
    km = PythocKernel(chain_mask_bf16, [scratch_ty, np.int32, np.int32, np.int32, np.int32],
                      extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[])
    kc2r = PythocKernel(chain_conv2res_bf16, [scratch_ty, wslot_ty, final_ty, patch_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
                        extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[_store_bn_silu_res_4x8_rows])
    # helpers come from kc1's _MMUL_HELPERS (same core program) — listing
    # _store_bn_silu_4x8_rows again would redefine the symbol
    kgm = PythocKernel(chain_gemm_bf16, [patch_ty, scratch_ty, wslot_ty, final_ty, np.int32, np.int32, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[]) if rnm else None
    kcp = PythocKernel(chain_copy_bf16, [patch_ty, scratch_ty, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[]) if rnm else None

    workers = []
    col_in, col_out, col_wt = [], [], []

    def core_fn(a, w, o, scratch, c1, m, c2r, gm, cp, iters, base_row, col):
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
        if rnm:
            # rnm epilogue: per spatial pass, EPASS oc-passes of a 1x1 GEMM
            # over concat(chain output, x2). One patch element per tile (cur),
            # x2 ships through the wt FIFO (one elem per tile, then N_BLK
            # weight slots — wt FIFO depth 2 holds x2 + slot together).
            eps = 0
            while eps < PASSES:
                pe = 0
                while pe < EPASS:
                    eo2 = o.acquire(1)
                    t2 = 0
                    while t2 < NT:
                        real2 = base_row + t2 < GRID_ROWS
                        # park this tile's x2 in scratch, then stream cur
                        ex2 = a.acquire(1)
                        if compute:
                            cp(ex2, scratch, IC)
                        a.release(1)
                        ei2 = a.acquire(1)
                        s2 = 0
                        while s2 < N_BLK:
                            ew2 = w.acquire(1)
                            if real2 and compute:
                                gm(ei2, scratch, ew2, eo2, t2, s2, IC)
                            w.release(1)
                            s2 = s2 + 1
                        a.release(1)
                        t2 = t2 + 1
                    o.release(1)
                    pe = pe + 1
                eps = eps + 1

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
                 scratch, kc1, km, kc2r, kgm if rnm else kc1, kcp if rnm else kc1,
                 n_iters, i * NT, c],
                # pin one chain col per device col — the default placer packs
                # workers 4-per-column, doubling FIFOs on shared shims
                tile=Tile(c, 2 + i),
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
                        (img_elems,), offset=gx * IC,
                        sizes=[NW, CHUNK_ROWS, 12, IC],
                        strides=[8 * NT * IMG * IC, IMG * IC, IC, 1]), task_group=tg)
                    # 2*N_BLK slots per pair, streamed once per tile pass
                    # (broadcast: every worker consumes the same elems)
                    # linear, host duplicates the slot block NT times per iter
                    rt.fill(col_wt[c].prod(), WT, TensorAccessPattern(
                        (wt_elems,), offset=it * NT * SLOTS_PER_PAIR * WSLOT,
                        sizes=[1, NT * SLOTS_PER_PAIR * WSLOT],
                        strides=[0, 1]), task_group=tg)
                    rt.drain(col_out[c].cons(), dst, TensorAccessPattern(
                        (img_elems,), offset=(PAD * IMG + PAD + gx) * IC,
                        # leading size-1 dim: <4-dim shim S2MM BDs hang
                        sizes=[1, TPC, TILE, TILE * IC],
                        strides=[0, 8 * IMG * IC, IMG * IC, 1]), task_group=tg, wait=True)
                rt.finish_task_group(tg)

        if rnm:
            # Epilogue: rnm 1x1 GEMM on concat(final chain image, x2).
            # x2 lives in BOTH image BOs below the chain region (rows X2_ROW0+)
            # so every epilogue fill sources the final image BO — a separate
            # x2 BO would need a third shim MM2S channel which doesn't exist.
            srcF = B if n_iters % 2 == 1 else A
            for ps in range(PASSES):
                for pe in range(EPASS):
                    tg = rt.task_group()
                    for c in range(COLS):
                        gx = (c * PASSES + ps) * 8
                        # per round r: x2 tile r of every worker, then cur tile
                        # r — matches the per-tile park-then-stream consumption.
                        # Full 20-row chunks; rows past 8 are junk never read.
                        for r in range(NT):
                            rt.fill(col_in[c].prod(), srcF, TensorAccessPattern(
                                (img_elems,),
                                offset=((X2_ROW0 + PAD + r * 8) * IMG + PAD + gx - 2) * IC,
                                sizes=[1, NW, CHUNK_ROWS, 12 * IC],
                                strides=[0, 8 * NT * IMG * IC, IMG * IC, 1]), task_group=tg)
                            rt.fill(col_in[c].prod(), srcF, TensorAccessPattern(
                                (img_elems,),
                                offset=((PAD + r * 8) * IMG + PAD + gx - 2) * IC,
                                sizes=[1, NW, CHUNK_ROWS, 12 * IC],
                                strides=[0, 8 * NT * IMG * IC, IMG * IC, 1]), task_group=tg)
                        rt.fill(col_wt[c].prod(), WT, TensorAccessPattern(
                            (wt_elems,), offset=EPI_W0 + pe * NT * N_BLK * WSLOT,
                            sizes=[1, NT * N_BLK * WSLOT],
                            strides=[0, 1]), task_group=tg)
                        # one drain per tile: shim S2MM has 3 real dims (the
                        # 4th must stay leading size-1); pixels land OC2 apart
                        for ti in range(TPC):
                            rt.drain(col_out[c].cons(), srcF, TensorAccessPattern(
                                (img_elems,),
                                offset=OUT_OFF + ((ti * TILE * GBOUND) + gx) * OC2 + pe * IC,
                                sizes=[1, TILE, TILE, IC],
                                strides=[0, GBOUND * OC2, OC2, 1]),
                                task_group=tg, wait=(ti == TPC - 1))
                    rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


RASTER_GEOS = {
    # raster: TPR tiles per core, tile idx = (col*NWORK+w)*TPR + r over GRID^2.
    "re6w": dict(IC=48, GBOUND=40, COLS=7, NWORK=4, TPR=1),  # DEPTH2 doesn't fit L1 at IC48
    # DEPTH=2 measured wall-flat (8.20 vs 8.00 ms): per-round drain waits set
    # the pace, and removing them wedges (BD reuse before drains complete).
    # Chain pace is wt slot DMA (~83 KB/col/round vs 14 KB patches).
    "re4w": dict(IC=32, GBOUND=80, COLS=7, NWORK=4, TPR=4),
    # 1-col bisect: 8x8 grid covered by 4 workers x 16 rounds
    "re1w": dict(IC=32, GBOUND=8, COLS=1, NWORK=1, TPR=1),
}


def raster_params(geo: str):
    g = RASTER_GEOS[geo]
    ic = g["IC"]
    grid = (g["GBOUND"] + TILE - 1) // TILE
    img_w = grid * TILE + 2 * PAD
    junk = g["COLS"] * g["NWORK"] * g["TPR"] - grid * grid
    band_rows = ((junk + grid - 1) // grid) * TILE
    img_h = img_w + band_rows  # junk drain band below the padded image
    return dict(g, N_BLK=ic // 16, WSLOT=16 * ic * 9 + 32, GRID=grid,
                IMG=img_w, IMG_H=img_h, IMG_ELEMS=img_h * img_w * ic,
                FINAL=TILE * TILE * ic, SCRATCH=(ic // 16) * 1600)


def rn3_chain_raster(geo: str, n_iters: int = 2, stack_size: int = 4096, compute: int = 1):
    """TPR tiles per core: tile idx = (col*NWORK + w)*TPR + r rastered over the
    GRIDxGRID image. vs column-major chain: per-core load drops to TPR tiles
    (re6w 1 vs 2, re4w 4 vs 6) and 25-28 cores run instead of 15-20."""
    p = raster_params(geo)
    IC, COLS, NWORK, TPR = p["IC"], p["COLS"], p["NWORK"], p["TPR"]
    GRID, IMG, IMG_ELEMS = p["GRID"], p["IMG"], p["IMG_ELEMS"]
    FINAL, SCRATCH, N_BLK, WSLOT = p["FINAL"], p["SCRATCH"], p["N_BLK"], p["WSLOT"]
    GBOUND = p["GBOUND"]
    DEPTH = p.get("DEPTH", 1)
    CHUNK = 12 * 12 * IC
    SLOTS_PER_PAIR = 2 * N_BLK
    JUNK_ROW = GRID * TILE + 2 * PAD
    N_TILES = GRID * GRID

    def tile_of(c, w, r):
        idx = (c * NWORK + w) * TPR + r
        if idx < N_TILES:
            return True, (idx // GRID) * TILE, (idx % GRID) * TILE
        k = idx - N_TILES
        return False, JUNK_ROW + (k // GRID) * TILE - PAD, (k % GRID) * TILE - PAD

    dev = NPU2()
    patch_ty = np.ndarray[(CHUNK,), np.dtype[np.uint16]]
    final_ty = np.ndarray[(FINAL,), np.dtype[np.uint16]]
    col_out_ty = np.ndarray[(NWORK * FINAL,), np.dtype[np.uint16]]
    col_in_ty = np.ndarray[(NWORK * CHUNK,), np.dtype[np.uint16]]
    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(SCRATCH,), np.dtype[np.uint16]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    wt_host_ty = np.ndarray[(n_iters * TPR * SLOTS_PER_PAIR * WSLOT,), np.dtype[np.uint16]]

    kc1 = PythocKernel(chain_conv1_bf16, [patch_ty, wslot_ty, scratch_ty, np.int32, np.int32, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=_MMUL_HELPERS)
    km = PythocKernel(chain_mask_bf16, [scratch_ty, np.int32, np.int32, np.int32, np.int32],
                      extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[])
    kc2r = PythocKernel(chain_conv2res_bf16, [scratch_ty, wslot_ty, final_ty, patch_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
                        extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[_store_bn_silu_res_4x8_rows])

    workers, col_in, col_out, col_wt = [], [], [], []

    def core_fn(a, w, o, scratch, c1, m, c2r, iters, coords):
        for it in range_(iters):
            for (real, grow, gcol) in coords:
                ein = a.acquire(1)
                eout = o.acquire(1)
                mb = 0
                while mb < N_BLK:
                    ew = w.acquire(1)
                    if real and compute:
                        c1(ein, ew, scratch, mb, 0, IC)
                    w.release(1)
                    mb = mb + 1
                if real and compute:
                    m(scratch, grow, gcol, GBOUND, N_BLK)
                ob = 0
                while ob < N_BLK:
                    ew = w.acquire(1)
                    if real and compute:
                        c2r(scratch, ew, eout, ein, ob, 0, IC, grow, gcol, GBOUND)
                    w.release(1)
                    ob = ob + 1
                a.release(1)
                o.release(1)

    for c in range(COLS):
        fin = ObjectFifo(col_in_ty, depth=DEPTH, name=f"cr_in_{c}")
        fout = ObjectFifo(col_out_ty, depth=DEPTH, name=f"cr_out_{c}")
        fwt = ObjectFifo(wslot_ty, depth=DEPTH, name=f"cr_wt_{c}")
        in_sp = fin.cons().split(offsets=[w * CHUNK for w in range(NWORK)],
                                 obj_types=[patch_ty] * NWORK, depths=[DEPTH] * NWORK,
                                 names=[f"cr_in_{c}_{i}" for i in range(NWORK)])
        out_j = fout.prod().join(offsets=[w * FINAL for w in range(NWORK)],
                                 obj_types=[final_ty] * NWORK, depths=[DEPTH] * NWORK,
                                 names=[f"cr_out_{c}_{i}" for i in range(NWORK)])
        col_in.append(fin); col_out.append(fout); col_wt.append(fwt)
        for i in range(NWORK):
            coords = tuple(tile_of(c, i, r) for r in range(TPR))
            scratch = Buffer(scratch_ty, name=f"cr_scr_{c}_{i}")
            workers.append(Worker(core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(),
                 scratch, kc1, km, kc2r, n_iters, coords],
                tile=Tile(c, 2 + i), stack_size=stack_size))

    rt = Runtime()
    with rt.sequence(img_ty, wt_host_ty, img_ty) as (A, WT, B):
        rt.start(*workers)
        for it in range(n_iters):
            src, dst = (A, B) if it % 2 == 0 else (B, A)
            for r in range(TPR):
                tg = rt.task_group()
                for c in range(COLS):
                    for w in range(NWORK):
                        _, gr, gc = tile_of(c, w, r)
                        rt.fill(col_in[c].prod(), src, TensorAccessPattern(
                            (IMG_ELEMS,), offset=(gr * IMG + gc) * IC,
                            sizes=[1, 12, 12 * IC],
                            strides=[0, IMG * IC, 1]), task_group=tg)
                    rt.fill(col_wt[c].prod(), WT, TensorAccessPattern(
                        (n_iters * TPR * SLOTS_PER_PAIR * WSLOT,),
                        offset=(it * TPR + r) * SLOTS_PER_PAIR * WSLOT,
                        sizes=[1, SLOTS_PER_PAIR * WSLOT], strides=[0, 1]), task_group=tg)
                    for w in range(NWORK):
                        _, gr, gc = tile_of(c, w, r)
                        rt.drain(col_out[c].cons(), dst, TensorAccessPattern(
                            (IMG_ELEMS,), offset=((PAD + gr) * IMG + PAD + gc) * IC,
                            sizes=[1, 1, TILE, TILE * IC],
                            strides=[0, 0, IMG * IC, 1]), task_group=tg,
                            wait=(c == COLS - 1 and w == NWORK - 1))
                rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


def rn3_chain_raster_wr(geo: str, n_iters: int = 2, stack_size: int = 4096, compute: int = 1,
                        static_wt: int = 0):
    """Raster chain with memtile weight replay: slots fill the memtile once per
    iter (shim MM2S ch1), a raw memtile MM2S ch1 BD replays them TPR times to
    the column's cores; cores arm their own S2MM ch1 per slot into a fixed
    L1 buffer (no wt ObjectFifo, no host TPR duplication)."""
    from kernels.rn3_chain_pythoc import WT_BUF_ADDR
    p = raster_params(geo)
    IC, COLS, NWORK, TPR = p["IC"], p["COLS"], p["NWORK"], p["TPR"]
    GRID, IMG, IMG_ELEMS = p["GRID"], p["IMG"], p["IMG_ELEMS"]
    FINAL, SCRATCH, N_BLK, WSLOT = p["FINAL"], p["SCRATCH"], p["N_BLK"], p["WSLOT"]
    GBOUND = p["GBOUND"]
    CHUNK = 12 * 12 * IC
    SLOTS_PER_PAIR = 2 * N_BLK
    JUNK_ROW = GRID * TILE + 2 * PAD
    N_TILES = GRID * GRID
    MEM_STREAM = SLOTS_PER_PAIR * WSLOT          # u16 per iter, memtile resident
    SLOT_I32 = WSLOT // 2

    def tile_of(c, w, r):
        idx = (c * NWORK + w) * TPR + r
        if idx < N_TILES:
            return True, (idx // GRID) * TILE, (idx % GRID) * TILE
        k = idx - N_TILES
        return False, JUNK_ROW + (k // GRID) * TILE - PAD, (k % GRID) * TILE - PAD

    dev = NPU2()
    patch_ty = np.ndarray[(CHUNK,), np.dtype[np.uint16]]
    final_ty = np.ndarray[(FINAL,), np.dtype[np.uint16]]
    col_out_ty = np.ndarray[(NWORK * FINAL,), np.dtype[np.uint16]]
    col_in_ty = np.ndarray[(NWORK * CHUNK,), np.dtype[np.uint16]]
    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    scratch_ty = np.ndarray[(SCRATCH,), np.dtype[np.uint16]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    wt_host_ty = np.ndarray[(n_iters * MEM_STREAM,), np.dtype[np.uint16]]

    kc1 = PythocKernel(chain_conv1_bf16, [patch_ty, wslot_ty, scratch_ty, np.int32, np.int32, np.int32],
                       extra_globals=KERNEL_EXTRA_GLOBALS, helpers=_MMUL_HELPERS)
    km = PythocKernel(chain_mask_bf16, [scratch_ty, np.int32, np.int32, np.int32, np.int32],
                      extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[])
    kc2r = PythocKernel(chain_conv2res_bf16, [scratch_ty, wslot_ty, final_ty, patch_ty, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
                        extra_globals=KERNEL_EXTRA_GLOBALS, helpers=[_store_bn_silu_res_4x8_rows])
    from kernels.rn3_chain_pythoc import WT_BD, WT_LOCK, DMA_BD_BASE, DMA_S2MM_1_START_QUEUE
    wt_globals = dict(KERNEL_EXTRA_GLOBALS,
                      WT_BD=WT_BD, WT_LOCK=WT_LOCK, WT_BUF_ADDR=WT_BUF_ADDR,
                      WT_SLOT_I32=SLOT_I32,
                      DMA_BD_BASE=DMA_BD_BASE,
                      DMA_S2MM_1_START_QUEUE=DMA_S2MM_1_START_QUEUE)
    from kernels.rn3_chain_pythoc import TOK_ADDR, DMA_MM2S_1_START_QUEUE
    wt_globals["TOK_ADDR"] = TOK_ADDR
    wt_globals["DMA_MM2S_1_START_QUEUE"] = DMA_MM2S_1_START_QUEUE
    karm = PythocKernel(chain_wt_arm, [], extra_globals=wt_globals, helpers=[])
    kwait = PythocKernel(chain_wt_wait, [], extra_globals=wt_globals, helpers=[])

    workers, col_in, col_out, wbufs = [], [], [], []

    def core_fn(a, o, wbuf, scratch, c1, m, c2r, arm, wait, pkt_id, iters, coords):
        # queue a full round of receives before any data flows: the stream
        # must never park (parked beats head-of-line block chain fills)
        arm()
        arm()
        arm()
        arm()
        for it in range_(iters):
            for (real, grow, gcol) in coords:
                ein = a.acquire(1)
                eout = o.acquire(1)
                mb = 0
                while mb < N_BLK:
                    if compute:
                        wait()
                        arm()
                        if real and compute == 1:
                            c1(ein, wbuf, scratch, mb, 0, IC)
                    mb = mb + 1
                if real and compute == 1:
                    m(scratch, grow, gcol, GBOUND, N_BLK)
                ob = 0
                while ob < N_BLK:
                    if compute:
                        wait()
                        arm()
                        if real and compute == 1:
                            c2r(scratch, wbuf, eout, ein, ob, 0, IC, grow, gcol, GBOUND)
                    ob = ob + 1
                a.release(1)
                o.release(1)

    for c in range(COLS):
        fin = ObjectFifo(col_in_ty, depth=1, name=f"cw_in_{c}")
        fout = ObjectFifo(col_out_ty, depth=1, name=f"cw_out_{c}")
        in_sp = fin.cons().split(offsets=[w * CHUNK for w in range(NWORK)],
                                 obj_types=[patch_ty] * NWORK, depths=[1] * NWORK,
                                 names=[f"cw_in_{c}_{i}" for i in range(NWORK)])
        out_j = fout.prod().join(offsets=[w * FINAL for w in range(NWORK)],
                                 obj_types=[final_ty] * NWORK, depths=[1] * NWORK,
                                 names=[f"cw_out_{c}_{i}" for i in range(NWORK)])
        col_in.append(fin); col_out.append(fout)
        PKT_IDS = (1, 8, 12, 13)
        for i in range(NWORK):
            coords = tuple(tile_of(c, i, r) for r in range(TPR))
            scratch = Buffer(scratch_ty, name=f"cw_scr_{c}_{i}")
            wbuf = Buffer(wslot_ty, name=f"cw_wt_{c}_{i}")
            wbufs.append((c, i, f"cw_wt_{c}_{i}"))
            workers.append(Worker(core_fn,
                [in_sp[i].cons(), out_j[i].prod(), wbuf,
                 scratch, kc1, km, kc2r, karm, kwait, PKT_IDS[i], n_iters, coords],
                tile=Tile(c, 2 + i), stack_size=stack_size))

    rt = Runtime()
    with rt.sequence(img_ty, wt_host_ty, img_ty) as (A, WT, B):
        rt.start(*workers)
        for it in range(n_iters):
            src, dst = (A, B) if it % 2 == 0 else (B, A)
            for r in range(TPR):
                tg = rt.task_group()
                for c in range(COLS):
                    for w in range(NWORK):
                        _, gr, gc = tile_of(c, w, r)
                        rt.fill(col_in[c].prod(), src, TensorAccessPattern(
                            (IMG_ELEMS,), offset=(gr * IMG + gc) * IC,
                            sizes=[1, 12, 12 * IC],
                            strides=[0, IMG * IC, 1]), task_group=tg)
                    for w in range(NWORK):
                        _, gr, gc = tile_of(c, w, r)
                        rt.drain(col_out[c].cons(), dst, TensorAccessPattern(
                            (IMG_ELEMS,), offset=((PAD + gr) * IMG + PAD + gc) * IC,
                            sizes=[1, 1, TILE, TILE * IC],
                            strides=[0, 0, IMG * IC, 1]), task_group=tg,
                            wait=(c == COLS - 1 and w == NWORK - 1))
                rt.finish_task_group(tg)

    module = Program(dev, rt).resolve_program()
    # lower placement + FIFOs first: the join BDs the credit pacing patches
    # only exist after objectFifo-stateful-transform
    from aie.passmanager import PassManager
    PassManager.parse(
        "builtin.module(canonicalize,aie-canonicalize-device,"
        "aie.device(aie-place-tiles,aie-assign-lock-ids,aie-register-objectFifos,"
        "aie-objectFifo-stateful-transform{dynamic-objFifos=true}))",
        module.context).run(module.operation)
    _patch_wt_replay(module, COLS, NWORK, TPR, n_iters, MEM_STREAM, WT_BUF_ADDR,
                     static_wt=static_wt, n_slot=SLOTS_PER_PAIR)
    return module


def _patch_wt_replay(module, cols, nwork, tpr, n_iters, mem_stream, wbuf_addr,
                     static_wt=0, n_slot=4):
    N_SLOT_P = n_slot
    """Post-resolve patch: fixed wbuf addresses; per col memtile slot buffer +
    locks + S2MM1 fill / MM2S1 replay DMA + flows + shim alloc + per-iter
    runtime fills of WT (sequence arg 1)."""
    from aie.dialects.aie import (
        DMAChannelDir, EndOp, LockAction, WireBundle, buffer, dma_bd,
        dma_start, flow, lock, memtile_dma, next_bd, packetflow,
        shim_dma_allocation, use_lock,
    )
    from aie.dialects.aiex import (
        shim_dma_single_bd_task, dma_start_task,
    )
    from aie.ir import InsertionPoint

    dev_op = None
    for op in module.body.operations:
        if op.operation.name == "aie.device":
            dev_op = op
    body = dev_op.regions[0].blocks[0]
    tiles = {}
    seq_op = None
    for op in body.operations:
        nm = op.operation.name
        if nm == "aie.tile":
            import re as _re
            mt = _re.search(r"col\s*=\s*(\d+),\s*row\s*=\s*(\d+)", str(op))
            tiles[(int(mt.group(1)), int(mt.group(2)))] = op
        if "runtime_sequence" in nm:
            seq_op = op
        if nm == "aie.buffer":
            try:
                bname = str(op.attributes["sym_name"]).strip('"')
            except KeyError:
                continue
            if bname.startswith("cw_wt_"):
                from aie.ir import IntegerAttr, IntegerType
                op.attributes["address"] = IntegerAttr.get(
                    IntegerType.get_signless(32, module.context), wbuf_addr)

    from aie.ir import Location
    last = list(body.operations)[-1]
    # locks must precede their uses inside iron's memtile dma blocks
    cred_locks = []
    from aie.dialects.aie import lock as _lock
    from aie.ir import InsertionPoint as _IP, Location as _Loc
    echo_locks = []
    first_dma = next(o for o in body.operations if o.operation.name == "aie.memtile_dma")
    with _IP(first_dma), _Loc.unknown(module.context):
        for c in range(cols):
            cred_locks.append(_lock(tiles[(c, 1)], lock_id=28, init=n_slot,
                                    sym_name=f"wt_cr_{c}"))
            echo_locks.append(_lock(tiles[(c, 1)], lock_id=27, init=n_slot,
                                    sym_name=f"wt_echo_{c}"))
    mem_n = mem_stream // 2  # i32 view of u16 stream
    src_ty = np.ndarray[(mem_n,), np.dtype[np.int32]]
    with InsertionPoint(last), Location.unknown(module.context):
        for c in range(cols):
            mem_t = tiles[(c, 1)]
            shim_t = tiles[(c, 0)]
            # explicit high address — the post-resolve allocator gives
            # patched buffers address 0, overlapping the FIFO buffers
            init = np.arange(100, 100 + mem_n, dtype=np.int32) if static_wt else None
            msrc = buffer(mem_t, datatype=src_ty, name=f"wt_src_{c}", address=0x70000,
                          initial_value=init)
            lk_e = lock(mem_t, lock_id=30, init=tpr, sym_name=f"wt_e_{c}")
            lk_f = lock(mem_t, lock_id=31, init=0, sym_name=f"wt_f_{c}")
            flow(shim_t, WireBundle.DMA, 1, mem_t, WireBundle.DMA, 5)
            for w in range(nwork):
                flow(mem_t, WireBundle.DMA, 5, tiles[(c, 2 + w)], WireBundle.DMA, 1)
            # join-paced credits: iron's finals-join BDs (one per drained
            # tile) each release +1; init covers round 0 (N_SLOT slots ahead)
            lk_cr = cred_locks[c]
            shim_dma_allocation(f"wt_in_{c}", shim_t, DMAChannelDir.MM2S, 1)

            def _mk(msrc, lk_e, lk_f):
                @memtile_dma(mem_t)
                def mt(block):
                    if static_wt:
                        # per-slot credit-paced replay: emit one slot only when
                        # all NWORK cores have armed (credits via token packets)
                        slot_n = mem_n // N_SLOT_P
                        dma_start(DMAChannelDir.MM2S, 5, dest=block[1],
                                  chain=block[N_SLOT_P + 1],
                                  repeat_count=n_iters * tpr - 1)
                        for si in range(N_SLOT_P):
                            with block[1 + si]:
                                use_lock(lk_cr, LockAction.AcquireGreaterEqual, value=1)
                                dma_bd(msrc, offset=si * slot_n, len=slot_n)
                                use_lock(echo_locks[c], LockAction.Release, value=1)
                                next_bd(block[2 + si] if si < N_SLOT_P - 1 else block[N_SLOT_P + 1])
                        with block[N_SLOT_P + 1]:
                            EndOp()
                    else:
                        _mt_body(block, msrc, lk_e, lk_f)
            def _mt_body(block, msrc, lk_e, lk_f):
                dma_start(DMAChannelDir.S2MM, 5, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(lk_e, LockAction.AcquireGreaterEqual, value=tpr)
                    dma_bd(msrc, offset=0, len=mem_n)
                    use_lock(lk_f, LockAction.Release, value=tpr)
                    next_bd(block[1])
                with block[2]:
                    dma_start(DMAChannelDir.MM2S, 5, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(lk_f, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(msrc, offset=0, len=mem_n)
                    use_lock(lk_e, LockAction.Release, value=1)
                    next_bd(block[3])
                with block[4]:
                    EndOp()
            _mk(msrc, lk_e, lk_f)

    # join-paced credits: BD blocks allow only one release, so chain a
    # zero-length credit BD after each join BD: join -> credit(+1) -> orig
    from aie.ir import Block
    targets = []  # (block, bd_op, next_bd_op, col)
    for op in body.operations:
        if op.operation.name != "aie.memtile_dma":
            continue
        for blk in list(op.regions[0].blocks):
            ops = list(blk.operations)
            for i, o in enumerate(ops):
                if o.operation.name != "aie.dma_bd":
                    continue
                txt = str(o)
                if "cw_out_" not in txt:
                    continue
                rel = next((o2 for o2 in ops[i:] if o2.operation.name == "aie.use_lock"
                            and "Release" in str(o2)), None)
                if rel is None or "_cons_lock" not in str(rel):
                    continue
                col = int(txt.split("cw_out_")[1].split("_")[0])
                nxt = next(o2 for o2 in ops[i:] if o2.operation.name == "aie.next_bd")
                targets.append((blk, o, nxt, col))
    for blk, bd_op, nxt, col in targets:
        orig_target = nxt.successors[0]
        nb = Block.create_after(blk)
        buf_val = bd_op.operands[0]
        with InsertionPoint(nb), Location.unknown(module.context):
            use_lock(echo_locks[col], LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buf_val, offset=0, len=0)
            use_lock(cred_locks[col], LockAction.Release, value=1)
            next_bd(orig_target)
        nxt.operation.erase()
        with InsertionPoint(blk), Location.unknown(module.context):
            next_bd(nb)

    seq_block = seq_op.regions[0].blocks[0]
    wt_arg = seq_block.arguments[1]
    with InsertionPoint.at_block_begin(seq_block), Location.unknown(module.context):
        for c in range(0 if static_wt else cols):
            for it in range(n_iters):
                t = shim_dma_single_bd_task(
                    f"wt_in_{c}", wt_arg, offset=it * mem_stream,
                    sizes=[1, 1, 1, mem_stream])
                dma_start_task(t)


if __name__ == "__main__":
    import os
    print(rn3_chain_geo(os.environ.get("GEO", "re8"), int(os.environ.get("N_ITERS", "2"))))
