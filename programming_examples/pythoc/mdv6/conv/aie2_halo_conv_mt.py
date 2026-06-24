#!/usr/bin/env python3
"""Tiles-per-core (multi-tile) halo conv — re4 (80x80, GRID=10) scaling probe.

NEW generator. Does NOT modify aie2_halo_conv.py (one-tile-per-core, the proven
re8/re6 fusion path a sibling agent is measuring).

PROBLEM the one-tile-per-core path hits:
  aie2_halo_conv.halo_conv places ONE Worker per 8x8 output tile (N_TILES=GRID^2
  workers). re8 GRID=3 -> 9 workers (fits 32 cores); re6 GRID=5 -> 25 (fits);
  re4 GRID=10 -> 100 workers > 32 cores -> aiecc "no available compute tiles".

THIS generator: each Worker processes `tpc` (tiles-per-core) output tiles. re4's
100 tiles map onto ceil(100/tpc) workers (tpc=4 -> 25 workers <= 32 cores). The
tile->core assignment + per-tile core loop is lifted DIRECTLY from the proven rn3
raster chain (conv/aie2_rn3_chain_geo.py rn3_chain_raster): tile index
  idx = (col*NWORK + w)*tpc + r,  raster row-major over GRID^2,
each worker owns the `tpc` tiles {idx(r) : r in 0..tpc-1}, loops them in core_fn,
and the host fills/drains one window/C per (col, w, r) round-major.

The COMPUTE kernel is the proven single-tile halo_conv3x3_bfp (re-exported via
kernels/halo_conv3x3_bfp_mt.py) called once per tile — so exactly ONE tile's
window (WIN*WIN*ic) and ONE tile's C (C_ELEMS) are L1-resident at a time,
independent of tpc. tpc costs L1 only via objectfifo depth, not resident tiles.

Geometry: padded image GRID*8+2*PAD square, valid GBOUND x GBOUND feature map.
Output assembled host-side from N_TILES tiled-C blocks (same untile_c as the
one-tile path). Junk tiles (idx >= GRID^2 when GRID^2 not a multiple of NWORK*tpc)
are gathered/drained from a junk band below the image (raster chain trick) so the
DMA shapes stay uniform per round.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDV6 = HERE.parent
for _p in (str(HERE), str(MDV6)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.halo_conv3x3_bfp_mt import (
    halo_conv3x3_bfp_mt, halo_conv3x3_bfp_mt_ocb1,
    KERNEL_EXTRA_GLOBALS, HALO_CONV_HELPERS,
)

TILE = 8
PAD = 2
WIN = TILE + 2          # 10


def halo_conv_mt(ic=64, oc=64, gbound=20, tpc=4, n_cols=None, n_iters=1,
                 stack_size=4096, shift=0, io_depth=1, stream_oc=False,
                 junk_band=True, auto_place=False):
    """Tiles-per-core 3x3 halo conv. Output = gbound x gbound x oc tiled 8x8.

    tpc    — tiles per core (each worker loops this many output tiles).
    n_cols — device columns to use (workers = n_cols * NWORK). Default: choose
             the smallest layout (NWORK<=4 per col, n_cols<=8) that covers
             ceil(GRID^2 / tpc) workers. 32-core cap = NWORK(4) * n_cols(8).

    Tile->core map (raster, row-major over GRID^2):
        idx(col, w, r) = (col*NWORK + w)*tpc + r,  r in 0..tpc-1
        real tile      => (idx // GRID, idx % GRID) in 8x8 grid coords
        junk tile      => parked in a junk band below the padded image
    The host fills one WIN*WIN*ic window and drains one C_ELEMS C per (col,w,r),
    round-major (r outer). Each worker's core_fn loops its tpc tiles, one kernel
    call per tile (one window in, one C out) — single-tile L1 residency.

    stream_oc — False: one L1 weight slot holds ALL OC + full-OC C buffer (proven
                OC<=32 for re4 tpc=4; OC=64 overflows the 72KB full-OC weight slot).
                "block" (BLK_UNIT=1): per-SINGLE-oc-block weight streaming + per-block
                C drain (the OC=64 L1 fix lifted from aie2_halo_conv.py). Per tile the
                worker loops N_BLK_OC oc-blocks, each call MACs one oc-block (8 chan)
                and drains its 2KB C; only ONE block's wt (~9KB IC=64) + C (2KB) are
                ever L1-resident, so OC=64 fits (~27KB). Output is drained
                BLOCK-major within each (col,w,r) slot — the host de-rasters the same
                raster slot order, then de-interleaves the N_BLK_OC block frames.
    """
    single_block = (stream_oc == "block")
    GRID = (gbound + TILE - 1) // TILE
    N_TILES = GRID * GRID
    n_workers_needed = (N_TILES + tpc - 1) // tpc

    # pick NWORK (cores/col, <=4) and n_cols (<=8) covering the needed workers
    NWORK = min(4, n_workers_needed)
    if n_cols is None:
        n_cols = (n_workers_needed + NWORK - 1) // NWORK
    NW = NWORK
    COLS = n_cols
    assert COLS * NW >= n_workers_needed, (
        f"layout {COLS}x{NW} too small for {n_workers_needed} workers")
    assert COLS <= 8 and NW <= 4, f"layout {COLS}x{NW} exceeds 32-core device"

    IMG_W = GRID * TILE + 2 * PAD
    n_slots = COLS * NW * tpc
    junk = n_slots - N_TILES
    # junk_band=True (standalone): tiles idx>=N_TILES gather from a junk band of
    # rows BELOW the image, keeping the input buffer's DMA shapes uniform but
    # making IMG_H > IMG_W. junk_band=False (MERGED SEAM): the input IMG is the
    # exact producer seam (IMG_H == IMG_W, no junk band) — junk windows instead
    # read tile-0's window (origin shift,shift, fully inside the valid image);
    # their output is drained and ignored, so reading real data is harmless and
    # the seam size matches the dcg producer exactly.
    if junk_band:
        junk_rows = ((junk + GRID - 1) // GRID) * TILE if junk > 0 else 0
        IMG_H = IMG_W + junk_rows
    else:
        IMG_H = IMG_W
    IMG_ELEMS = IMG_H * IMG_W * ic
    JUNK_ROW0 = GRID * TILE + 2 * PAD     # first junk row (in padded coords)

    N_BLK_IC = ic // 8
    N_BLK_OC = oc // 8
    WIN_ELEMS = WIN * WIN * ic
    WSLOT = N_BLK_OC * (N_BLK_IC * 9) * 64 + 2 * oc     # conv + bn_w(oc)+bn_b(oc)
    C_ELEMS = N_BLK_OC * TILE * 8 * 8                   # [N_BLK_OC,8,8,8]

    # --- OC-block streaming (the OC=64 L1 fix lifted from aie2_halo_conv.py) ---
    # one streamed unit = 1 oc-block conv slot + bn_w(8) + bn_b(8), padded up to a
    # 64-elem multiple (clean mmul-block inner DMA size). N_BLK_OC units streamed.
    _wslot_raw = (N_BLK_IC * 9) * 64 + 2 * 8
    WSLOT_BLK = ((_wslot_raw + 63) // 64) * 64           # one block's wt slot
    BLK_C = TILE * 8 * 8                                  # one block's tiled C [8,8,8]
    if single_block:
        # streamed host WT buffer = N_BLK_OC units of WSLOT_BLK back-to-back
        WSLOT = N_BLK_OC * WSLOT_BLK

    def tile_of(col, w, r):
        """(col,w,r) -> (real, grow, gcol) in PADDED coords (window top-left)."""
        idx = (col * NW + w) * tpc + r
        if idx < N_TILES:
            tr, tc = idx // GRID, idx % GRID
            # window origin in padded coords = (tr*8+shift, tc*8+shift)
            return True, tr * TILE + shift, tc * TILE + shift, idx
        k = idx - N_TILES
        if junk_band:
            # junk: park in the band below the image (well inside IMG_H)
            return False, JUNK_ROW0 + (k // GRID) * TILE, (k % GRID) * TILE, idx
        # no junk band (merged seam): junk reads tile-0's window (inside the
        # valid image); its drained output is ignored by the host de-raster.
        return False, shift, shift, idx

    dev = NPU2()
    win_ty = np.ndarray[(WIN_ELEMS,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    wt_blk_ty = np.ndarray[(WSLOT_BLK,), np.dtype[np.uint16]]
    c_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]
    blk_c_ty = np.ndarray[(BLK_C,), np.dtype[np.float32]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    # host OUT: one C_ELEMS slot per (col,w,r) round-major slot (n_slots total).
    # Junk slots get a slot too (drained, ignored); the host de-rasters reals.
    # In the streaming path each slot still holds N_BLK_OC blocks back-to-back
    # (BLOCK-major within the slot), so C_ELEMS per slot is unchanged.
    host_out_ty = np.ndarray[(n_slots * C_ELEMS,), np.dtype[np.float32]]

    if single_block:
        kern = PythocKernel(
            halo_conv3x3_bfp_mt_ocb1,
            [win_ty, wt_blk_ty, blk_c_ty, np.int32, np.int32, np.int32],
            extra_globals=KERNEL_EXTRA_GLOBALS, helpers=list(HALO_CONV_HELPERS))
    else:
        kern = PythocKernel(halo_conv3x3_bfp_mt, [win_ty, wt_ty, c_ty, np.int32, np.int32],
                            extra_globals=KERNEL_EXTRA_GLOBALS, helpers=list(HALO_CONV_HELPERS))

    def core_fn(a, w, o, k, iters):
        # tiles-per-core loop: tpc kernel calls, one window/C per call. Weights
        # are broadcast (same WSLOT for every tile), acquired once per call.
        for _ in range_(iters):
            r = 0
            while r < tpc:
                ein = a.acquire(1)
                ew = w.acquire(1)
                eo = o.acquire(1)
                k(ein, ew, eo, ic, oc)
                a.release(1)
                w.release(1)
                o.release(1)
                r = r + 1

    def core_fn_stream(a, w, o, k, iters):
        # OC-block streaming: per tile acquire the window ONCE, then loop the
        # N_BLK_OC oc-blocks (acquire one wt block-slot, acquire one block-C out,
        # MAC the block, release out so the host drains it, release wt). Only one
        # block's wt (~9KB IC=64) + C (2KB) are ever L1-resident -> OC=64 fits.
        for _ in range_(iters):
            r = 0
            while r < tpc:
                ein = a.acquire(1)
                nb = 0
                while nb < N_BLK_OC:
                    ew = w.acquire(1)
                    eo = o.acquire(1)
                    k(ein, ew, eo, ic, oc, nb)
                    o.release(1)
                    w.release(1)
                    nb = nb + 1
                a.release(1)
                r = r + 1

    # output FIFO object: full per-tile C (single-slot) or per-BLOCK C (streaming).
    OUT_OBJ = BLK_C if single_block else C_ELEMS
    out_obj_ty = blk_c_ty if single_block else c_ty
    wt_fifo_ty = wt_blk_ty if single_block else wt_ty

    workers, col_in, col_out, col_wt = [], [], [], []
    for col in range(COLS):
        # per-column FIFOs: in carries NW windows/round (one per worker), the
        # split fans them to per-worker depth-2 buffers; out joins NW C/round.
        col_in_ty = np.ndarray[(NW * WIN_ELEMS,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(NW * OUT_OBJ,), np.dtype[np.float32]]
        # weights are broadcast + identical every round -> depth-1 (the proven
        # one-tile path uses depth-1 wt). win/out at io_depth for round pipelining.
        fin = ObjectFifo(col_in_ty, depth=io_depth, name=f"hcm_in_{col}")
        fout = ObjectFifo(col_out_ty, depth=io_depth, name=f"hcm_out_{col}")
        fwt = ObjectFifo(wt_fifo_ty, depth=1, name=f"hcm_wt_{col}")
        in_sp = fin.cons().split(offsets=[i * WIN_ELEMS for i in range(NW)],
                                 obj_types=[win_ty] * NW, depths=[io_depth] * NW,
                                 names=[f"hcm_in_{col}_{i}" for i in range(NW)])
        out_j = fout.prod().join(offsets=[i * OUT_OBJ for i in range(NW)],
                                 obj_types=[out_obj_ty] * NW, depths=[io_depth] * NW,
                                 names=[f"hcm_out_{col}_{i}" for i in range(NW)])
        col_in.append(fin); col_out.append(fout); col_wt.append(fwt)
        for i in range(NW):
            wkw = dict(stack_size=stack_size)
            # auto_place=True (merged ELF): let the placer pick tiles so the halo
            # doesn't collide with the chain/dcg sub-devices in the same columns.
            # Standalone (False): pin Tile(col, 2+i) for a deterministic layout.
            if not auto_place:
                wkw["tile"] = Tile(col, 2 + i)
            workers.append(Worker(core_fn_stream if single_block else core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(), kern, n_iters],
                **wkw))

    rt = Runtime()
    with rt.sequence(img_ty, wt_ty, host_out_ty) as (IMG, WT, OUT):
        rt.start(*workers)
        for _ in range(n_iters):
            for r in range(tpc):
                tg = rt.task_group()
                for col in range(COLS):
                    # one window per worker this round
                    for w in range(NW):
                        _, grow, gcol, _idx = tile_of(col, w, r)
                        off = (grow * IMG_W + gcol) * ic
                        rt.fill(col_in[col].prod(), IMG, TensorAccessPattern(
                            (IMG_ELEMS,), offset=off,
                            sizes=[1, WIN, WIN, ic],
                            strides=[0, IMG_W * ic, ic, 1]), task_group=tg)
                    if single_block:
                        # weights: one DMA, outer dim N_BLK_OC units of WSLOT_BLK
                        # (block b = WT[b*WSLOT_BLK:]). Broadcast to the column;
                        # the depth-1 wt FIFO lock-paces the N_BLK_OC pushes over a
                        # single BD chain (same as the one-tile stream path).
                        rt.fill(col_wt[col].prod(), WT, TensorAccessPattern(
                            (WSLOT,), offset=0,
                            sizes=[N_BLK_OC, 1, WSLOT_BLK],
                            strides=[WSLOT_BLK, 0, 1]), task_group=tg)
                        # output: ONE CONTIGUOUS DMA per column-round (a per-(b,w)
                        # scatter blows past the shim BD pool — same lesson the
                        # one-tile stream path documents). The join emits BLOCK-major
                        # frames this round ([block b][core w][BLK_C]); drain them
                        # LINEARLY into this round's slice of a column-packed OUT
                        # region. The host de-interleaves (block-major -> in-slot
                        # block) — same bytes, free permutation. Per-round column
                        # base: each round contributes NW*N_BLK_OC*BLK_C f32.
                        col_round_base = ((col * tpc + r) * NW * N_BLK_OC) * BLK_C
                        rt.drain(col_out[col].cons(), OUT, TensorAccessPattern(
                            (n_slots * C_ELEMS,), offset=col_round_base,
                            sizes=[1, NW * N_BLK_OC * BLK_C], strides=[0, 1]),
                            task_group=tg, wait=(col == COLS - 1))
                    else:
                        # weights: whole-buffer broadcast to the column (the FIFO's
                        # NW consumers fan it out, same as the one-tile path).
                        rt.fill(col_wt[col].prod(), WT, task_group=tg)
                        # drain NW C's this round, round-major into OUT slot layout:
                        # slot(col,w,r) = ((col*NW + w)*tpc + r) -> contiguous C_ELEMS
                        for w in range(NW):
                            slot = (col * NW + w) * tpc + r
                            rt.drain(col_out[col].cons(), OUT, TensorAccessPattern(
                                (n_slots * C_ELEMS,), offset=slot * C_ELEMS,
                                sizes=[1, C_ELEMS], strides=[0, 1]), task_group=tg,
                                wait=(col == COLS - 1 and w == NW - 1))
                rt.finish_task_group(tg)

    meta = dict(GRID=GRID, N_TILES=N_TILES, IMG_W=IMG_W, IMG_H=IMG_H,
                IMG_ELEMS=IMG_ELEMS, WIN=WIN, WIN_ELEMS=WIN_ELEMS,
                WSLOT=WSLOT, C_ELEMS=C_ELEMS, N_BLK_IC=N_BLK_IC,
                N_BLK_OC=N_BLK_OC, TILE=TILE, PAD=PAD, ic=ic, oc=oc, gbound=gbound,
                tpc=tpc, NWORK=NW, COLS=COLS, n_workers=COLS * NW,
                n_slots=n_slots, JUNK_ROW0=JUNK_ROW0, shift=shift,
                stream_oc=bool(single_block), WSLOT_BLK=WSLOT_BLK, BLK_C=BLK_C)
    return Program(dev, rt).resolve_program(), meta


def slot_to_tile(slot, meta):
    """OUT slot index -> output-grid tile index (or None if junk)."""
    NW, tpc, GRID, N_TILES = meta["NWORK"], meta["tpc"], meta["GRID"], meta["N_TILES"]
    # slot = (col*NW + w)*tpc + r  ==  idx (since OUT slot IS the raster idx)
    idx = slot
    if idx < N_TILES:
        return idx
    return None


def deinterleave_stream_mt(flat, meta):
    """Reorder the stream_oc='block' mt OUT (column-round-packed, BLOCK-major)
    into the canonical [n_slots, C_ELEMS] = [n_slots, N_BLK_OC, BLK_C] slot-major
    layout that slot_to_tile() + untile_c() expect.

    The device drains, per (col, r) round, ONE contiguous frame laid out as
    [block b][core w][BLK_C] (block-major, core-minor — the join's emission order
    when each worker releases its N_BLK_OC blocks in order). Here we scatter each
    (b, w) BLK_C chunk into canonical slot s = (col*NW + w)*tpc + r at +b*BLK_C."""
    NW, tpc, COLS = meta["NWORK"], meta["tpc"], meta["COLS"]
    N_BLK_OC, BLK_C, C_ELEMS = meta["N_BLK_OC"], meta["BLK_C"], meta["C_ELEMS"]
    n_slots = meta["n_slots"]
    canon = np.zeros(n_slots * C_ELEMS, np.float32)
    pos = 0
    for col in range(COLS):
        for r in range(tpc):
            for b in range(N_BLK_OC):
                for w in range(NW):
                    s = (col * NW + w) * tpc + r
                    dst = s * C_ELEMS + b * BLK_C
                    canon[dst:dst + BLK_C] = flat[pos:pos + BLK_C]
                    pos = pos + BLK_C
    return canon


if __name__ == "__main__":
    ic = int(os.environ.get("HC_IC", "64"))
    oc = int(os.environ.get("HC_OC", "64"))
    gb = int(os.environ.get("HC_GBOUND", "20"))
    tpc = int(os.environ.get("HC_TPC", "4"))
    module, meta = halo_conv_mt(ic=ic, oc=oc, gbound=gb, tpc=tpc)
    assert module.operation.verify()
    print(module)
    print("META", meta, file=sys.stderr)
