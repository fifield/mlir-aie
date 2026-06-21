#!/usr/bin/env python3
"""KEYSTONE generator: 3x3 multicore conv whose input is gathered ON-DEVICE as
halo'd windows directly from a contiguous PAD-padded HWC image.

NO host im2col. The host buffer is exactly the format every on-device producer
emits: a PAD(2)-padded HWC image (memref<IMG_H*IMG_W*IC>). Per output 8x8 tile
an input fill TAP gathers the overlapping (8+2)x(8+2)xIC source window from that
shared image into the per-core L1 patch buffer (option (b)); the compute kernel
then halo-reads the window with the chain's proven on-the-fly im2col indexing
(`_build_a64_halo`, patch_w=10). Overlapping windows are a *source gather* (tile
stride 8 < window width 10 -> overlapping source rows): the non-overlapping
restriction only bit `split`, which we do NOT use for the gather.

This is the seam construct for chain->c3: the rn3 chain emits a 28x28x64 PAD(2)
padded HWC BO; c3 (3x3 stride-1 pad-1) reads 8x8 output tiles whose halo windows
are all inside that buffer (PAD=2 >= conv pad=1). No 4.08x im2col layout bridge.

Geometry (re8 c3 seam): padded image IMG_H x IMG_W x IC, valid feature map
GBOUND x GBOUND at offset (PAD,PAD). Output GBOUND x GBOUND x OC, produced as
ceil(GBOUND/8)^2 8x8 tiles. The TAP window origin for output tile (tr,tc) is
(tr*8, tc*8) in padded coords -- exactly the pad-1 halo because PAD=2 buffer
origin already sits one pixel before the valid region for a pad-1 conv... see
test_halo_conv_hw.py for the host-im2col-reference parity proof.
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

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

from kernels.halo_conv3x3_bfp import (
    halo_conv3x3_bfp, KERNEL_EXTRA_GLOBALS, HALO_CONV_HELPERS,
)

TILE = 8
PAD = 2
WIN = TILE + 2          # 10 (8x8 tile + halo for a 3x3 stride-1 pad-1 conv)


def halo_conv(ic=64, oc=64, gbound=20, n_iters=1, stack_size=4096):
    """Multicore 3x3 halo conv. Output = gbound x gbound x oc tiled into 8x8.

    Tiles laid out raster: tile idx = tr*GRID + tc, GRID = ceil(gbound/8). Each
    of the up-to-4-per-column cores owns one tile; the input fill TAP gathers
    that tile's WIN x WIN x IC window from the shared padded image.
    """
    GRID = (gbound + TILE - 1) // TILE
    N_TILES = GRID * GRID
    IMG_W = GRID * TILE + 2 * PAD          # padded image width (covers all halos)
    IMG_H = IMG_W
    IMG_ELEMS = IMG_H * IMG_W * ic

    N_BLK_IC = ic // 8
    N_BLK_OC = oc // 8
    WIN_ELEMS = WIN * WIN * ic
    WSLOT = N_BLK_OC * (N_BLK_IC * 9) * 64  # BFP B layout [ocb, kk, 8, 8]
    C_ELEMS = N_BLK_OC * TILE * 8 * 8       # [N_BLOCKS, M_BLOCKS=8, 8, 8]

    cores_per_col = 4
    n_cols = (N_TILES + cores_per_col - 1) // cores_per_col

    dev = NPU2()
    win_ty = np.ndarray[(WIN_ELEMS,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    c_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    host_out_ty = np.ndarray[(N_TILES * C_ELEMS,), np.dtype[np.float32]]

    kern = PythocKernel(halo_conv3x3_bfp, [win_ty, wt_ty, c_ty, np.int32, np.int32],
                        extra_globals=KERNEL_EXTRA_GLOBALS, helpers=list(HALO_CONV_HELPERS))

    def core_fn(a, w, o, k, iters):
        for _ in range_(iters):
            ew = w.acquire(1)
            ein = a.acquire(1)
            eo = o.acquire(1)
            k(ein, ew, eo, ic, oc)
            a.release(1)
            o.release(1)
            w.release(1)

    workers, col_in, col_out, col_wt = [], [], [], []
    for col in range(n_cols):
        ct = min(cores_per_col, N_TILES - col * cores_per_col)
        col_in_ty = np.ndarray[(ct * WIN_ELEMS,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(ct * C_ELEMS,), np.dtype[np.float32]]
        fin = ObjectFifo(col_in_ty, depth=1, name=f"hc_in_{col}")
        fout = ObjectFifo(col_out_ty, depth=1, name=f"hc_out_{col}")
        fwt = ObjectFifo(wt_ty, depth=1, name=f"hc_wt_{col}")
        in_sp = fin.cons().split(offsets=[i * WIN_ELEMS for i in range(ct)],
                                 obj_types=[win_ty] * ct, depths=[1] * ct,
                                 names=[f"hc_in_{col}_{i}" for i in range(ct)])
        out_j = fout.prod().join(offsets=[i * C_ELEMS for i in range(ct)],
                                 obj_types=[c_ty] * ct, depths=[1] * ct,
                                 names=[f"hc_out_{col}_{i}" for i in range(ct)])
        col_in.append(fin); col_out.append(fout); col_wt.append(fwt)
        for i in range(ct):
            workers.append(Worker(core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(), kern, n_iters],
                stack_size=stack_size))

    def tile_rc(t):
        return (t // GRID), (t % GRID)

    rt = Runtime()
    with rt.sequence(img_ty, wt_ty, host_out_ty) as (IMG, WT, OUT):
        rt.start(*workers)
        for _ in range(n_iters):
            for col in range(n_cols):
                ct = min(cores_per_col, N_TILES - col * cores_per_col)
                # weights broadcast to the column (every core convolves the
                # same OC; full-layer per-tile weight reuse)
                rt.fill(col_wt[col].prod(), WT)
                # ---- KEYSTONE: per-tile overlapping-window source gather ----
                # One fill per core-tile in the column: the TAP origin is the
                # tile's window top-left (tr*8, tc*8) in PADDED coords; it reads
                # a WIN x WIN x IC block. Adjacent tiles' origins are 8 apart
                # but each window is 10 wide => overlapping SOURCE rows. No host
                # im2col: the source is the contiguous padded HWC image.
                for i in range(ct):
                    t = col * cores_per_col + i
                    tr, tc = tile_rc(t)
                    off = (tr * TILE * IMG_W + tc * TILE) * ic
                    rt.fill(col_in[col].prod(), IMG, TensorAccessPattern(
                        (IMG_ELEMS,), offset=off,
                        sizes=[1, WIN, WIN, ic],
                        strides=[0, IMG_W * ic, ic, 1]))
                # drain the column's tiled C accumulators
                tap_out = TensorAccessPattern(
                    (N_TILES * C_ELEMS,),
                    offset=col * cores_per_col * C_ELEMS,
                    sizes=[1, ct * C_ELEMS], strides=[0, 1])
                rt.drain(col_out[col].cons(), OUT, tap_out, wait=True)

    meta = dict(GRID=GRID, N_TILES=N_TILES, IMG_W=IMG_W, IMG_H=IMG_H,
                IMG_ELEMS=IMG_ELEMS, WIN=WIN, WIN_ELEMS=WIN_ELEMS,
                WSLOT=WSLOT, C_ELEMS=C_ELEMS, N_BLK_IC=N_BLK_IC,
                N_BLK_OC=N_BLK_OC, TILE=TILE, PAD=PAD, ic=ic, oc=oc, gbound=gbound)
    return Program(dev, rt).resolve_program(), meta


if __name__ == "__main__":
    ic = int(os.environ.get("HC_IC", "64"))
    oc = int(os.environ.get("HC_OC", "64"))
    gb = int(os.environ.get("HC_GBOUND", "20"))
    module, meta = halo_conv(ic=ic, oc=oc, gbound=gb)
    assert module.operation.verify()
    print(module)
    print("META", meta, file=sys.stderr)
