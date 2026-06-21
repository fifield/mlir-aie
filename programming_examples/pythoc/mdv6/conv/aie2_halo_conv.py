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
    halo_conv3x3_bfp, halo_conv3x3_bfp_ocb, halo_conv3x3_bfp_ocb1,
    KERNEL_EXTRA_GLOBALS, HALO_CONV_HELPERS,
)

TILE = 8
PAD = 2
WIN = TILE + 2          # 10 (8x8 tile + halo for a 3x3 stride-1 pad-1 conv)


def halo_conv(ic=64, oc=64, gbound=20, n_iters=1, stack_size=4096, shift=0,
              stream_oc=False):
    """Multicore 3x3 halo conv. Output = gbound x gbound x oc tiled into 8x8.

    Tiles laid out raster: tile idx = tr*GRID + tc, GRID = ceil(gbound/8). Each
    of the up-to-4-per-column cores owns one tile; the input fill TAP gathers
    that tile's WIN x WIN x IC window from the shared padded image.

    stream_oc:
      False  — single L1 weight slot holding ALL OC + full-OC C buffer (proven
               OC<=64; OC=128 overflows L1 on BOTH weights and C).
      True / "pair" — per-oc-block-PAIR weight streaming + per-PAIR C drain
               (BLK_UNIT=2). Weight slot = one pair (IC=128: 36KB), C = 4KB.
               Proven OC<=64; at OC=128 the 36KB pair weight + 25KB window +
               4KB C + 4KB stack = ~70KB still overflows the 64KB L1.
      "block" — per-SINGLE-oc-block weight streaming + per-block C drain
               (BLK_UNIT=1). Weight slot = one block (IC=128: 18KB), C = 2KB.
               IC=128: 18+25+2+4 = ~50KB -> FITS. This is the OC=128 C-drain
               that makes the real mc_re8_c3 shape (IC=128->OC=128) run.

    shift: SEAM origin offset (plumbing #2). When a producer (rn3 chain) emits a
    PAD(2)-padded HWC buffer whose VALID feature map sits at [PAD:PAD+G], a pad-1
    3x3 conv's output pixel (0,0) needs the window at padded coord (PAD-1)=shift.
    Baking shift into the TAP origin (tr*8+shift, tc*8+shift) makes halo_c3 read
    the producer buffer at the correct phase with ZERO host shift -- the merged
    chain->halo_c3 seam needs no host-side reformat. Default 0 = standalone use
    (host pre-shifts the image, see test_halo_conv_hw.py).
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

    # plumbing #1 + OC=128 C-drain — stream the weights in oc-block UNITS of
    # BLK_UNIT (1 or 2 oc-blocks) and drain that unit's C, so neither the full
    # weight set nor the full C set is ever resident. BLK_UNIT=2 (pair) reuses
    # the proven 2x2-register kernel; BLK_UNIT=1 (block) halves the L1 weight
    # slot (the lever that makes IC=128->OC=128 fit). Both stream N_UNITS =
    # N_BLK_OC // BLK_UNIT slots and write the SAME [N_BLK_OC,8,8,8] C layout.
    single_block = (stream_oc == "block")
    BLK_UNIT = 1 if single_block else 2
    if stream_oc:
        assert N_BLK_OC % BLK_UNIT == 0, "stream_oc needs oc//8 divisible by BLK_UNIT"
    # legacy names: PAIR == one streamed UNIT (BLK_UNIT oc-blocks)
    N_PAIRS = N_BLK_OC // BLK_UNIT          # number of streamed wt/C units
    WSLOT_PAIR = BLK_UNIT * (N_BLK_IC * 9) * 64
    PAIR_C = BLK_UNIT * TILE * 64           # one unit's tiled C (BLK_UNIT blocks)

    cores_per_col = 4
    n_cols = (N_TILES + cores_per_col - 1) // cores_per_col

    dev = NPU2()
    win_ty = np.ndarray[(WIN_ELEMS,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    wt_pair_ty = np.ndarray[(WSLOT_PAIR,), np.dtype[np.uint16]]
    c_ty = np.ndarray[(C_ELEMS,), np.dtype[np.float32]]
    pair_c_ty = np.ndarray[(PAIR_C,), np.dtype[np.float32]]
    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]
    host_out_ty = np.ndarray[(N_TILES * C_ELEMS,), np.dtype[np.float32]]

    if stream_oc:
        # plumbing #1 + OC=128 C-DRAIN: stream the weights one oc-block-pair at
        # a time (L1 weight buffer = one pair, IC=128: 36KB, vs 288KB full slot)
        # AND give the kernel a per-PAIR C buffer (PAIR_C = 4KB, vs 32KB full-OC
        # C). The kernel writes its pair at cbase=0; the host drains each pair
        # into the right OUT oc-block slot. Neither full weights nor full C is
        # ever resident -> OC=128 fits L1.
        ocb_kern = halo_conv3x3_bfp_ocb1 if single_block else halo_conv3x3_bfp_ocb
        kern = PythocKernel(
            ocb_kern,
            [win_ty, wt_pair_ty, pair_c_ty, np.int32, np.int32, np.int32],
            extra_globals=KERNEL_EXTRA_GLOBALS, helpers=list(HALO_CONV_HELPERS))
    else:
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

    def core_fn_stream(a, w, o, k, iters):
        # Weight + C-DRAIN streaming path (plumbing #1 + OC=128 C-drain):
        # one wt slot AND one C buffer per oc-block-pair. The kernel writes its
        # pair into a PER-PAIR PAIR_C buffer (cbase=0); after each pair the core
        # releases that PAIR_C output buffer and the host drains it into the
        # right OUT oc-block slot. Only ONE pair's weights (36KB IC=128) AND
        # ONE pair's C (4KB) are ever resident -- neither the full weight set
        # (288KB) nor the full C set (32KB) is. This is what lets OC=128 fit.
        for _ in range_(iters):
            ein = a.acquire(1)
            pp = 0
            while pp < N_PAIRS:
                ew = w.acquire(1)
                eo = o.acquire(1)          # one PAIR_C output buffer per pair
                k(ein, ew, eo, ic, oc, pp)
                o.release(1)               # drain this pair before the next
                w.release(1)
                pp = pp + 1
            a.release(1)

    # output FIFO object size: full per-tile C (single-slot path) or per-PAIR C
    # (stream_oc C-drain path). In the C-drain path the FIFO carries one
    # ct*PAIR_C frame per oc-block-pair, drained N_PAIRS times per iter.
    OUT_OBJ = PAIR_C if stream_oc else C_ELEMS

    workers, col_in, col_out, col_wt = [], [], [], []
    for col in range(n_cols):
        ct = min(cores_per_col, N_TILES - col * cores_per_col)
        col_in_ty = np.ndarray[(ct * WIN_ELEMS,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(ct * OUT_OBJ,), np.dtype[np.float32]]
        out_obj_ty = pair_c_ty if stream_oc else c_ty
        fin = ObjectFifo(col_in_ty, depth=1, name=f"hc_in_{col}")
        fout = ObjectFifo(col_out_ty, depth=1, name=f"hc_out_{col}")
        fwt = ObjectFifo(wt_pair_ty if stream_oc else wt_ty, depth=1,
                         name=f"hc_wt_{col}")
        in_sp = fin.cons().split(offsets=[i * WIN_ELEMS for i in range(ct)],
                                 obj_types=[win_ty] * ct, depths=[1] * ct,
                                 names=[f"hc_in_{col}_{i}" for i in range(ct)])
        out_j = fout.prod().join(offsets=[i * OUT_OBJ for i in range(ct)],
                                 obj_types=[out_obj_ty] * ct, depths=[1] * ct,
                                 names=[f"hc_out_{col}_{i}" for i in range(ct)])
        col_in.append(fin); col_out.append(fout); col_wt.append(fwt)
        for i in range(ct):
            workers.append(Worker(core_fn_stream if stream_oc else core_fn,
                [in_sp[i].cons(), fwt.cons(), out_j[i].prod(), kern, n_iters],
                stack_size=stack_size))

    def tile_rc(t):
        return (t // GRID), (t % GRID)

    def fill_windows(col, ct):
        # ---- KEYSTONE: per-tile overlapping-window source gather ----
        # One fill per core-tile in the column: the TAP origin is the tile's
        # window top-left (tr*8, tc*8) in PADDED coords; it reads a WIN x WIN x
        # IC block. Adjacent tiles' origins are 8 apart but each window is 10
        # wide => overlapping SOURCE rows. No host im2col: the source is the
        # contiguous padded HWC image.
        for i in range(ct):
            t = col * cores_per_col + i
            tr, tc = tile_rc(t)
            # SEAM origin offset (plumbing #2): shift window origin by `shift`
            # px in row and col so the conv reads the producer's PAD(2) buffer
            # at the pad-1 phase with no host shift. shift=0 = standalone.
            off = ((tr * TILE + shift) * IMG_W + (tc * TILE + shift)) * ic
            rt.fill(col_in[col].prod(), IMG, TensorAccessPattern(
                (IMG_ELEMS,), offset=off,
                sizes=[1, WIN, WIN, ic],
                strides=[0, IMG_W * ic, ic, 1]))

    rt = Runtime()
    with rt.sequence(img_ty, wt_ty, host_out_ty) as (IMG, WT, OUT):
        rt.start(*workers)
        for _ in range(n_iters):
            for col in range(n_cols):
                ct = min(cores_per_col, N_TILES - col * cores_per_col)
                if stream_oc:
                    # OC=128 C-DRAIN: the core acquires the window ONCE then
                    # loops oc-block UNITS (acquire wt, acquire out, compute,
                    # release out, release wt). One DMA each (NOT one per unit:
                    # N_PAIRS separate fills/drains blow past the shim BD pool at
                    # N_BLK_OC=16). Each transfer carries an OUTER dim = N_PAIRS,
                    # which the depth-1 wt/out FIFOs lock-pace into N_PAIRS pushes
                    # over a SINGLE BD chain. Only one unit's wt+C live in L1.
                    fill_windows(col, ct)
                    # weights: one DMA, outer dim N_PAIRS units of WSLOT_PAIR
                    # (unit p = WT[p*WSLOT_PAIR:]). Broadcast to the column.
                    rt.fill(col_wt[col].prod(), WT, TensorAccessPattern(
                        (WSLOT,), offset=0,
                        sizes=[N_PAIRS, 1, WSLOT_PAIR],
                        strides=[WSLOT_PAIR, 0, 1]))
                    # output: one CONTIGUOUS DMA. The join emits unit-major frames
                    # ([unit p][core i][PAIR_C]); drain them LINEARLY into this
                    # column's OUT region [col*4*N_PAIRS*PAIR_C ..]. A strided
                    # scatter to the [tile, oc_block] OUT layout would need 4 shim
                    # dims (unit x tile x block x frame) -> over the 3D limit and
                    # the BD pool; instead drain linearly here and let the host
                    # de-interleave (unit-major -> [tile, oc_block]) — same total
                    # bytes, the permutation is free on the host.
                    col_base = col * cores_per_col * C_ELEMS
                    tap_out = TensorAccessPattern(
                        (N_TILES * C_ELEMS,),
                        offset=col_base,
                        sizes=[1, N_PAIRS * ct * PAIR_C], strides=[0, 1])
                    rt.drain(col_out[col].cons(), OUT, tap_out, wait=True)
                else:
                    # single-slot path: full per-tile C, one drain per column.
                    rt.fill(col_wt[col].prod(), WT)
                    fill_windows(col, ct)
                    tap_out = TensorAccessPattern(
                        (N_TILES * C_ELEMS,),
                        offset=col * cores_per_col * C_ELEMS,
                        sizes=[1, ct * C_ELEMS], strides=[0, 1])
                    rt.drain(col_out[col].cons(), OUT, tap_out, wait=True)

    meta = dict(GRID=GRID, N_TILES=N_TILES, IMG_W=IMG_W, IMG_H=IMG_H,
                IMG_ELEMS=IMG_ELEMS, WIN=WIN, WIN_ELEMS=WIN_ELEMS,
                WSLOT=WSLOT, C_ELEMS=C_ELEMS, N_BLK_IC=N_BLK_IC,
                N_BLK_OC=N_BLK_OC, TILE=TILE, PAD=PAD, ic=ic, oc=oc, gbound=gbound,
                stream_oc=bool(stream_oc), BLK_UNIT=BLK_UNIT, PAIR_C=PAIR_C,
                N_UNITS=N_PAIRS, cores_per_col=cores_per_col)
    return Program(dev, rt).resolve_program(), meta


def deinterleave_stream_out(flat, meta):
    """Reorder the stream_oc C-drain OUT (column-packed, unit-major) into the
    canonical [N_TILES, C_ELEMS] = [N_TILES, N_BLK_OC, 8, 8, 8] layout that
    untile_c() expects. The device drains each column LINEARLY as
    [unit p][local-tile i][PAIR_C]; here we scatter PAIR_C (= BLK_UNIT oc-blocks)
    of tile t = col*cpc + i into OUT_canon[t, p*BLK_UNIT*512 ..]."""
    N_TILES = meta["N_TILES"]; C_ELEMS = meta["C_ELEMS"]
    N_UNITS = meta["N_UNITS"]; PAIR_C = meta["PAIR_C"]
    cpc = meta["cores_per_col"]; BLK_UNIT = meta["BLK_UNIT"]
    n_cols = (N_TILES + cpc - 1) // cpc
    canon = np.zeros(N_TILES * C_ELEMS, np.float32)
    pos = 0
    for col in range(n_cols):
        ct = min(cpc, N_TILES - col * cpc)
        for p in range(N_UNITS):
            for i in range(ct):
                t = col * cpc + i
                dst = t * C_ELEMS + p * PAIR_C   # PAIR_C = BLK_UNIT*512
                canon[dst:dst + PAIR_C] = flat[pos:pos + PAIR_C]
                pos = pos + PAIR_C
    return canon


if __name__ == "__main__":
    ic = int(os.environ.get("HC_IC", "64"))
    oc = int(os.environ.get("HC_OC", "64"))
    gb = int(os.environ.get("HC_GBOUND", "20"))
    module, meta = halo_conv(ic=ic, oc=oc, gbound=gb)
    assert module.operation.verify()
    print(module)
    print("META", meta, file=sys.stderr)
