#!/usr/bin/env python3
"""C1 generator: de-pad(chain) + channel-concat(x2) -> rnm 1x1 GEMM, device-resident.

This is the chain->rnm device-resident seam (re8 RepNCSP conv3). The rn3 chain
produces the bottleneck output as a PAD(2)-padded HWC image (28x28x64, the valid
20x20x64 interior parked at [PAD:PAD+20]). The model then does
`concat(depad(chain), x2)` = 20x20x128 HWC and feeds it to the rnm GEMM
(1x1 128->128 + BN + SiLU). Today that de-pad + concat + per-core repack is on the
host. This generator does it device-resident as the rnm GEMM's INPUT GATHER:

  - The rnm GEMM (aie2_gemm_pad_out) reads ONE valid image row per core: core r
    consumes its row's [gbound, ic=128] HWC patch and emits [gbound, oc].
  - We replace its single per-core gather with a DE-PAD + CONCAT gather: per pixel
    of core r's row, channels [0:IC2] come from the chain padded image (interior,
    strided to strip the PAD border) and channels [IC2:2*IC2] come from x2.

KEY LAYOUT INSIGHT (verified): a single rt.fill gather TAP can interleave two
source regions per pixel ONLY if both regions share the same per-pixel stride
(the concat_proof quarter trick). The chain's valid interior has padded-row
stride IMG*IC2 (28*64); a contiguous x2 would have stride GBOUND*IC2 (20*64) — a
mismatch that breaks the single uniform TAP. So we require x2 to arrive in the
SAME PAD(2)-padded layout as the chain (28x28x64, interior at [PAD:PAD+20]). Then
both halves share stride IMG*IC2, and ONE gather TAP

    sizes   = [cc_cores, gbound_px, 2_halves, IC2_ch]
    strides = [<per-core>, IC2,    HALF_ELEMS, 1]      (+ PAD offset)

interleaves per pixel into the [gbound, 128] HWC buffer the GEMM consumes — the
de-pad (PAD offset + IMG*IC2 row stride) and concat (HALF_ELEMS half stride) in
ONE BD. For C1 (standalone) x2 is host-padded to 28x28 (cheap; in C2 x2 is filled
into the chain's BO region directly). The two padded halves are STACKED in one
input BO: [chain_padded(IMG*IMG*IC2) | x2_padded(IMG*IMG*IC2)].

The compute kernel is the SAME proven gemm_conv1x1_fused_packed_bf16 (matmul +
BN + SiLU); only the input gather TAP changes vs aie2_gemm_pad_out. The output
DRAIN is identical to aie2_gemm_pad_out (PAD(2)-padded HWC seam, = halo_c3 input).

Run:  source env.sh && python3 conv/aie2_depad_concat_gemm.py [ic2] [oc] [gbound]
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

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

TILE = 8
PAD = 2

KERNELS_DIR = os.path.join(MDV6, "kernels", "build")


def depad_concat_gemm(ic2=64, oc=128, gbound=20, stack_size=8192, chain_img_h=None):
    """1x1 GEMM (2*ic2 -> oc) over a gbound x gbound valid feature map whose input
    is assembled device-resident from a PAD(2)-padded chain image (channels
    [0:ic2]) + a PAD(2)-padded x2 image (channels [ic2:2*ic2]).

    Input BO (stacked):  [chain_padded(HALF_ELEMS) | x2_padded(HALF_ELEMS)].
    Output BO (seam):    IMG*IMG*oc PAD(2)-padded HWC (== halo_conv(ic=oc).img_ty).

    chain_img_h: when the producing rn3 chain emits a NON-SQUARE padded image
    (chain image HEIGHT > square IMG width — re6: 52-row tall buffer from the
    extra junk worker-tile row, vs the 44-wide square), pass that taller height
    so the chain->x2 stacking boundary (HALF_ELEMS) matches the chain's real
    IMG_ELEMS and x2 stacks ABOVE the tall chain image. The de-pad gather still
    reads only the valid [PAD:PAD+gbound] rows at the IMG-wide row stride, so the
    extra junk rows are never read. Default None = square (re8: IMG==height).
    """
    assert gbound % 4 == 0, f"gbound={gbound} must be %4==0 (mmul<4,8,8>)"
    assert ic2 % 8 == 0 and oc % 8 == 0
    ic = 2 * ic2                                 # fused input channels (128)

    GRID = (gbound + TILE - 1) // TILE
    IMG = GRID * TILE + 2 * PAD                  # padded image width (re8: 28)
    img_h = IMG if chain_img_h is None else chain_img_h  # chain image height
    HALF_ELEMS = img_h * IMG * ic2               # one padded source half (tall for re6)
    IN_ELEMS = 2 * HALF_ELEMS                    # stacked [chain | x2]
    IMG_ELEMS = IMG * IMG * oc                   # seam BO size (== halo img_ty)

    # rows_per_core: a core processes `rpc` consecutive valid rows (rpc*gbound
    # pixels in the GEMM M dim). gbound rows over a 32-compute-tile NPU2 needs
    # rpc>=ceil(gbound/32) (re8 gbound=20 -> rpc=1, 20 cores; re6 gbound=40 ->
    # rpc=2, 20 cores). The de-pad+concat gather becomes a 4D per-core TAP
    # [rpc rows, gbound px, 2 halves, ic2] (one fill per core: the rpc valid
    # rows are IMG*ic2 apart in the padded source, not contiguous, so they can't
    # fold into the column's outer core dim without a 5th shim BD dim).
    cores_per_col = 4
    max_cores = 32
    rpc = 1
    while (gbound + rpc - 1) // rpc > max_cores:
        rpc += 1
    assert gbound % rpc == 0, f"gbound={gbound} must be divisible by rows_per_core={rpc}"
    n_cores = gbound // rpc
    tile_m = rpc * gbound                         # pixels per core (M dim)
    input_tile_size = tile_m * ic                 # per-core fused input
    output_tile_size = tile_m * oc
    weight_size = oc * ic + 2 * oc               # conv + BN scale/bias

    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    dev = NPU2()
    input_ty = np.ndarray[(input_tile_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]
    host_in_ty = np.ndarray[(IN_ELEMS,), np.dtype[np.uint16]]
    host_wt_ty = weight_ty
    seam_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]

    kern_name = "gemm_conv1x1_fused_packed_bf16"
    obj_path = os.path.join(KERNELS_DIR, f"{kern_name}.o")
    kernel = PythocKernel(kern_name, obj_path, [
        input_ty, weight_ty, output_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    RTP_LEN = 6
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    init_rtp = np.array([tile_m, 1, ic, oc, 1, 0], dtype=np.int32)
    rtps = [Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
            for i in range(n_cores)]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
        barrier.wait_for_value(1)
        t_h = my_rtp[0]; t_w = my_rtp[1]; ic_v = my_rtp[2]
        oc_v = my_rtp[3]; s_v = my_rtp[4]; p_v = my_rtp[5]
        elem_wt = of_wt.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kern(elem_in, elem_wt, elem_out, t_h, t_w, ic_v, oc_v, s_v, p_v)
        of_in.release(1)
        of_out.release(1)
        of_wt.release(1)
        barrier.release_with_value(1)

    col_in_fifos, col_out_fifos, wt_fifos, workers = [], [], [], []
    for col in range(n_cols):
        cc = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_ty = np.ndarray[(cc * input_tile_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(cc * output_tile_size,), np.dtype[np.uint16]]
        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"dcg_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[input_tile_size * i for i in range(cc)],
            obj_types=[input_ty] * cc,
            names=[f"dcg_in_{col}_{i}" for i in range(cc)])
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"dcg_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_tile_size * i for i in range(cc)],
            obj_types=[output_ty] * cc,
            names=[f"dcg_out_{col}_{i}" for i in range(cc)])
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"dcg_wt_{col}")
        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)
        for i in range(cc):
            gci = col * cores_per_col + i
            workers.append(Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(),
                kernel, rtps[gci], barriers[gci]], stack_size=stack_size))

    rt = Runtime()
    with rt.sequence(host_in_ty, host_wt_ty, seam_ty) as (I, W, OUT):
        rt.start(*workers)
        _rtp_vals = [int(v) for v in init_rtp]
        def set_rtps(*rtp_bufs):
            for rb in rtp_bufs:
                for k in range(RTP_LEN):
                    rb[k] = _rtp_vals[k]
        rt.inline_ops(set_rtps, rtps)
        for b in barriers:
            rt.set_barrier(b, 1)

        for wf in wt_fifos:
            rt.fill(wf.prod(), W)

        last_col = n_cols - 1
        for col in range(n_cols):
            cc = min(cores_per_col, n_cores - col * cores_per_col)
            # Cores in a column own CONSECUTIVE valid rows: core i owns rows
            # [(col*cpc+i)*rpc : +rpc], so the column covers cc*rpc consecutive
            # valid rows. The per-core buffer is rpc rows stacked, and the rows
            # are uniformly IMG*ic2 apart in the padded source => the core dim
            # and the rpc-row dim MERGE into ONE [cc*rpc rows] dim. This keeps
            # the gather/drain a single column-wide 4D shim DMA (rpc>1 per-core
            # fills exhaust the shim NOC). Split/join then divide cc*rpc*gbound
            # elems into cc cores of rpc*gbound px each.
            r0 = (col * cores_per_col) * rpc          # first valid row of column
            nrow = cc * rpc                           # rows in this column
            # ---- DE-PAD + CONCAT input gather (one column-wide fill) ----
            # de-pad: valid row r at padded row PAD+r, col PAD; concat half h
            # (chain=0, x2=1) at source offset h*HALF_ELEMS. Buffer linear =
            # [cc cores][rpc rows][gbound px][2 halves][ic2] = [nrow][gbound][2][ic2].
            tap_in = TensorAccessPattern(
                (IN_ELEMS,),
                offset=((PAD + r0) * IMG + PAD) * ic2,
                sizes=[nrow, gbound, 2, ic2],
                strides=[IMG * ic2, ic2, HALF_ELEMS, 1])
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            # ---- DRAIN: scatter the column's nrow output rows into the seam ----
            tap_out = TensorAccessPattern(
                (IMG_ELEMS,),
                offset=((PAD + r0) * IMG + PAD) * oc,
                sizes=[1, nrow, gbound, oc],
                strides=[0, IMG * oc, oc, 1])
            rt.drain(col_out_fifos[col].cons(), OUT, tap_out,
                     wait=(col == last_col))

    meta = dict(GRID=GRID, IMG=IMG, IMG_H=img_h, IMG_ELEMS=IMG_ELEMS, HALF_ELEMS=HALF_ELEMS,
                IN_ELEMS=IN_ELEMS, PAD=PAD, TILE=TILE, ic2=ic2, ic=ic, oc=oc,
                gbound=gbound, tile_m=tile_m, n_cores=n_cores, rows_per_core=rpc,
                input_tile_size=input_tile_size, output_tile_size=output_tile_size,
                weight_size=weight_size, n_cols=n_cols, cores_per_col=cores_per_col)
    return Program(dev, rt).resolve_program(), meta


if __name__ == "__main__":
    ic2 = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    oc = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    gb = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    module, meta = depad_concat_gemm(ic2=ic2, oc=oc, gbound=gb)
    assert module.operation.verify()
    print(module)
    print("META", meta, file=sys.stderr)
