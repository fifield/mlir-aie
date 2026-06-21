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


def depad_concat_gemm(ic2=64, oc=128, gbound=20, stack_size=8192):
    """1x1 GEMM (2*ic2 -> oc) over a gbound x gbound valid feature map whose input
    is assembled device-resident from a PAD(2)-padded chain image (channels
    [0:ic2]) + a PAD(2)-padded x2 image (channels [ic2:2*ic2]).

    Input BO (stacked):  [chain_padded(IMG*IMG*ic2) | x2_padded(IMG*IMG*ic2)].
    Output BO (seam):    IMG*IMG*oc PAD(2)-padded HWC (== halo_conv(ic=oc).img_ty).
    """
    assert gbound % 4 == 0, f"gbound={gbound} must be %4==0 (mmul<4,8,8>)"
    assert ic2 % 8 == 0 and oc % 8 == 0
    ic = 2 * ic2                                 # fused input channels (128)

    GRID = (gbound + TILE - 1) // TILE
    IMG = GRID * TILE + 2 * PAD                  # padded image width (re8: 28)
    HALF_ELEMS = IMG * IMG * ic2                 # one padded source half
    IN_ELEMS = 2 * HALF_ELEMS                    # stacked [chain | x2]
    IMG_ELEMS = IMG * IMG * oc                   # seam BO size (== halo img_ty)

    tile_m = gbound                              # one valid row per core
    n_cores = gbound
    input_tile_size = tile_m * ic                # per-core fused input [20,128]
    output_tile_size = tile_m * oc
    weight_size = oc * ic + 2 * oc               # conv + BN scale/bias

    cores_per_col = 4
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

        def _factor(n, mx=1023):
            for inner in range(mx, 0, -1):
                if n % inner == 0:
                    return n // inner, inner
            return n, 1
        out_d1, out_d0 = _factor(output_tile_size)

        for col in range(n_cols):
            cc = min(cores_per_col, n_cores - col * cores_per_col)
            r0 = col * cores_per_col
            # ---- DE-PAD + CONCAT input gather ----
            # Column buffer is [cc cores, gbound px, 2 halves, IC2 ch] linear =
            # [cc, gbound, ic=128] HWC. Core (r0+i) owns valid image row r0+i.
            # Source stacked BO: [chain_padded | x2_padded], each IMG*IMG*ic2.
            # de-pad: valid row r lives at padded row PAD+r, col PAD:
            #   chain base elem = ((PAD + r)*IMG + PAD)*ic2
            #   x2    base elem = HALF_ELEMS + ((PAD + r)*IMG + PAD)*ic2
            # concat: half h (chain=0, x2=1) at source offset h*HALF_ELEMS.
            # strides: core -> IMG*ic2 (next padded row, since core==row);
            #          pixel -> ic2 (next column within padded row);
            #          half  -> HALF_ELEMS (chain vs x2 region);
            #          ch    -> 1.
            tap_in = TensorAccessPattern(
                (IN_ELEMS,),
                offset=((PAD + r0) * IMG + PAD) * ic2,
                sizes=[cc, gbound, 2, ic2],
                strides=[IMG * ic2, ic2, HALF_ELEMS, 1])
            rt.fill(col_in_fifos[col].prod(), I, tap_in)
            # ---- DRAIN: scatter cc core outputs into the PADDED seam image ----
            tap_out = TensorAccessPattern(
                (IMG_ELEMS,),
                offset=((PAD + r0) * IMG + PAD) * oc,
                sizes=[1, cc, out_d1, out_d0],
                strides=[0, IMG * oc, out_d0, 1])
            rt.drain(col_out_fifos[col].cons(), OUT, tap_out,
                     wait=(col == n_cols - 1))

    meta = dict(GRID=GRID, IMG=IMG, IMG_ELEMS=IMG_ELEMS, HALF_ELEMS=HALF_ELEMS,
                IN_ELEMS=IN_ELEMS, PAD=PAD, TILE=TILE, ic2=ic2, ic=ic, oc=oc,
                gbound=gbound, tile_m=tile_m, n_cores=n_cores,
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
