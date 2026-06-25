#!/usr/bin/env python3
"""S1 generator: rnm 1x1 GEMM (IC->OC) whose output drain writes a PAD(2)-padded
HWC image — the format halo_c3 (the keystone OC=128 halo-gather conv) reads.

This is the rnm half of the real-model rnm->c3 device-resident seam. The model's
RepNCSP conv3 (rnm) is a 1x1 GEMM 128->128 over the concat(bottleneck, x2) HWC
feature map; the following run_re_mc applies c3 (3x3 128->128) to it. Today the
two are separated by a host tile->HWC reassembly + host im2col. The keystone
halo_c3 reads halo windows from a PAD(2)-padded HWC image (valid GxG feature map
parked at [PAD:PAD+G], border zero). A plain 1x1 GEMM emits tile-blocked unpadded
HWC. So the rnm GEMM's DRAIN must place its GxGxOC result into the INTERIOR of a
padded HWC buffer, exactly the way the rn3 chain drains into its padded image
(aie2_rn3_chain_geo.py drain TAP).

Geometry (re8 c3 seam, matches aie2_halo_conv.halo_conv(ic, oc, gbound)):
  GRID = ceil(gbound/8),  IMG = GRID*8 + 2*PAD  (re8: gbound=20 -> GRID=3, IMG=28)
  padded image = IMG x IMG x OC, valid GxGxOC at offset (PAD,PAD).
The seam BO type emitted here (memref<IMG*IMG*OC xui16>) is BYTE-IDENTICAL to
halo_conv(ic=OC).img_ty, so build_merged's chain_link type-check passes with no
on-device reformat — the rnm GEMM output IS the halo_c3 input.

Spatial->core mapping: M = gbound*gbound valid pixels, row-major HWC. We assign
ONE valid image row (gbound pixels) to each of `gbound` cores (tile_m=gbound,
%4==0 for gbound%4==0). Core r computes valid row r ([gbound, oc]); its drain
TAP scatters that row into the padded interior at row PAD+r, col PAD:
    offset = ((PAD + r) * IMG + PAD) * OC,  contiguous gbound*OC.
The host pre-zeros the seam BO so the PAD(2) border is zero (the halo windows of
the boundary output tiles read that border).

The compute kernel is the SAME proven gemm_conv1x1_fused_packed_bf16 (matmul +
BN + SiLU) used model-wide; only the runtime drain TAP changes. BN+SiLU is part
of the model's rnm (conv3 has its own BN+SiLU), so applying it here is correct.

Run:  source env.sh && python3 conv/aie2_gemm_pad_out.py [ic] [oc] [gbound]
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
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern

TILE = 8
PAD = 2

KERNELS_DIR = os.path.join(MDV6, "kernels", "build")


def gemm_pad_out(ic=128, oc=128, gbound=20, stack_size=8192,
                 img_h=None, seam_elems=None, dst_base=0, m_split=None):
    """1x1 GEMM IC->OC over a gbound x gbound valid feature map; the output DRAIN
    writes a PAD(2)-padded IMG x IMG x OC HWC buffer (interior placed, border
    zero), matching halo_conv(ic=oc).img_ty exactly.

    One valid image row per core: tile_m = gbound, n_cores = gbound. Each core
    consumes its row's [gbound, ic] HWC patch (host packs them per-core) and
    emits [gbound, oc]; the drain scatters it into padded row PAD+r.

    WB1 extension (on-device rn1 -> chain stacked BO): the default is a SQUARE
    IMG x IMG x OC seam (re8). For the rn3-chain stacked-input format the output
    half is TALL (img_h > IMG, re6/re4) and may be one half of a wider stacked BO:
      img_h     : output image HEIGHT (rows). None -> IMG (square, re8).
      seam_elems: TOTAL output BO size in elems. None -> img_h*IMG*oc (one half).
                  Pass 2*HALF to make the output the full stacked [lower|upper] BO.
      dst_base  : element offset where this half's padded image starts in the
                  output BO (0 = lower half, HALF_ELEMS = upper half). The drain
                  TAP base offset adds dst_base; everything else (the GEMM math,
                  the per-core valid-row scatter) is identical. The border of the
                  written half stays whatever the BO held (host zeros it once).
    """
    assert gbound % 4 == 0, f"gbound={gbound} must be %4==0 (mmul<4,8,8>)"
    assert ic % 8 == 0 and oc % 8 == 0

    GRID = (gbound + TILE - 1) // TILE
    IMG = GRID * TILE + 2 * PAD                 # padded image width (re8: 28)
    if img_h is None:
        img_h = IMG                             # square seam (re8: img_h == IMG)
    half_elems = img_h * IMG * oc               # one padded half (tall for re6/re4)
    if seam_elems is None:
        seam_elems = half_elems                 # standalone square/tall half
    IMG_ELEMS = seam_elems                       # actual seam BO size

    # rows-per-core: one valid image row per core needs gbound cores, which
    # exceeds the 32-core array for gbound>32 (re6 gbound=40, re4 gbound=80).
    # Pack `rpc` CONSECUTIVE valid rows per core so n_cores = gbound//rpc <= 32
    # (rpc must divide gbound so the column-wide drain TAP folds cleanly). re8
    # gbound=20 -> rpc=1 (20 cores, unchanged). re6 40 -> rpc=2 (20). re4 80 ->
    # rpc=4 (20). The kernel processes tile_m = rpc*gbound pixels per call.
    cores_per_col = 4
    max_cores = 32
    rpc = 1
    while gbound % rpc != 0 or (gbound + rpc - 1) // rpc > max_cores:
        rpc += 1
        assert rpc <= gbound, f"no rows_per_core divides gbound={gbound} under {max_cores} cores"
    n_cores = gbound // rpc
    tile_m = rpc * gbound                        # pixels per core (M dim)
    # m_split: stream each core's tile_m pixels as `m_split` sub-tiles through the
    # FIFO (the core loops m_split kernel calls, one chunk in/out per call). Shrinks
    # the L1 in/out buffers by m_split — the lever that makes the large re4
    # (rpc=4, tile_m=320 -> 40KB in) GEMM fit alongside the chain in one merged ELF.
    # Default None: m_split = rpc, i.e. ONE valid row (gbound px) per sub-tile, so
    # each chunk is one whole row (the drain folds it cleanly). re8/re6 keep rpc=1/2.
    if m_split is None:
        m_split = rpc
    assert tile_m % m_split == 0, f"tile_m={tile_m} not divisible by m_split={m_split}"
    rows_per_chunk = rpc // m_split
    assert rpc % m_split == 0 and rows_per_chunk >= 1, (
        f"m_split={m_split} must divide rpc={rpc} into whole-row chunks")
    chunk_m = tile_m // m_split                   # pixels per streamed sub-tile
    input_tile_size = chunk_m * ic
    output_tile_size = chunk_m * oc
    weight_size = oc * ic + 2 * oc              # conv + BN scale/bias

    n_cols = (n_cores + cores_per_col - 1) // cores_per_col

    full_in_tile = tile_m * ic                    # per-core full input (all chunks)
    total_in = n_cores * full_in_tile             # host input BO size
    dev = NPU2()
    input_ty = np.ndarray[(input_tile_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]
    host_in_ty = np.ndarray[(total_in,), np.dtype[np.uint16]]
    host_wt_ty = weight_ty
    # OUT is the PADDED seam image — same type as halo_conv(ic=oc).img_ty.
    seam_ty = np.ndarray[(IMG_ELEMS,), np.dtype[np.uint16]]

    kern_name = "gemm_conv1x1_fused_packed_bf16"
    obj_path = os.path.join(KERNELS_DIR, f"{kern_name}.o")
    kernel = PythocKernel(kern_name, obj_path, [
        input_ty, weight_ty, output_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    # RTP: [tile_h, tile_w, ic, oc, stride, padding] = [chunk_m, 1, ic, oc, 1, 0]
    # (kernel processes one chunk_m sub-tile per call; core loops m_split chunks).
    RTP_LEN = 6
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    init_rtp = np.array([chunk_m, 1, ic, oc, 1, 0], dtype=np.int32)
    rtps = [Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
            for i in range(n_cores)]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    _M_SPLIT = m_split

    def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
        barrier.wait_for_value(1)
        t_h = my_rtp[0]; t_w = my_rtp[1]; ic_v = my_rtp[2]
        oc_v = my_rtp[3]; s_v = my_rtp[4]; p_v = my_rtp[5]
        elem_wt = of_wt.acquire(1)
        c = 0
        while c < _M_SPLIT:
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kern(elem_in, elem_wt, elem_out, t_h, t_w, ic_v, oc_v, s_v, p_v)
            of_in.release(1)
            of_out.release(1)
            c = c + 1
        of_wt.release(1)
        barrier.release_with_value(1)

    col_in_fifos, col_out_fifos, wt_fifos, workers = [], [], [], []
    for col in range(n_cols):
        cc = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_ty = np.ndarray[(cc * input_tile_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(cc * output_tile_size,), np.dtype[np.uint16]]
        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"gpo_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[input_tile_size * i for i in range(cc)],
            obj_types=[input_ty] * cc,
            names=[f"gpo_in_{col}_{i}" for i in range(cc)])
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"gpo_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_tile_size * i for i in range(cc)],
            obj_types=[output_ty] * cc,
            names=[f"gpo_out_{col}_{i}" for i in range(cc)])
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"gpo_wt_{col}")
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
        in_d1, in_d0 = _factor(input_tile_size)

        last_col = n_cols - 1
        for col in range(n_cols):
            cc = min(cores_per_col, n_cores - col * cores_per_col)
          # STREAM m_split sub-tile rounds: per round cr the column's cc cores
          # each contribute their cr-th chunk (rows_per_chunk rows). One round =
          # one column-wide fill/drain (one BD chain); m_split rounds over the
          # depth-1 FIFO (lock-paced). m_split==rpc==1 (re8) == single round.
            for cr in range(m_split):
              # input: gather core c's cr-th chunk [chunk_m, ic] HWC. Host packs
              # per-core full tile_m contiguous; chunk cr is at +cr*input_tile_size
              # within a core's block (core stride full_in_tile).
                tap_in = TensorAccessPattern(
                    (total_in,),
                    offset=col * cores_per_col * full_in_tile + cr * input_tile_size,
                    sizes=[1, cc, in_d1, in_d0],
                    strides=[0, full_in_tile, in_d0, 1])
                rt.fill(col_in_fifos[col].prod(), I, tap_in)
                # DRAIN: scatter this round's cc chunks into the PADDED image.
                # Core (col*cpc+i) owns rpc rows from r_base=(col*cpc+i)*rpc; chunk
                # cr covers rows r0=r_base+cr*rows_per_chunk .. +rows_per_chunk,
                # each row gbound*oc at row stride IMG*oc. core stride rpc*IMG*oc.
                r0 = (col * cores_per_col) * rpc + cr * rows_per_chunk
                if rows_per_chunk == 1:
                    out_d1, out_d0 = _factor(output_tile_size)
                    tap_out = TensorAccessPattern(
                        (IMG_ELEMS,),
                        offset=dst_base + ((PAD + r0) * IMG + PAD) * oc,
                        sizes=[1, cc, out_d1, out_d0],
                        strides=[0, rpc * IMG * oc, out_d0, 1])
                else:
                    tap_out = TensorAccessPattern(
                        (IMG_ELEMS,),
                        offset=dst_base + ((PAD + r0) * IMG + PAD) * oc,
                        sizes=[cc, rows_per_chunk, gbound, oc],
                        strides=[rpc * IMG * oc, IMG * oc, oc, 1])
                rt.drain(col_out_fifos[col].cons(), OUT, tap_out,
                         wait=(col == last_col and cr == m_split - 1))

    meta = dict(GRID=GRID, IMG=IMG, IMG_H=img_h, HALF_ELEMS=half_elems,
                IMG_ELEMS=IMG_ELEMS, DST_BASE=dst_base, PAD=PAD, TILE=TILE,
                ic=ic, oc=oc, gbound=gbound, tile_m=tile_m, n_cores=n_cores,
                rows_per_core=rpc, m_split=m_split, chunk_m=chunk_m,
                rows_per_chunk=rows_per_chunk, full_in_tile=full_in_tile,
                input_tile_size=input_tile_size, output_tile_size=output_tile_size,
                weight_size=weight_size, n_cols=n_cols, cores_per_col=cores_per_col)
    return Program(dev, rt).resolve_program(), meta


if __name__ == "__main__":
    ic = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    oc = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    gb = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    module, meta = gemm_pad_out(ic=ic, oc=oc, gbound=gb)
    assert module.operation.verify()
    print(module)
    print("META", meta, file=sys.stderr)
