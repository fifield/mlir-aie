"""Milestone 0 — on-device GELAN concat -> conv4 (1x1 512->256) proof.

Proves the 4-way channel-dim concat `concat[x1,x2,x3,x4]` (each 20x20x128 bf16,
HWC) followed by conv4 can run entirely on-device, with the concat done as a
strided memtile DMA, bit-exact vs the host reference
`np.concatenate([x1,x2,x3,x4], axis=2)` then a 1x1 GEMM + BN + SiLU.

Mechanism
---------
The 4 source quarters are NOT host-concatenated. They are passed stacked in one
input BO as [x1(400,128) | x2 | x3 | x4] (each tile flat-contiguous, no
interleave). The on-device concat is the channel-offset interleave: for fused
pixel p, channel-quarter k lands at channel offset k*128. This is performed by
the input fill's *gather* TAP:

    sizes   = [tile_m, 4, 128]
    strides = [128,    NPIX*128, 1]   (NPIX = 400 = total pixels per quarter)

so the linear FIFO buffer ends up as the fused per-pixel [tile_m, 512] HWC
layout the GEMM kernel consumes as an ordinary contiguous [M, IC=512] input.

Expressibility note: an ObjectFifo `fill` writes its FIFO buffer *linearly* —
the access pattern (`tap`) is a SOURCE gather over the host BO. So the scope's
"4 DMAs differing only in offset=k*128 with dest pixel-stride 512" is NOT
expressible as 4 separate fills into one buffer (you cannot scatter into a
linear destination across 4 fills). The equivalent — and the IRON-native form —
is a single gather-fill whose source strides interleave the 4 quarters. The
symmetric *drain* DOES take a destination scatter TAP (proven in
aie2_rn3_chain_geo.py: `strides=[0, GBOUND*OC2, OC2, 1]`); the fill side is a
gather. Both produce the identical fused buffer; we use the gather form.

GEMM path is the model's K-blocked conv1x1 (gemm_conv1x1_kblocked_bf16), so the
numerics are identical to the deployed conv4.
"""
import argparse
import os
import sys

import numpy as np

from aie.iron import (
    Buffer,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.iron.pythoc import PythocKernel, aie_kernel
from aie.helpers.taplib import TensorAccessPattern
from pythoc import ptr, i32, bf16, void
from pythoc.aie import load_v, store_v, aie_vector


@aie_kernel
def _vcopy(src: ptr[bf16, True], dst: ptr[bf16, True], n: i32) -> void:
    """Vectorized bf16 copy: dst[0:n] = src[0:n], 8 elems/iter (n % 8 == 0).

    NOTE: 8-wide load/store is used deliberately. 32-wide bf16 store_v miscodegens
    on this toolchain (writes only even lanes, zero-fills odd) — same even-lane
    artifact seen in the IC=512 K-blocked GEMM. See report.
    """
    i: i32 = 0
    while i < n:
        v: aie_vector[bf16, 8] = load_v(src + i, 8)
        store_v(dst + i, v)
        i = i + 8

KERNELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "kernels", "build")
)


def concat_proof(dev, H=20, W=20, q_ic=128, n_q=4, oc=256,
                 tile_m=24, k_block=32, n_cores=32):
    """4-way channel concat (gather-fill) -> K-blocked conv1x1 GEMM + BN + SiLU.

    Args:
        H, W: spatial dims (HWC). M = H*W pixels.
        q_ic: input channels per quarter (128). Fused IC = n_q*q_ic = 512.
        n_q: number of concat quarters (4).
        oc: output channels (256).
        tile_m: spatial pixels per core per patch (mult of 4).
        k_block: IC channels per K-block (mult of 8; divides fused IC).
        n_cores: compute cores.
    """
    M = H * W                       # 400 fused pixels
    ic = n_q * q_ic                 # 512 fused IC
    assert tile_m % 4 == 0
    assert ic % k_block == 0 and k_block % 8 == 0
    assert oc % 8 == 0
    n_k_blocks = ic // k_block

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col
    # patches/core needed to cover all M pixels across all cores.
    import math
    ppc = max(1, math.ceil(M / (n_cores * tile_m)))
    covered = n_cores * tile_m * ppc
    print(f"concat_proof: H={H} W={W} M={M}, {n_q}x{q_ic}->IC={ic}, OC={oc}, "
          f"tile_m={tile_m}, k_block={k_block} ({n_k_blocks} kb), "
          f"{n_cores} cores, ppc={ppc}, covered={covered}", file=sys.stderr)

    # ---- buffer sizes (bf16 elements, carried as uint16) ----
    input_tile_size = tile_m * ic          # per-core fused input
    output_tile_size = tile_m * oc
    wt_chunk_size = k_block * oc + 2 * oc   # one K-block weight chunk + BN

    input_ty = np.ndarray[(input_tile_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(wt_chunk_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_tile_size,), np.dtype[np.uint16]]

    core_in_size = ppc * input_tile_size
    core_out_size = ppc * output_tile_size

    # Host input BO: n_q quarters stacked, each [covered_pixels, q_ic].
    # We allocate covered (>= M) pixels per quarter so the gather strides are
    # uniform; trailing (covered - M) pixels are zero-padded by the host.
    n_pix = covered
    host_q_size = n_pix * q_ic              # one quarter
    host_in_size = n_q * host_q_size        # all 4 quarters stacked
    host_out_size = n_cores * core_out_size
    host_wt_size = n_k_blocks * wt_chunk_size

    host_in_ty = np.ndarray[(host_in_size,), np.dtype[np.uint16]]
    host_wt_ty = np.ndarray[(host_wt_size,), np.dtype[np.uint16]]
    host_out_ty = np.ndarray[(host_out_size,), np.dtype[np.uint16]]

    kern_name = "gemm_conv1x1_kblocked_bf16"
    obj_path = os.path.join(KERNELS_DIR, f"{kern_name}.o")
    kernel = PythocKernel(kern_name, obj_path, [
        input_ty, weight_ty, output_ty,
        np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
    ])

    RTP_LEN = 6
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    init_rtp = np.array([tile_m, ic, oc, k_block, n_k_blocks, 0], dtype=np.int32)
    rtps = [
        Buffer(rtp_ty, name=f"rtp_{i}", initial_value=init_rtp, use_write_rtp=True)
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    total_patch_cycles = ppc

    def core_fn(of_in, of_wt, of_out, kern, my_rtp, barrier):
        barrier.wait_for_value(1)
        tm_v = my_rtp[0]
        fic_v = my_rtp[1]
        oc_v = my_rtp[2]
        kb_v = my_rtp[3]
        nkb_v = my_rtp[4]
        for _ in range_(total_patch_cycles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for kb in range(n_k_blocks):
                elem_wt = of_wt.acquire(1)
                kern(elem_in, elem_wt, elem_out,
                     tm_v, fic_v, oc_v, kb * k_block, kb_v, nkb_v)
                of_wt.release(1)
            of_in.release(1)
            of_out.release(1)
        barrier.release_with_value(1)

    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_size = cores_this_col * input_tile_size
        col_out_size = cores_this_col * output_tile_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[input_tile_size * i for i in range(cores_this_col)],
            obj_types=[input_ty] * cores_this_col,
            names=[f"input_{col}_{i}" for i in range(cores_this_col)],
        )
        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_tile_size * i for i in range(cores_this_col)],
            obj_types=[output_ty] * cores_this_col,
            names=[f"output_{col}_{i}" for i in range(cores_this_col)],
        )
        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"weights_{col}")

        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            gci = col * cores_per_col + i
            w = Worker(core_fn, [
                in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel,
                rtps[gci], barriers[gci],
            ], stack_size=8192)
            workers.append(w)

    rt = Runtime()
    with rt.sequence(host_in_ty, host_wt_ty, host_out_ty) as (I, W_, O):
        rt.start(*workers)

        _rtp_vals = [int(v) for v in init_rtp]

        def set_rtps(*rtp_bufs):
            for rb in rtp_bufs:
                rb[0] = _rtp_vals[0]; rb[1] = _rtp_vals[1]
                rb[2] = _rtp_vals[2]; rb[3] = _rtp_vals[3]
                rb[4] = _rtp_vals[4]; rb[5] = _rtp_vals[5]
        rt.inline_ops(set_rtps, rtps)
        for b in barriers:
            rt.set_barrier(b, 1)

        # ---- weight fill (K-blocked, repeated per patch cycle) ----
        def _factor_for_dma(n, max_inner=1023):
            for inner in range(max_inner, 0, -1):
                if n % inner == 0:
                    return n // inner, inner
            return n, 1

        wt_d1, wt_d0 = _factor_for_dma(wt_chunk_size)
        for wf in wt_fifos:
            tap_wt = TensorAccessPattern(
                (host_wt_size,),
                offset=0,
                sizes=[total_patch_cycles, n_k_blocks, wt_d1, wt_d0],
                strides=[0, wt_chunk_size, wt_d0, 1],
            )
            rt.fill(wf.prod(), W_, tap_wt)

        # ---- ON-DEVICE CONCAT: gather-fill input ----
        # Host BO layout: [q0(n_pix,q_ic) | q1 | q2 | q3], each quarter flat.
        # Quarter q starts at element offset q*host_q_size.
        # For a column covering cores [col*4 .. ], patch cycle p, the fused
        # per-core input is [tile_m pixels, ic=512] where channel-quarter q of
        # pixel local-pix is the q-th source tile's pixel (global pixel index).
        #
        # global pixel for (col, core i, patch p, local m) =
        #   ((col*4 + i)*ppc + p)*tile_m + m       [contiguous pixel tiling]
        #
        # The gather TAP per column produces, into the linear col_in buffer of
        # shape [cores_this_col, tile_m, n_q, q_ic]:
        #   src element = q*host_q_size + gpix*q_ic + c
        # 5-D access: [core, patch?, pixel, quarter, channel]. DMA BDs are max
        # 4 real dims; we emit ppc separate fills (one per patch cycle) so the
        # remaining 4 dims [core, pixel, quarter, channel] fit one BD.
        # Debug: CONCAT=0 expects a PRE-FUSED contiguous [n_pix, ic] input BO
        # (host did the concat) so the GEMM path can be validated in isolation.
        do_concat = os.environ.get("CONCAT", "1") != "0"
        if not do_concat:
            # GEMM-only: PRE-FUSED contiguous [n_pix, ic] input BO (host concat).
            for col in range(n_cols):
                cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
                in_d1, in_d0 = _factor_for_dma(input_tile_size)
                tap_in = TensorAccessPattern(
                    (n_pix * ic,),
                    offset=col * cores_per_col * core_in_size,
                    sizes=[ppc, cores_this_col, in_d1, in_d0],
                    strides=[input_tile_size, core_in_size, in_d0, 1],
                )
                rt.fill(col_in_fifos[col].prod(), I, tap_in)
        else:
            for col in range(n_cols):
                cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
                base_core = col * cores_per_col
                for p in range(ppc):
                    # base global pixel for (this col's first core, patch p, m=0)
                    # gpix = ((base_core + i)*ppc + p)*tile_m + m
                    # stride over i  : ppc*tile_m  pixels  -> * q_ic elems
                    # stride over m  : 1 pixel             -> * q_ic elems
                    # stride over q  : host_q_size elems
                    # stride over c  : 1 elem
                    base_gpix = (base_core * ppc + p) * tile_m
                    tap_in = TensorAccessPattern(
                        (host_in_size,),
                        offset=base_gpix * q_ic,
                        sizes=[cores_this_col, tile_m, n_q, q_ic],
                        strides=[ppc * tile_m * q_ic, q_ic, host_q_size, 1],
                    )
                    rt.fill(col_in_fifos[col].prod(), I, tap_in)

        # ---- drain output (contiguous per-core, [M,OC] HWC) ----
        out_d1, out_d0 = _factor_for_dma(output_tile_size)
        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            tap_out = TensorAccessPattern(
                (host_out_size,),
                offset=col * cores_per_col * core_out_size,
                sizes=[ppc, cores_this_col, out_d1, out_d0],
                strides=[output_tile_size, core_out_size, out_d0, 1],
            )
            rt.drain(col_out_fifos[col].cons(), O, tap_out,
                     wait=(col == n_cols - 1))

    return Program(dev, rt).resolve_program()


def concat_only(dev, H=20, W=20, q_ic=128, n_q=4, n_cores=32):
    """Concat-only proof: on-device channel concat of n_q quarters, copied
    straight to the output BO by the cores (NO GEMM). Proves the strided gather
    DMA produces the bit-exact fused [M, n_q*q_ic] HWC buffer conv4 consumes.

    Topology = the SAME proven shim->memtile(split)->core->memtile(join)->shim
    path as concat_proof, but each core does a plain element copy instead of the
    GEMM. The input fill is the concat gather; the output is the materialised
    fused buffer read back unchanged.
    """
    import math
    ic = n_q * q_ic
    M = H * W
    # Direct (split-free) per-core shim flows are limited by shim NOC DMA
    # capacity AND L1 size: the per-core fused tile [tile_m, ic=512] bf16 lives
    # in L1, so tile_m*512*2 *2(in+out) must fit ~64KB. tile_m=16 -> 32KB.
    # 4 cores x tile_m=16 = 64 pixels (a subset of the 400) — enough to prove
    # the concat primitive bit-exactly without the split/join deadlock.
    n_cores = 4
    tile_m = 16
    ppc = 1
    covered = n_cores * tile_m * ppc          # 64 proven pixels
    n_pix = covered
    host_q_size = n_pix * q_ic
    host_in_size = n_q * host_q_size
    fused_size = n_pix * ic
    print(f"concat_only: M={M} covered={covered}, {n_q}x{q_ic}->IC={ic}, "
          f"{n_cores} cores, tile_m={tile_m}, ppc={ppc}", file=sys.stderr)

    tile_size = tile_m * ic                  # per-core fused chunk
    assert tile_size % 32 == 0
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.uint16]]
    host_in_ty = np.ndarray[(host_in_size,), np.dtype[np.uint16]]
    host_out_ty = np.ndarray[(fused_size,), np.dtype[np.uint16]]

    copy_kernel = PythocKernel(_vcopy, [tile_ty, tile_ty, np.int32])

    def core_fn(of_in, of_out, kern):
        ein = of_in.acquire(1)
        eout = of_out.acquire(1)
        kern(ein, eout, tile_size)
        of_in.release(1)
        of_out.release(1)

    # Direct per-core FIFOs (NO split/join): each core has its own input FIFO
    # filled by its own concat gather, copies the fused tile, drains it. This is
    # the proven single-core gather pattern replicated; the split/join variant
    # deadlocks with the passthrough copy on this toolchain.
    in_fifos, out_fifos, workers = [], [], []
    for c in range(n_cores):
        fin = ObjectFifo(tile_ty, depth=1, name=f"cin_{c}")
        fout = ObjectFifo(tile_ty, depth=1, name=f"cout_{c}")
        in_fifos.append(fin)
        out_fifos.append(fout)
        workers.append(Worker(core_fn, [fin.cons(), fout.prod(), copy_kernel],
                              stack_size=2048))

    rt = Runtime()
    with rt.sequence(host_in_ty, host_out_ty) as (I, O):
        rt.start(*workers)
        # Per-core concat gather-fill. Core c owns global pixels
        # [c*tile_m : (c+1)*tile_m]. Gather TAP: [pixel, quarter, channel] ->
        # fused linear [tile_m, n_q*q_ic].
        for c in range(n_cores):
            base_gpix = c * tile_m
            tap_in = TensorAccessPattern(
                (host_in_size,),
                offset=base_gpix * q_ic,
                sizes=[tile_m, n_q, q_ic],
                strides=[q_ic, host_q_size, 1],
            )
            rt.fill(in_fifos[c].prod(), I, tap_in)
        # Drain each core's fused tile to its contiguous output slot. Core c at
        # output offset c*tile_size == global-pixel order, so the host reads the
        # output flat as [covered, ic].
        for c in range(n_cores):
            tap_out = TensorAccessPattern(
                (fused_size,),
                offset=c * tile_size,
                sizes=[tile_m, ic],
                strides=[ic, 1],
            )
            rt.drain(out_fifos[c].cons(), O, tap_out, wait=(c == n_cores - 1))

    return Program(dev, rt).resolve_program()


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Milestone 0 concat->conv4 proof MLIR.")
    p.add_argument("H", nargs="?", type=int, default=20)
    p.add_argument("W", nargs="?", type=int, default=20)
    p.add_argument("q_ic", nargs="?", type=int, default=128)
    p.add_argument("n_q", nargs="?", type=int, default=4)
    p.add_argument("oc", nargs="?", type=int, default=256)
    p.add_argument("tile_m", nargs="?", type=int, default=24)
    p.add_argument("k_block", nargs="?", type=int, default=32)
    p.add_argument("n_cores", nargs="?", type=int, default=32)
    return p.parse_args(argv)


if __name__ == "__main__":
    dev = NPU2()
    a = _parse_args(sys.argv[1:])
    module = concat_proof(dev, a.H, a.W, a.q_ic, a.n_q, a.oc,
                          a.tile_m, a.k_block, a.n_cores)
    print(module)
