#!/usr/bin/env python3
"""Multicore/memtile-staged rn3 pair fused-device generator.

This is the next step after the one-worker smoke device: use the same scalar
fused rn3-pair external kernel, but route input/output through the standard
IRON memtile split/join pattern so multiple cores can process independent
output tiles in one dispatch.
"""
import argparse
import os
import sys
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.pythoc import PythocKernel
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern

KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "build"))


def rn3_pair_mc(dev, n_cores=4, tile_h=8, tile_w=8, ic=48, mid=48, ocb=4):
    input_size = (tile_h + 4) * (tile_w + 4) * ic
    w1_size = mid * ic * 9
    w2_size = ocb * mid * 9
    weight_size = w1_size + 2 * mid + w2_size + 2 * ocb
    output_size = tile_h * tile_w * ocb

    patch_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    cores_per_col = 4
    n_cols = (n_cores + cores_per_col - 1) // cores_per_col
    host_input_size = n_cores * input_size
    host_output_size = n_cores * output_size
    host_input_ty = np.ndarray[(host_input_size,), np.dtype[np.uint16]]
    host_output_ty = np.ndarray[(host_output_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "rn3_pair_fused_bf16",
        os.path.join(KERNELS_DIR, "rn3_pair_fused_bf16.o"),
        [patch_ty, weight_ty, output_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_in, of_wt, of_out, kern, barrier):
        barrier.wait_for_value(1)
        elem_wt = of_wt.acquire(1)
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        kern(elem_in, elem_wt, elem_out, tile_h, tile_w, ic, mid, ocb)
        of_in.release(1)
        of_out.release(1)
        of_wt.release(1)
        barrier.release_with_value(1)

    col_in_fifos = []
    col_out_fifos = []
    wt_fifos = []
    workers = []

    for col in range(n_cols):
        cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
        col_in_size = cores_this_col * input_size
        col_out_size = cores_this_col * output_size
        col_in_ty = np.ndarray[(col_in_size,), np.dtype[np.uint16]]
        col_out_ty = np.ndarray[(col_out_size,), np.dtype[np.uint16]]

        col_in_fifo = ObjectFifo(col_in_ty, depth=1, name=f"rn3p_col_in_{col}")
        in_splits = col_in_fifo.cons().split(
            offsets=[input_size * i for i in range(cores_this_col)],
            obj_types=[patch_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"rn3p_input_{col}_{i}" for i in range(cores_this_col)],
        )

        col_out_fifo = ObjectFifo(col_out_ty, depth=1, name=f"rn3p_col_out_{col}")
        out_joins = col_out_fifo.prod().join(
            offsets=[output_size * i for i in range(cores_this_col)],
            obj_types=[output_ty] * cores_this_col,
            depths=[1] * cores_this_col,
            names=[f"rn3p_output_{col}_{i}" for i in range(cores_this_col)],
        )

        wt_fifo = ObjectFifo(weight_ty, depth=1, name=f"rn3p_weights_{col}")
        col_in_fifos.append(col_in_fifo)
        col_out_fifos.append(col_out_fifo)
        wt_fifos.append(wt_fifo)

        for i in range(cores_this_col):
            g = col * cores_per_col + i
            workers.append(Worker(
                core_fn,
                [in_splits[i].cons(), wt_fifo.cons(), out_joins[i].prod(), kernel, barriers[g]],
                stack_size=4096,
            ))

    def sequence(I, W, O, wt_fifos_prods, col_in_fifos_prods, col_out_fifos_conss):
        for b in barriers:
            b.set(1)
        for wf_i, wf in enumerate(wt_fifos):
            wt_fifos_prods[wf_i].fill(W)
        for col in range(n_cols):
            cores_this_col = min(cores_per_col, n_cores - col * cores_per_col)
            col_in_size = cores_this_col * input_size
            col_out_size = cores_this_col * output_size
            tap_in = TensorAccessPattern(
                (host_input_size,),
                offset=col * cores_per_col * input_size,
                sizes=[1, col_in_size],
                strides=[0, 1],
            )
            tap_out = TensorAccessPattern(
                (host_output_size,),
                offset=col * cores_per_col * output_size,
                sizes=[1, col_out_size],
                strides=[0, 1],
            )
            col_in_fifos_prods[col].fill(I, tap_in)
            col_out_fifos_conss[col].drain(O, tap_out, wait=True)

    rt = Runtime(
        sequence,
        [host_input_ty, weight_ty, host_output_ty, [__f.prod() for __f in wt_fifos], [__f.prod() for __f in col_in_fifos], [__f.cons() for __f in col_out_fifos]],
    )

    return Program(dev, rt, workers=[*workers]).resolve_program()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("n_cores", nargs="?", type=int, default=4)
    p.add_argument("tile_h", nargs="?", type=int, default=8)
    p.add_argument("tile_w", nargs="?", type=int, default=8)
    p.add_argument("ic", nargs="?", type=int, default=48)
    p.add_argument("mid", nargs="?", type=int, default=48)
    p.add_argument("ocb", nargs="?", type=int, default=4)
    args = p.parse_args(argv)
    print(rn3_pair_mc(NPU2(), args.n_cores, args.tile_h, args.tile_w, args.ic, args.mid, args.ocb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
