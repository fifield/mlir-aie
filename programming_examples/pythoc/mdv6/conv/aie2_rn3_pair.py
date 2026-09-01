#!/usr/bin/env python3
"""Minimal rn3 pair fused-device generator.

Correctness prototype only: one worker, one input patch, one fused weight slot,
one output tile. The external kernel is scalar and uses simple OIHW weights.
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


def rn3_pair(dev, tile_h=8, tile_w=8, ic=4, mid=4, ocb=4):
    input_size = (tile_h + 4) * (tile_w + 4) * ic
    w1_size = mid * ic * 9
    w2_size = ocb * mid * 9
    weight_size = w1_size + 2 * mid + w2_size + 2 * ocb
    output_size = tile_h * tile_w * ocb

    input_ty = np.ndarray[(input_size,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(weight_size,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(output_size,), np.dtype[np.uint16]]

    kernel = PythocKernel(
        "rn3_pair_fused_bf16",
        os.path.join(KERNELS_DIR, "rn3_pair_fused_bf16.o"),
        [input_ty, weight_ty, output_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    of_in = ObjectFifo(input_ty, depth=1, name="rn3p_in")
    of_wt = ObjectFifo(weight_ty, depth=1, name="rn3p_wt")
    of_out = ObjectFifo(output_ty, depth=1, name="rn3p_out")
    barrier = WorkerRuntimeBarrier()

    def core_fn(in_cons, wt_cons, out_prod, kern, b):
        b.wait_for_value(1)
        elem_in = in_cons.acquire(1)
        elem_wt = wt_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        kern(elem_in, elem_wt, elem_out, tile_h, tile_w, ic, mid, ocb)
        in_cons.release(1)
        wt_cons.release(1)
        out_prod.release(1)
        b.release_with_value(1)

    worker = Worker(core_fn, [of_in.cons(), of_wt.cons(), of_out.prod(), kernel, barrier], stack_size=4096)

    def sequence(I, W, O, of_in_prod, of_wt_prod, of_out_cons):
        barrier.set(1)
        tap_in = TensorAccessPattern((input_size,), offset=0, sizes=[1, input_size], strides=[0, 1])
        tap_wt = TensorAccessPattern((weight_size,), offset=0, sizes=[1, weight_size], strides=[0, 1])
        tap_out = TensorAccessPattern((output_size,), offset=0, sizes=[1, output_size], strides=[0, 1])
        of_in_prod.fill(I, tap_in)
        of_wt_prod.fill(W, tap_wt)
        of_out_cons.drain(O, tap_out, wait=True)

    rt = Runtime(
        sequence,
        [input_ty, weight_ty, output_ty, of_in.prod(), of_wt.prod(), of_out.cons()],
    )

    return Program(dev, rt, workers=[worker]).resolve_program()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("tile_h", nargs="?", type=int, default=8)
    p.add_argument("tile_w", nargs="?", type=int, default=8)
    p.add_argument("ic", nargs="?", type=int, default=4)
    p.add_argument("mid", nargs="?", type=int, default=4)
    p.add_argument("ocb", nargs="?", type=int, default=4)
    args = p.parse_args(argv)
    print(rn3_pair(NPU2(), args.tile_h, args.tile_w, args.ic, args.mid, args.ocb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
