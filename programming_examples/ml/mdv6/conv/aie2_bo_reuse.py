"""Experimental 3-tile HWC single stage / on-chip two-stage comparison.

Each tile is independent 8x8x16, packed tile,H,W,C with no OC blocking/padding.
Weights have O,I then BN scale then BN bias. Two stages use distinct buffers.
Each worker acquires its weights once across all three tiles per submission.
"""
import sys
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2


def build(stages):
    if stages not in (1, 2):
        raise ValueError("stages must be 1 or 2")
    tile_ty = np.ndarray[(1024,), np.dtype[np.uint16]]
    all_ty = np.ndarray[(3072,), np.dtype[np.uint16]]
    wt_ty = np.ndarray[(288,), np.dtype[np.uint16]]
    kernel = Kernel("conv1x1_fused_packed_bf16", "rep_elan_bf16.o",
                    [tile_ty, wt_ty, tile_ty] + [np.int32] * 6)
    edges = [ObjectFifo(tile_ty, name=f"activation{i}") for i in range(stages + 1)]
    weights = [ObjectFifo(wt_ty, name=f"weights{i}") for i in range(stages)]

    def worker(inp, wt, out, kern):
        w = wt.acquire(1)
        for _ in range_(3):
            x = inp.acquire(1)
            y = out.acquire(1)
            kern(x, w, y, 8, 8, 16, 16, 1, 0)
            inp.release(1)
            out.release(1)
        wt.release(1)

    workers = [Worker(worker, [edges[i].cons(), weights[i].cons(),
                              edges[i + 1].prod(), kernel], stack_size=4096)
               for i in range(stages)]
    rt = Runtime()
    with rt.sequence(all_ty, *([wt_ty] * stages), all_ty) as args:
        rt.start(*workers)
        for i in range(stages):
            rt.fill(weights[i].prod(), args[i + 1])
        rt.fill(edges[0].prod(), args[0])
        rt.drain(edges[-1].cons(), args[-1], wait=True)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    print(build(int(sys.argv[1])))
