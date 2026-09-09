"""One-worker four-row IC16/OC8 probe of production KB8 partial accumulation."""
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def program():
    input_ty = np.ndarray[(64,), np.dtype[np.uint16]]
    weight_ty = np.ndarray[(192,), np.dtype[np.uint16]]
    output_ty = np.ndarray[(704,), np.dtype[np.uint16]]
    kernel = Kernel('kblocked_sparse_probe', 'kblocked_sparse_probe.o',
                    [input_ty, weight_ty, output_ty])
    inp = ObjectFifo(input_ty, depth=1, name='input')
    wt = ObjectFifo(weight_ty, depth=1, name='weights')
    out = ObjectFifo(output_ty, depth=1, name='output')

    def worker(fi, fw, fo, kern):
        w = fw.acquire(1)
        i = fi.acquire(1)
        o = fo.acquire(1)
        kern(i, w, o)
        fi.release(1)
        fw.release(1)
        fo.release(1)

    core = Worker(worker, [inp.cons(), wt.cons(), out.prod(), kernel], stack_size=8192)
    rt = Runtime()
    with rt.sequence(input_ty, weight_ty, output_ty) as (I, W, O):
        rt.start(core)
        rt.fill(wt.prod(), W)
        rt.fill(inp.prod(), I)
        rt.drain(out.cons(), O, wait=True)
    return Program(NPU2(), rt).resolve_program()


if __name__ == '__main__':
    print(program())
