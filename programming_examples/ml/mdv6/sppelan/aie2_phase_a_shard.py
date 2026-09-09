"""SPP9 phase-A eight-channel shard; no model dispatch integration.

Normal ABI: I[400,256], W[2,1056], O[4,400,8], M[32], all uint16.
Pool-only ABI: I[400,8], O[4,400,8], M[32]. M[0] is observed crRnd;
remaining metadata words are zero. No output-plane duplication in L1.
"""
import argparse
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.dialects import arith
from aie.extras import types as T


def generate(pool_only=False):
    ty = lambda count: np.ndarray[(count,), np.dtype[np.uint16]]
    input_type = ty(3200 if pool_only else 4096)
    weight_type, output_type, metadata_type = ty(2112), ty(12800), ty(32)
    inputs = ObjectFifo(input_type, depth=1, name="shard_input")
    outputs = ObjectFifo(output_type, depth=1, name="shard_planes")
    metadata = ObjectFifo(metadata_type, depth=1, name="shard_metadata")
    rt = Runtime()
    if pool_only:
        kernel = Kernel("phase_a_pool_only", "phase_a_shard.o",
                        [input_type, output_type, metadata_type])

        def worker(fi, fo, fm, kern):
            i = fi.acquire(1)
            o = fo.acquire(1)
            m = fm.acquire(1)
            kern(i, o, m)
            fi.release(1)
            fo.release(1)
            fm.release(1)

        core = Worker(worker, [inputs.cons(), outputs.prod(), metadata.prod(), kernel],
                      stack_size=8192)
        with rt.sequence(input_type, output_type, metadata_type) as (I, O, M):
            rt.start(core)
            rt.fill(inputs.prod(), I)
            rt.drain(outputs.cons(), O, wait=True)
            rt.drain(metadata.cons(), M, wait=True)
    else:
        weights = ObjectFifo(weight_type, depth=1, name="shard_weights")
        projection = Kernel("phase_a_project_stripe", "phase_a_shard.o",
                            [input_type, weight_type, output_type, np.int32])
        pooling = Kernel("phase_a_pool_planes", "phase_a_shard.o",
                         [output_type, metadata_type])

        def worker(fi, fw, fo, fm, conv, pool):
            w = fw.acquire(1)
            o = fo.acquire(1)
            m = fm.acquire(1)
            for stripe in range_(25):
                i = fi.acquire(1)
                conv(i, w, o, arith.index_cast(T.i32(), stripe))
                fi.release(1)
            pool(o, m)
            fw.release(1)
            fo.release(1)
            fm.release(1)

        core = Worker(worker, [inputs.cons(), weights.cons(), outputs.prod(),
                              metadata.prod(), projection, pooling], stack_size=8192)
        with rt.sequence(ty(102400), weight_type, output_type, metadata_type) as (I, W, O, M):
            rt.start(core)
            rt.fill(weights.prod(), W)
            rt.fill(inputs.prod(), I)
            rt.drain(outputs.cons(), O, wait=True)
            rt.drain(metadata.cons(), M, wait=True)
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-only", action="store_true")
    print(generate(parser.parse_args().pool_only))
