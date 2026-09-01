#!/usr/bin/env python3
"""Memtile split/join + task-group ObjectFifo microtest.

This isolates the rn3pair task-group hang from rn3pair compute and weight FIFOs.
It builds a 4-worker int32 add over split input FIFOs and a joined output FIFO:

  A full tile -> split into 4 worker chunks
  B full tile -> split into 4 worker chunks
  4 worker chunk outputs -> joined full output tile

Variants compare plain unrolled runtime tasks against one task group per tile.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU2
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from my_kernels import add_kernel


def build_module(tiles=4, chunk_size=64, n_workers=4, finish_per_tile=False, persistent_weight=False, use_barrier=False):
    full_size = chunk_size * n_workers
    tensor_size = tiles * full_size
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
    full_ty = np.ndarray[(full_size,), np.dtype[np.int32]]
    chunk_ty = np.ndarray[(chunk_size,), np.dtype[np.int32]]

    kernel = PythocKernel(add_kernel, [chunk_ty, chunk_ty, chunk_ty, np.int32])

    in_a_full = ObjectFifo(full_ty, depth=1, name="sj_in_a_full")
    in_a_chunks = in_a_full.cons().split(
        offsets=[i * chunk_size for i in range(n_workers)],
        obj_types=[chunk_ty] * n_workers,
        depths=[1] * n_workers,
        names=[f"sj_in_a_{i}" for i in range(n_workers)],
    )
    in_b_full = ObjectFifo(full_ty, depth=1, name="sj_in_b_full")
    in_b_chunks = in_b_full.cons().split(
        offsets=[i * chunk_size for i in range(n_workers)],
        obj_types=[chunk_ty] * n_workers,
        depths=[1] * n_workers,
        names=[f"sj_in_b_{i}" for i in range(n_workers)],
    )
    out_full = ObjectFifo(full_ty, depth=1, name="sj_out_full")
    out_chunks = out_full.prod().join(
        offsets=[i * chunk_size for i in range(n_workers)],
        obj_types=[chunk_ty] * n_workers,
        depths=[1] * n_workers,
        names=[f"sj_out_{i}" for i in range(n_workers)],
    )
    barriers = [WorkerRuntimeBarrier() for _ in range(n_workers)]

    def core_fn(a, b, c, kern, barrier):
        if use_barrier:
            barrier.wait_for_value(1)
        if persistent_weight:
            eb = b.acquire(1)
            for _ in range(tiles):
                ea = a.acquire(1)
                ec = c.acquire(1)
                kern(ea, eb, ec, chunk_size)
                a.release(1)
                c.release(1)
            b.release(1)
        else:
            for _ in range(tiles):
                ea = a.acquire(1)
                eb = b.acquire(1)
                ec = c.acquire(1)
                kern(ea, eb, ec, chunk_size)
                a.release(1)
                b.release(1)
                c.release(1)
        if use_barrier:
            barrier.release_with_value(1)

    workers = [
        Worker(core_fn, [in_a_chunks[i].cons(), in_b_chunks[i].cons(), out_chunks[i].prod(), kernel, barriers[i]], stack_size=4096)
        for i in range(n_workers)
    ]

    def sequence(A, B, C, in_b_full_prod, in_a_full_prod, out_full_cons):
        if use_barrier:
            for b in barriers:
                b.set(1)
        if persistent_weight:
            weight_tg = TaskGroup() if finish_per_tile else None
            weight_tap = TensorAccessPattern((full_size,), offset=0, sizes=[1, full_size], strides=[0, 1])
            in_b_full_prod.fill(B, weight_tap, group=weight_tg)
            if weight_tg is not None:
                weight_tg.finish()
        for t in range(tiles):
            tg = TaskGroup() if finish_per_tile else None
            tap = TensorAccessPattern(
                (tensor_size,), offset=t * full_size, sizes=[1, full_size], strides=[0, 1]
            )
            in_a_full_prod.fill(A, tap, group=tg)
            if not persistent_weight:
                in_b_full_prod.fill(B, tap, group=tg)
            out_full_cons.drain(C, tap, group=tg, wait=True)
            if tg is not None:
                tg.finish()

    rt = Runtime(
        sequence,
        [tensor_ty, full_ty if persistent_weight else tensor_ty, tensor_ty, in_b_full.prod(), in_a_full.prod(), out_full.cons()],
    )

    return Program(NPU2(), rt, workers=[*workers]).resolve_program()


def compile_module(module, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mlir_path = workdir / "kernel.mlir"
    mlir_path.write_text(str(module))
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(workdir / "insts.bin"),
        xclbin_path=str(workdir / "final.xclbin"),
        work_dir=str(workdir),
        verbose=False,
    )
    return workdir / "final.xclbin", workdir / "insts.bin", mlir_path


def run_kernel(xclbin, insts, tiles, full_size, persistent_weight=False):
    tensor_size = tiles * full_size
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    a_np = np.arange(tensor_size, dtype=np.int32)
    if persistent_weight:
        b_np = np.arange(full_size, dtype=np.int32) + 1000
        expected = a_np + np.tile(b_np, tiles)
    else:
        b_np = np.arange(tensor_size, dtype=np.int32) + 1000
        expected = a_np + b_np
    a = iron.tensor(a_np, dtype=np.int32)
    b = iron.tensor(b_np, dtype=np.int32)
    c = iron.zeros(tensor_size, dtype=np.int32)
    DefaultNPURuntime.run(handle, [a, b, c])
    got = c.numpy().copy()
    if not np.array_equal(got, expected):
        idx = int(np.argmax(got != expected))
        raise AssertionError(f"mismatch at {idx}: got={got[idx]} expected={expected[idx]}")
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--finish-per-tile", action="store_true")
    p.add_argument("--persistent-weight", action="store_true")
    p.add_argument("--barrier", action="store_true")
    p.add_argument("--workdir", default="conv/build_split_join_taskgroup_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)

    suffix = "_fpt" if args.finish_per_tile else ""
    suffix += "_pw" if args.persistent_weight else ""
    suffix += "_bar" if args.barrier else ""
    wd = Path(args.workdir) / f"t{args.tiles}_c{args.chunk_size}_w{args.n_workers}{suffix}"
    module = build_module(args.tiles, args.chunk_size, args.n_workers, args.finish_per_tile, args.persistent_weight, args.barrier)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built split_join tiles={args.tiles} workers={args.n_workers} finish_per_tile={args.finish_per_tile} persistent_weight={args.persistent_weight} barrier={args.barrier} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.tiles, args.chunk_size * args.n_workers, args.persistent_weight)
    print(f"PASS: split_join tiles={args.tiles} workers={args.n_workers} finish_per_tile={args.finish_per_tile} persistent_weight={args.persistent_weight} first={got[:8].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
