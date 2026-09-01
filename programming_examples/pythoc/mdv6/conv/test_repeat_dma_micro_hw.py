#!/usr/bin/env python3
"""Minimal repeated shim-DMA/ObjectFifo microtests.

Purpose: isolate whether IRON `TensorAccessPattern(..., sizes=[N, tile],
strides=[tile, 1])` repeat-count DMA is safe with simple depth-1 ObjectFifos
before using it in rn3pair.

Variants:
- unrolled: all tile fills/drains are per tile.
- repeat-in: A input uses one repeated fill; B and output are unrolled.
- repeat-out: A/B are unrolled; output uses one repeated drain.
- repeat-io: A input and output use repeated tasks; B is unrolled.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PYTHOC_EXAMPLES = Path(__file__).resolve().parents[2]
if str(PYTHOC_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(PYTHOC_EXAMPLES))

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import PythocKernel
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.compile import compile_mlir_module
from my_kernels import add_kernel


def build_module(tiles=8, tile_size=64, repeat_in=False, repeat_out=False, fifo_depth=1, finish_per_tile=False):
    tensor_size = tiles * tile_size
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

    kernel = PythocKernel(add_kernel, [tile_ty, tile_ty, tile_ty, np.int32])
    of_a = ObjectFifo(tile_ty, depth=fifo_depth, name="micro_in_a")
    of_b = ObjectFifo(tile_ty, depth=fifo_depth, name="micro_in_b")
    of_c = ObjectFifo(tile_ty, depth=fifo_depth, name="micro_out_c")

    def core_fn(a, b, c, kern):
        for _ in range(tiles):
            ea = a.acquire(1)
            eb = b.acquire(1)
            ec = c.acquire(1)
            kern(ea, eb, ec, tile_size)
            a.release(1)
            b.release(1)
            c.release(1)

    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), kernel])
    def sequence(A, B, C, of_a_prod, of_c_cons, of_b_prod):
        if repeat_in:
            tap_a = TensorAccessPattern(
                (tensor_size,), offset=0, sizes=[tiles, tile_size], strides=[tile_size, 1]
            )
            of_a_prod.fill(A, tap_a)
        if repeat_out:
            tap_c = TensorAccessPattern(
                (tensor_size,), offset=0, sizes=[tiles, tile_size], strides=[tile_size, 1]
            )
            of_c_cons.drain(C, tap_c, wait=True)
        for t in range(tiles):
            tg = TaskGroup() if finish_per_tile else None
            tap = TensorAccessPattern(
                (tensor_size,), offset=t * tile_size, sizes=[1, tile_size], strides=[0, 1]
            )
            if not repeat_in:
                of_a_prod.fill(A, tap, group=tg)
            of_b_prod.fill(B, tap, group=tg)
            if not repeat_out:
                of_c_cons.drain(C, tap, group=tg, wait=True)
            if tg is not None:
                tg.finish()

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, tensor_ty, of_a.prod(), of_c.cons(), of_b.prod()],
    )
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


def compile_module(module, workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    mlir_path = workdir / "kernel.mlir"
    with open(mlir_path, "w", encoding="utf-8") as f:
        print(module, file=f)
    compile_mlir_module(
        mlir_module=module,
        insts_path=str(workdir / "insts.bin"),
        xclbin_path=str(workdir / "final.xclbin"),
        work_dir=str(workdir),
        verbose=False,
    )
    return workdir / "final.xclbin", workdir / "insts.bin", mlir_path


def run_kernel(xclbin: Path, insts: Path, tiles=8, tile_size=64):
    tensor_size = tiles * tile_size
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    a_np = np.arange(tensor_size, dtype=np.int32)
    b_np = (1000 + np.arange(tensor_size, dtype=np.int32))
    a = iron.tensor(a_np, dtype=np.int32)
    b = iron.tensor(b_np, dtype=np.int32)
    c = iron.zeros(tensor_size, dtype=np.int32)
    DefaultNPURuntime.run(handle, [a, b, c])
    got = c.numpy().copy()
    exp = a_np + b_np
    np.testing.assert_array_equal(got, exp)
    return got


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["unrolled", "repeat-in", "repeat-out", "repeat-io"], default="unrolled")
    p.add_argument("--tiles", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--fifo-depth", type=int, default=1)
    p.add_argument("--finish-per-tile", action="store_true")
    p.add_argument("--workdir", default="conv/build_repeat_dma_micro")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)

    repeat_in = args.variant in {"repeat-in", "repeat-io"}
    repeat_out = args.variant in {"repeat-out", "repeat-io"}
    suffix = "_fpt" if args.finish_per_tile else ""
    wd = Path(args.workdir) / f"{args.variant}_t{args.tiles}_s{args.tile_size}_d{args.fifo_depth}{suffix}"
    module = build_module(args.tiles, args.tile_size, repeat_in, repeat_out, args.fifo_depth, args.finish_per_tile)
    xclbin, insts, mlir = compile_module(module, wd)
    print(f"built variant={args.variant} mlir={mlir}")
    if args.build_only:
        return 0
    got = run_kernel(xclbin, insts, args.tiles, args.tile_size)
    print(f"PASS: {args.variant} tiles={args.tiles} tile_size={args.tile_size} fifo_depth={args.fifo_depth} first={got[:8].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
