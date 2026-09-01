#!/usr/bin/env python3
"""Distilled repro: iteration-counted 4D fill + 4D drain on one shim column.

One core, two FIFOs, no compute. Suspected hang when BOTH the shim MM2S
(fill) and S2MM (drain) BDs carry iteration counts. Variants:

  --fill-linear / --drain-linear : flatten one side's TAP
  --same-bo                      : fill and drain share BO 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYE = Path(__file__).resolve().parents[2]
if str(PYE) not in sys.path:
    sys.path.insert(0, str(PYE))

from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.compile import compile_mlir_module
from conv.resident_xclbin_runner import ResidentXCLBinRunner

BO = 65536          # u16 elems per host buffer
IN_ELEM = 4608      # 4 chunks x 1152
OUT_ELEM = 2048     # 8 chunks x 256


def build(fill_linear: bool, drain_linear: bool, same_bo: bool):
    in_ty = np.ndarray[(IN_ELEM,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(OUT_ELEM,), np.dtype[np.uint16]]
    bo_ty = np.ndarray[(BO,), np.dtype[np.uint16]]

    fin = ObjectFifo(in_ty, depth=1, name="dt_in")
    fout = ObjectFifo(out_ty, depth=1, name="dt_out")

    def core_fn(a, o):
        a.acquire(1)
        o.acquire(1)
        a.release(1)
        o.release(1)

    w = Worker(core_fn, [fin.cons(), fout.prod()])

    fill_tap = (TensorAccessPattern((BO,), 0, [1, IN_ELEM], [0, 1]) if fill_linear
                else TensorAccessPattern((BO,), 0, [4, 1, 1, 1152], [2048, 0, 0, 1]))
    import os
    dsz = [int(x) for x in os.environ.get("DSIZES", "8,1,256").split(",")]
    dst = [int(x) for x in os.environ.get("DSTRIDES", "512,0,1").split(",")]
    drain_tap = (TensorAccessPattern((BO,), 0, [1, OUT_ELEM], [0, 1]) if drain_linear
                 else TensorAccessPattern((BO,), 0, dsz, dst))

    if same_bo:

        def sequence(A, fin_prod, fout_cons):
            tg = TaskGroup()
            fin_prod.fill(A, fill_tap, group=tg)
            fout_cons.drain(A, drain_tap, group=tg, wait=True)
            tg.finish()

        rt = Runtime(sequence, [bo_ty, fin.prod(), fout.cons()])
    else:

        def sequence(A, B, fin_prod, fout_cons):
            tg = TaskGroup()
            fin_prod.fill(A, fill_tap, group=tg)
            fout_cons.drain(B, drain_tap, group=tg, wait=True)
            tg.finish()

        rt = Runtime(sequence, [bo_ty, bo_ty, fin.prod(), fout.cons()])

    return Program(NPU2(), rt, workers=[w]).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fill-linear", action="store_true")
    p.add_argument("--drain-linear", action="store_true")
    p.add_argument("--same-bo", action="store_true")
    a = p.parse_args()

    import os
    shp = os.environ.get("DSIZES", "8,1,256").replace(",", "x")
    tag = f"{shp}_f{'L' if a.fill_linear else '4'}_d{'L' if a.drain_linear else '4'}_b{'1' if a.same_bo else '2'}"
    wd = Path(__file__).parent / "build_dualtap" / tag
    (wd / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = wd / "final.xclbin", wd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        compile_mlir_module(mlir_module=build(a.fill_linear, a.drain_linear, a.same_bo),
                            insts_path=str(insts), xclbin_path=str(xclbin),
                            work_dir=str(wd / "work"), verbose=False)
    runner = ResidentXCLBinRunner(xclbin, insts)
    args = [np.zeros(BO, np.uint16)] if a.same_bo else [np.zeros(BO, np.uint16), np.zeros(BO, np.uint16)]
    try:
        runner.run(*args, bo_key=tag, output_indices={0})
        print(f"[{tag}] COMPLETED")
    except Exception as e:
        print(f"[{tag}] HANG/FAIL: {e}")


if __name__ == "__main__":
    main()
