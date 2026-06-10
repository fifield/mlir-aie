#!/usr/bin/env python3
"""Data-validating micro: 4D-iteration fill (chain in_tap shape) -> memtile
forward -> linear drain. Checks BYTES, not just completion (the dualtap micro
only checked completion; the chain bug is silent zero delivery)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYE = Path(__file__).resolve().parents[2]
if str(PYE) not in sys.path:
    sys.path.insert(0, str(PYE))

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.compile import compile_mlir_module
from conv.resident_xclbin_runner import ResidentXCLBinRunner

IC = 48
IMG = 44
IMG_H = 68
BO = IMG_H * IMG * IC            # 143616
CHUNK = 20 * 12 * IC             # 11520
N = 4
ELEMS = N * CHUNK                # 46080

FILL = os.environ.get("FILL", "4d")  # 4d | linear


def build():
    io_ty = np.ndarray[(ELEMS,), np.dtype[np.uint16]]
    bo_ty = np.ndarray[(BO,), np.dtype[np.uint16]]

    fin = ObjectFifo(io_ty, depth=1, name="f4_in")
    fout = fin.cons().forward(name="f4_out")

    fill_tap = (TensorAccessPattern((BO,), 0, [1, ELEMS], [0, 1]) if FILL == "linear"
                else TensorAccessPattern((BO,), 0, [N, 20, 12, IC],
                                         [16 * IMG * IC, IMG * IC, IC, 1]))
    drain_tap = TensorAccessPattern((BO,), 0, [1, ELEMS], [0, 1])

    rt = Runtime()
    with rt.sequence(bo_ty, bo_ty) as (A, B):
        tg = rt.task_group()
        rt.fill(fin.prod(), A, fill_tap, task_group=tg)
        rt.drain(fout.cons(), B, drain_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    wd = Path(__file__).parent / "build_fill4d_data" / FILL
    (wd / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = wd / "final.xclbin", wd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        compile_mlir_module(mlir_module=build(), insts_path=str(insts),
                            xclbin_path=str(xclbin), work_dir=str(wd / "work"),
                            verbose=False)

    src = (np.arange(BO, dtype=np.uint32) % 60000).astype(np.uint16) + 1
    runner = ResidentXCLBinRunner(xclbin, insts)
    res = runner.run(src.copy(), np.zeros(BO, np.uint16), bo_key=f"f4_{FILL}", output_indices={1})
    got = res[1][:ELEMS]

    if FILL == "linear":
        exp = src[:ELEMS]
    else:
        img = src.reshape(IMG_H, IMG, IC)
        exp = np.concatenate([img[16 * n:16 * n + 20, :12, :].reshape(-1) for n in range(N)])
    nz = np.count_nonzero(got)
    eq = np.array_equal(got, exp)
    print(f"[{FILL}] nonzero={nz}/{ELEMS} exact={'PASS' if eq else 'FAIL'}")
    if not eq:
        d = np.flatnonzero(got != exp)
        print(f"first mismatches at {d[:6]}; got {got[d[:4]]} exp {exp[d[:4]]}")


if __name__ == "__main__":
    main()
