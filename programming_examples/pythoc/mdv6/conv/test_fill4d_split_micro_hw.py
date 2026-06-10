#!/usr/bin/env python3
"""Data-validating micro #2: chain-shaped 4D fill -> 4-way memtile split ->
4 worker scalar copies -> join -> linear drain. Mirrors the rn3-chain in/out
FIFO topology (overlapping patch split chunks, final join), no weight FIFO."""
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

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.compile import compile_mlir_module
from conv.resident_xclbin_runner import ResidentXCLBinRunner

IC = 48
IMG = 44
IMG_H = 68
BO = IMG_H * IMG * IC
CHUNK = 20 * 12 * IC             # 11520 per worker view
COL = 4 * CHUNK                  # 46080 col fifo elems
OUT_W = 2 * 8 * 8 * IC           # 6144 finals per worker (2 tiles)
OUT_COL = 4 * OUT_W              # 24576

FILL = os.environ.get("FILL", "4d")
COPYN = int(os.environ.get("COPYN", str(OUT_W)))  # elems copied per worker
WT = int(os.environ.get("WT", "0"))               # add broadcast weight fifo, 12 acquires
WSLOT = 16 * 48 * 9 + 32


def build():
    col_ty = np.ndarray[(COL,), np.dtype[np.uint16]]
    w_in_ty = np.ndarray[(CHUNK,), np.dtype[np.uint16]]
    w_out_ty = np.ndarray[(OUT_W,), np.dtype[np.uint16]]
    out_ty = np.ndarray[(OUT_COL,), np.dtype[np.uint16]]
    bo_ty = np.ndarray[(BO,), np.dtype[np.uint16]]

    fin = ObjectFifo(col_ty, depth=1, name="s4_in")
    fout = ObjectFifo(out_ty, depth=1, name="s4_out")
    wslot_ty = np.ndarray[(WSLOT,), np.dtype[np.uint16]]
    fwt = ObjectFifo(wslot_ty, depth=1, name="s4_wt") if WT else None

    p_off = [0, CHUNK, 2 * CHUNK, 3 * CHUNK]
    f_off = [0, OUT_W, 2 * OUT_W, 3 * OUT_W]
    in_sp = fin.cons().split(offsets=p_off, obj_types=[w_in_ty] * 4,
                             depths=[1] * 4, names=[f"s4_in_{i}" for i in range(4)])
    out_j = fout.prod().join(offsets=f_off, obj_types=[w_out_ty] * 4,
                             depths=[1] * 4, names=[f"s4_out_{i}" for i in range(4)])

    def core_fn(a, o):
        ein = a.acquire(1)
        eout = o.acquire(1)
        i = 0
        while i < COPYN:
            eout[i] = ein[i]
            i = i + 1
        a.release(1)
        o.release(1)

    def core_fn_wt(a, o, w):
        ein = a.acquire(1)
        eout = o.acquire(1)
        s = 0
        while s < 12:
            w.acquire(1)
            w.release(1)
            s = s + 1
        i = 0
        while i < COPYN:
            eout[i] = ein[i]
            i = i + 1
        a.release(1)
        o.release(1)

    if WT:
        workers = [Worker(core_fn_wt, [in_sp[i].cons(), out_j[i].prod(), fwt.cons()])
                   for i in range(4)]
    else:
        workers = [Worker(core_fn, [in_sp[i].cons(), out_j[i].prod()]) for i in range(4)]

    fill_tap = (TensorAccessPattern((BO,), 0, [1, COL], [0, 1]) if FILL == "linear"
                else TensorAccessPattern((BO,), 0, [4, 20, 12, IC],
                                         [16 * IMG * IC, IMG * IC, IC, 1]))
    drain_tap = TensorAccessPattern((BO,), 0, [1, OUT_COL], [0, 1])

    rt = Runtime()
    with rt.sequence(bo_ty, bo_ty) as (A, B):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(fin.prod(), A, fill_tap, task_group=tg)
        if WT:
            rt.fill(fwt.prod(), A, TensorAccessPattern(
                (BO,), 0, [12, WSLOT], [WSLOT, 1]), task_group=tg)
        rt.drain(fout.cons(), B, drain_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    wd = Path(__file__).parent / "build_fill4d_split" / f"{FILL}_{COPYN}_w{WT}"
    (wd / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = wd / "final.xclbin", wd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        compile_mlir_module(mlir_module=build(), insts_path=str(insts),
                            xclbin_path=str(xclbin), work_dir=str(wd / "work"),
                            verbose=False)

    src = (np.arange(BO, dtype=np.uint32) % 60000).astype(np.uint16) + 1
    runner = ResidentXCLBinRunner(xclbin, insts)
    res = runner.run(src.copy(), np.zeros(BO, np.uint16), bo_key=f"s4_{FILL}_{COPYN}_w{WT}", output_indices={1})
    got = res[1][:OUT_COL].reshape(4, OUT_W)

    if FILL == "linear":
        col = src[:COL]
    else:
        img = src.reshape(IMG_H, IMG, IC)
        col = np.concatenate([img[16 * n:16 * n + 20, :12, :].reshape(-1) for n in range(4)])
    chunks = col.reshape(4, CHUNK)
    ok = all(np.array_equal(got[w, :COPYN], chunks[w, :COPYN]) for w in range(4))
    for w in range(4):
        nz = np.count_nonzero(got[w, :COPYN])
        eq = np.array_equal(got[w, :COPYN], chunks[w, :COPYN])
        print(f"worker{w}: nonzero={nz}/{COPYN} {'ok' if eq else 'MISMATCH'}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
