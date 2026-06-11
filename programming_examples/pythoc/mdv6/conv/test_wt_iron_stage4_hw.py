#!/usr/bin/env python3
"""Bisect stage 4: ONE iron Worker + patch/out FIFOs + raw wt replay patch.

First stage with iron lowering in the loop. Worker: per round acquire patch,
arm 4 wt slots (mpc helper), spin lock 12, stamp out elem, release. Raw
post-resolve patch adds memtile wt buffer + ungated MM2S5 ring -> core S2MM1.
PASS = iron worker context fine; FAIL = iron wrap is the chain wedge.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "microbench" / "memtile_program_cost"))
sys.path.insert(0, str(HERE.parents[0]))

import memtile_program_cost as mpc  # noqa: E402

from aie.iron import ObjectFifo, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import NPU2, Tile  # noqa: E402
from aie.iron.pythoc import aie_kernel, PythocKernel  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402
from aie.utils.compile import compile_mlir_module  # noqa: E402
from conv.resident_xclbin_runner import ResidentXCLBinRunner  # noqa: E402
from pythoc import ptr, i32, void  # noqa: E402

SLOT_W = 2320
N_SLOT = 4
ROUNDS = 4
MEM_N = SLOT_W * N_SLOT
PATCH_W = 1152
WT_BUF = 0xC800


@aie_kernel
def round_fn(patch: ptr[i32, True], out: ptr[i32, True], wt_words: i32, r: i32) -> void:
    n: i32 = r * N_SLOT
    s: i32 = 0
    while s < N_SLOT:
        n = n + 1
        if MODE >= 1:
            program_dma_and_start(15, DMA_S2MM_1_START_QUEUE, wt_words, SLOT_W, 0)
        if MODE >= 2:
            spin_lock_ge(LOCK0_VALUE, n)
        s = s + 1
    out[0] = 0x77AA
    out[1] = patch[1]
    out[2] = n


def main():
    patch_ty = np.ndarray[(PATCH_W,), np.dtype[np.int32]]
    out_ty = np.ndarray[(64,), np.dtype[np.int32]]
    host_in = np.ndarray[(ROUNDS * PATCH_W,), np.dtype[np.int32]]
    host_out = np.ndarray[(ROUNDS * 64,), np.dtype[np.int32]]

    g = mpc._globals(MEM_N)
    g.update(N_SLOT=N_SLOT, SLOT_W=SLOT_W, MODE=int(os.environ.get('MODE', '2')))
    krnd = PythocKernel(round_fn, [patch_ty, out_ty, np.int32, np.int32],
                        extra_globals=g,
                        helpers=[mpc.program_dma_and_start, mpc.spin_lock_ge])

    fin = ObjectFifo(patch_ty, depth=1, name="pin")
    fout = ObjectFifo(out_ty, depth=1, name="pout")

    def core_fn(a, o, k):
        for r in range(ROUNDS):
            e = a.acquire(1)
            eo = o.acquire(1)
            k(e, eo, WT_BUF // 4, r)
            a.release(1)
            o.release(1)

    w = Worker(core_fn, [fin.cons(), fout.prod(), krnd], tile=Tile(0, 2), stack_size=4096)

    rt = Runtime()
    with rt.sequence(host_in, host_out) as (IN, OUT):
        rt.start(w)
        for r in range(ROUNDS):
            tg = rt.task_group()
            rt.fill(fin.prod(), IN, TensorAccessPattern(
                (ROUNDS * PATCH_W,), offset=r * PATCH_W,
                sizes=[1, PATCH_W], strides=[0, 1]), task_group=tg)
            rt.drain(fout.cons(), OUT, TensorAccessPattern(
                (ROUNDS * 64,), offset=r * 64,
                sizes=[1, 1, 1, 64], strides=[0, 0, 0, 1]), task_group=tg, wait=True)
            rt.finish_task_group(tg)

    module = Program(NPU2(), rt).resolve_program()

    # lower placement + FIFOs, then patch raw wt path
    from aie.passmanager import PassManager
    PassManager.parse(
        "builtin.module(canonicalize,aie-canonicalize-device,"
        "aie.device(aie-place-tiles,aie-assign-lock-ids,aie-register-objectFifos,"
        "aie-objectFifo-stateful-transform{dynamic-objFifos=true}))",
        module.context).run(module.operation)

    import re as _re
    from aie.dialects.aie import (
        DMAChannelDir, EndOp, WireBundle, buffer, dma_bd, dma_start, flow,
        memtile_dma, next_bd,
    )
    from aie.ir import InsertionPoint, Location, IntegerAttr, IntegerType

    dev = next(o for o in module.body.operations if o.operation.name == "aie.device")
    body = dev.regions[0].blocks[0]
    tiles = {}
    for op in body.operations:
        if op.operation.name == "aie.tile":
            m = _re.search(r"col\s*=\s*(\d+),\s*row\s*=\s*(\d+)", str(op))
            tiles[(int(m.group(1)), int(m.group(2)))] = op
    if (0, 1) not in tiles:
        from aie.dialects.aie import tile as _t
        first = list(body.operations)[0]
        with InsertionPoint(first), Location.unknown(module.context):
            tiles[(0, 1)] = _t(0, 1)
    last = list(body.operations)[-1]
    src_ty = np.ndarray[(MEM_N,), np.dtype[np.int32]]
    with InsertionPoint(last), Location.unknown(module.context):
        wsrc = buffer(tiles[(0, 1)], datatype=src_ty, name="wsrc", address=0x70000,
                      initial_value=np.tile(np.arange(1, SLOT_W + 1, dtype=np.int32), N_SLOT))
        flow(tiles[(0, 1)], WireBundle.DMA, 5, tiles[(0, 2)], WireBundle.DMA, 1)

        @memtile_dma(tiles[(0, 1)])
        def mt(block):
            dma_start(DMAChannelDir.MM2S, 5, dest=block[1], chain=block[2],
                      repeat_count=ROUNDS - 1)
            with block[1]:
                dma_bd(wsrc, offset=0, len=MEM_N, bd_id=44)
                next_bd(block[2])
            with block[2]:
                EndOp()
    # iron never enables Core_Processor_Bus — core MMIO writes (DMA arming)
    # wedge without it; set it at sequence start
    from aie.dialects.aiex import npu_maskwrite32
    seq = next(o for o in body.operations if "runtime_sequence" in o.operation.name)
    with InsertionPoint.at_block_begin(seq.regions[0].blocks[0]), Location.unknown(module.context):
        npu_maskwrite32(address=mpc.CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2)
    assert module.operation.verify()

    wd = HERE / "build_wt_stage4"
    (wd / "work").mkdir(parents=True, exist_ok=True)
    os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
    compile_mlir_module(mlir_module=module, insts_path=str(wd / "insts.bin"),
                        xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd / "work"))
    inp = np.arange(100, 100 + ROUNDS * PATCH_W, dtype=np.uint32)
    r = ResidentXCLBinRunner(wd / "final.xclbin", wd / "insts.bin")
    res = r.run(inp.view(np.uint32), np.zeros(ROUNDS * 64, np.uint32), bo_key="s4",
                output_indices={1})
    out = res[1].view(np.int32).reshape(ROUNDS, 64)
    ok = all(out[r][0] == 0x77AA and out[r][1] == 100 + r * PATCH_W + 1 and out[r][2] == (r + 1) * N_SLOT
             for r in range(ROUNDS))
    print("stage4 rounds:", [(hex(out[r][0]), out[r][1], out[r][2]) for r in range(ROUNDS)],
          "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
