#!/usr/bin/env python3
"""Memtile weight-replay micro: resident slot stream re-emitted TPR times.

Memtile holds a wt-slot stream (~83 KB = 6 slots x 6944 u16 = 20832 i32);
a static memtile MM2S BD chain with repeat_count=TPR-1 replays it without
any shim traffic. The core arms one S2MM for TPR*MEM_N words and times the
receive: bytes / cycles = mem->core replay bandwidth vs ~1.7 GB/s shim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "microbench" / "memtile_program_cost"))

import memtile_program_cost as mpc  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.compiler.aiecc.main import run as aiecc_run  # noqa: F401  (env check)
from aie.dialects.aie import (  # noqa: E402
    AIEDevice, DMAChannelDir, EndOp, WireBundle, buffer, core, device,
    dma_bd, dma_start, flow, lock, memtile_dma, next_bd, shim_dma_allocation, tile,
)
from aie.dialects.aiex import (  # noqa: E402
    npu_maskwrite32, runtime_sequence, shim_dma_single_bd_task,
    dma_start_task, dma_await_task, dma_free_task,
)
from aie.extras.context import mlir_mod_ctx  # noqa: E402
from aie.iron.pythoc import aie_kernel, PythocKernel  # noqa: E402
from aie.utils import NPUKernel, DefaultNPURuntime  # noqa: E402
from aie.utils.compile import compile_mlir_module  # noqa: E402
from pythoc import ptr, i32, void  # noqa: E402
from pythoc.aie.operations import read_tm  # noqa: E402

SLOT_W = 3472          # one wslot (6944 u16) in i32 words
N_SLOT = 6
TPR = int(os.environ.get("TPR", "4"))
MEM_N = SLOT_W * N_SLOT
TELEM_N = 8


@aie_kernel
def replay_core(in_buf: ptr[i32, True], telem: ptr[i32, True], in_words: i32) -> void:
    t0: i32 = read_tm(TIMER_LOW)
    r: i32 = 0
    while r < N_RECV:
        program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_words, SLOT_W, 0)
        spin_lock_ge(LOCK0_VALUE, r + 1)
        r = r + 1
    t2: i32 = read_tm(TIMER_LOW)
    telem[0] = 0
    telem[1] = t2 - t0
    telem[2] = in_buf[0]
    telem[3] = in_buf[1]
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, 8, 1)
    spin_lock_ge(LOCK1_VALUE, 1)


def main():
    src = np.arange(100, 100 + MEM_N, dtype=np.int32)
    telem_ty = np.ndarray[(TELEM_N,), np.dtype[np.int32]]
    src_ty = np.ndarray[(MEM_N,), np.dtype[np.int32]]
    in_ty = np.ndarray[(SLOT_W,), np.dtype[np.int32]]
    g = mpc._globals(MEM_N)
    g["SLOT_W"] = SLOT_W
    g["N_RECV"] = TPR * N_SLOT
    kernel = PythocKernel(replay_core, [in_ty, telem_ty, np.int32],
                          target_arch="aie2p", extra_globals=g,
                          helpers=[mpc.program_dma_and_start, mpc.spin_lock_ge])

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def dev():
            kernel.resolve()
            t00, t01, t02 = tile(0, 0), tile(0, 1), tile(0, 2)
            mem_src = buffer(t01, datatype=src_ty, name="mem_src",
                             address=mpc.MEM_SRC_ADDR, initial_value=src)
            telem = buffer(t02, datatype=telem_ty, name="telem", address=mpc.CORE_TELEM_ADDR)
            in_buf = buffer(t02, datatype=in_ty, name="in_buf", address=mpc.CORE_IN_ADDR)
            lock(t02, lock_id=0, init=0, sym_name="s2mm_done")
            lock(t02, lock_id=1, init=0, sym_name="mm2s_done")
            flow(t01, WireBundle.DMA, 0, t02, WireBundle.DMA, 1)
            flow(t02, WireBundle.DMA, 1, t00, WireBundle.DMA, 0)
            shim_dma_allocation("telem_alloc", t00, DMAChannelDir.S2MM, 0)

            @memtile_dma(t01)
            def mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2],
                          repeat_count=TPR - 1)
                with block[1]:
                    dma_bd(mem_src, offset=0, len=MEM_N)
                    next_bd(block[2])
                with block[2]:
                    EndOp()

            @core(t02)
            def core_body():
                kernel(in_buf, telem, mpc.CORE_IN_ADDR // 4)

            @runtime_sequence(telem_ty)
            def seq(C):
                npu_maskwrite32(address=mpc.CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2)
                t = shim_dma_single_bd_task("telem_alloc", C, sizes=[1, 1, 1, TELEM_N], issue_token=True)
                dma_start_task(t)
                dma_await_task(t)
                dma_free_task(t)

        assert ctx.module.operation.verify()
        wd = HERE / "build_wt_replay" / f"tpr{TPR}"
        wd.mkdir(parents=True, exist_ok=True)
        os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
        compile_mlir_module(mlir_module=ctx.module, insts_path=str(wd / "insts.bin"),
                            xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    npu_kernel = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu_kernel)
    out = iron.zeros(TELEM_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [out])
    t = [int(x) for x in np.array(out.numpy())]
    cyc = t[1]
    bytes_total = TPR * MEM_N * 4
    gbs = bytes_total / (cyc / 1.8e9) / 1e9 if cyc else 0
    print(f"replay x{TPR}: arm={t[0]} cyc, recv={cyc} cyc, {bytes_total} B -> {gbs:.2f} GB/s "
          f"(first={t[2]}, second={t[3]})")


if __name__ == "__main__":
    main()
