#!/usr/bin/env python3
"""Replay column micro — full chain wt topology, one column.

Per iter: shim refills the memtile slot stream from a host BO (S2MM ch0),
memtile MM2S0 broadcasts to NWORK cores, replaying the resident stream TPR
times per fill (double-buffer lock pacing scaled by TPR). Cores arm their
own S2MM per slot. Validates the wt-replay path that replaces the chain's
wt ObjectFifo. Telemetry from core 0; all cores check slot[0] sums.
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
from aie.dialects.aie import (  # noqa: E402
    AIEDevice, DMAChannelDir, EndOp, LockAction, WireBundle, buffer, core,
    device, dma_bd, dma_start, flow, lock, memtile_dma, next_bd,
    shim_dma_allocation, tile, use_lock,
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

SLOT_W = 2336      # re4w wslot (16*32*9+32 = 4640 u16) in i32 words
N_SLOT = 4         # slots per iter (2*N_BLK, ic32)
TPR = 4
N_ITERS = int(os.environ.get("N_ITERS", "3"))
NWORK = int(os.environ.get("NWORK", "4"))
MEM_N = SLOT_W * N_SLOT
TELEM_N = 8


@aie_kernel
def replay_core(in_buf: ptr[i32, True], telem: ptr[i32, True], in_words: i32) -> void:
    t0: i32 = read_tm(TIMER_LOW)
    chk: i32 = 0
    r: i32 = 0
    while r < N_ITERS * TPR * N_SLOT:
        program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_words, SLOT_W, 0)
        spin_lock_ge(LOCK0_VALUE, r + 1)
        chk = chk + in_buf[r % 64]  # varying index defeats load hoisting
        if r == 0:
            telem[5] = in_buf[2]
            telem[6] = in_buf[3]
            telem[7] = in_buf[SLOT_W - 1]
        r = r + 1
    t1: i32 = read_tm(TIMER_LOW)
    telem[0] = t1 - t0
    telem[1] = chk
    telem[2] = in_buf[0]
    telem[3] = in_buf[SLOT_W - 1]
    telem[4] = in_buf[1]
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, 8, 1)
    spin_lock_ge(LOCK1_VALUE, 1)


def main():
    telem_ty = np.ndarray[(TELEM_N,), np.dtype[np.int32]]
    src_ty = np.ndarray[(MEM_N,), np.dtype[np.int32]]
    in_ty = np.ndarray[(SLOT_W,), np.dtype[np.int32]]
    wt_ty = np.ndarray[(N_ITERS * MEM_N,), np.dtype[np.int32]]
    g = mpc._globals(MEM_N)
    g["SLOT_W"] = SLOT_W
    g["N_SLOT"] = N_SLOT
    g["TPR"] = TPR
    g["N_ITERS"] = N_ITERS
    kernel = PythocKernel(replay_core, [in_ty, telem_ty, np.int32],
                          target_arch="aie2p", extra_globals=g,
                          helpers=[mpc.program_dma_and_start, mpc.spin_lock_ge])

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def dev():
            kernel.resolve()
            t00 = tile(0, 0)
            t01 = tile(0, 1)
            cores = [tile(0, 2 + i) for i in range(NWORK)]
            mem_src = buffer(t01, datatype=src_ty, name="mem_src", address=mpc.MEM_SRC_ADDR,
                             initial_value=np.full(MEM_N, 777, np.int32))
            telems, in_bufs = [], []
            for i, ct in enumerate(cores):
                telems.append(buffer(ct, datatype=telem_ty, name=f"telem{i}", address=mpc.CORE_TELEM_ADDR))
                in_bufs.append(buffer(ct, datatype=in_ty, name=f"in{i}", address=mpc.CORE_IN_ADDR))
                lock(ct, lock_id=0, init=0, sym_name=f"s2mm_done{i}")
                lock(ct, lock_id=1, init=0, sym_name=f"mm2s_done{i}")
                flow(t01, WireBundle.DMA, 0, ct, WireBundle.DMA, 1)
            # memtile double-buffer locks scaled by TPR
            lk_empty = lock(t01, lock_id=0, init=TPR, sym_name="mt_empty")
            lk_full = lock(t01, lock_id=1, init=0, sym_name="mt_full")
            flow(t00, WireBundle.DMA, 0, t01, WireBundle.DMA, 0)
            flow(cores[0], WireBundle.DMA, 1, t00, WireBundle.DMA, 0)
            shim_dma_allocation("wt_in", t00, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("telem_alloc", t00, DMAChannelDir.S2MM, 0)

            STATIC = os.environ.get("STATIC", "0") == "1"

            @memtile_dma(t01)
            def mt(block):
                if STATIC:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2],
                              repeat_count=N_ITERS * TPR - 1)
                    with block[1]:
                        dma_bd(mem_src, offset=0, len=MEM_N)
                        next_bd(block[2])
                    with block[2]:
                        EndOp()
                    return
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(lk_empty, LockAction.AcquireGreaterEqual, value=TPR)
                    dma_bd(mem_src, offset=0, len=MEM_N)
                    use_lock(lk_full, LockAction.Release, value=TPR)
                    next_bd(block[1])
                with block[2]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                with block[3]:
                    use_lock(lk_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(mem_src, offset=0, len=MEM_N)
                    use_lock(lk_empty, LockAction.Release, value=1)
                    next_bd(block[3])
                with block[4]:
                    EndOp()

            def make_core(ct, ib, tl):
                @core(ct)
                def core_body():
                    kernel(ib, tl, mpc.CORE_IN_ADDR // 4)
            for i, ct in enumerate(cores):
                make_core(ct, in_bufs[i], telems[i])

            @runtime_sequence(wt_ty, telem_ty)
            def seq(WT, C):
                for i in range(NWORK):
                    npu_maskwrite32(address=mpc.CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2 + i)
                for it in range(0 if os.environ.get("STATIC", "0") == "1" else N_ITERS):
                    t = shim_dma_single_bd_task("wt_in", WT, offset=it * MEM_N,
                                                sizes=[1, 1, 1, MEM_N], issue_token=True)
                    dma_start_task(t)
                    dma_await_task(t)
                    dma_free_task(t)
                t = shim_dma_single_bd_task("telem_alloc", C, sizes=[1, 1, 1, TELEM_N], issue_token=True)
                dma_start_task(t)
                dma_await_task(t)
                dma_free_task(t)

        assert ctx.module.operation.verify()
        wd = HERE / "build_wt_replay_col"
        wd.mkdir(parents=True, exist_ok=True)
        os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
        compile_mlir_module(mlir_module=ctx.module, insts_path=str(wd / "insts.bin"),
                            xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    npu_kernel = NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu_kernel)
    wt = np.tile(np.arange(1, SLOT_W + 1, dtype=np.int32), N_ITERS * N_SLOT)
    wt_t = iron.tensor(wt.view(np.uint32))
    out = iron.zeros(TELEM_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [wt_t, out])
    t = [int(x) for x in np.array(out.numpy())]
    want = sum((r % 64) + 1 for r in range(N_ITERS * TPR * N_SLOT))
    n = N_ITERS * TPR * N_SLOT
    bw = n * SLOT_W * 4 / (t[0] / 1.8e9) / 1e9 if t[0] else 0
    print(f"col replay: {n} slots in {t[0]} cyc -> {bw:.2f} GB/s/core; chk={t[1]} want={want} "
          f"telem={t[2:8]} {'PASS' if t[1] == want else 'FAIL'}")


if __name__ == "__main__":
    main()
