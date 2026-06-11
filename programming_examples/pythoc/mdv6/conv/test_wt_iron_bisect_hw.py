#!/usr/bin/env python3
"""Forward bisect: add chain-shaped pieces to the PASSING bare micro.

STAGE=1 (default): chain channel layout, one core — memtile MM2S0 patch ring
(lock-paced) -> core S2MM0; memtile MM2S5 wt (ungated repeat, BD 44) ->
core S2MM1; core arms ch1 via mpc helper, consumes a patch per round then
4 wt slots. PASS proves the dual-channel layout is fine.
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
    device, dma_bd, dma_start, flow, lock, memtile_dma, mem as mem_op,
    next_bd, shim_dma_allocation, tile, use_lock,
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

SLOT_W = 2320
N_SLOT = 4
ROUNDS = 4
MEM_N = SLOT_W * N_SLOT
PATCH_W = 1152          # 12*12*32 patch in i32
TELEM_N = 8
WT_BUF = 0xC800
PATCH_BUF = 0x2000


@aie_kernel
def bench(patch: ptr[i32, True], wt: ptr[i32, True], telem: ptr[i32, True],
          wt_words: i32) -> void:
    t0: i32 = read_tm(TIMER_LOW)
    chk: i32 = 0
    nslot: i32 = 0
    r: i32 = 0
    while r < ROUNDS:
        # patch arrives via CDO-paced S2MM0 ring (cons lock 2 += 1)
        spin_lock_ge(LOCK2_VALUE, r + 1)
        chk = chk + patch[r % 64]
        s: i32 = 0
        while s < N_SLOT:
            nslot = nslot + 1
            program_dma_and_start(15, DMA_S2MM_1_START_QUEUE, wt_words, SLOT_W, 0)
            spin_lock_ge(LOCK0_VALUE, nslot)
            chk = chk + wt[s % 64]
            s = s + 1
        r = r + 1
    t1: i32 = read_tm(TIMER_LOW)
    telem[0] = t1 - t0
    telem[1] = chk
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, 8, 1)
    spin_lock_ge(LOCK1_VALUE, 1)


def main():
    telem_ty = np.ndarray[(TELEM_N,), np.dtype[np.int32]]
    src_ty = np.ndarray[(MEM_N,), np.dtype[np.int32]]
    psrc_ty = np.ndarray[(ROUNDS * PATCH_W,), np.dtype[np.int32]]
    wt_ty = np.ndarray[(SLOT_W,), np.dtype[np.int32]]
    patch_ty = np.ndarray[(PATCH_W,), np.dtype[np.int32]]
    g = mpc._globals(MEM_N)
    g.update(ROUNDS=ROUNDS, N_SLOT=N_SLOT, SLOT_W=SLOT_W,
             LOCK2_VALUE=0x0001F000 + 2 * 16)
    kernel = PythocKernel(bench, [patch_ty, wt_ty, telem_ty, np.int32],
                          target_arch="aie2p", extra_globals=g,
                          helpers=[mpc.program_dma_and_start, mpc.spin_lock_ge])

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def dev():
            kernel.resolve()
            t00, t01, t02 = tile(0, 0), tile(0, 1), tile(0, 2)
            wsrc = buffer(t01, datatype=src_ty, name="wsrc", address=0x70000,
                          initial_value=np.tile(np.arange(1, SLOT_W + 1, dtype=np.int32), N_SLOT))
            psrc = buffer(t01, datatype=psrc_ty, name="psrc", address=0x60000,
                          initial_value=np.arange(100, 100 + ROUNDS * PATCH_W, dtype=np.int32))
            telem = buffer(t02, datatype=telem_ty, name="telem", address=mpc.CORE_TELEM_ADDR)
            pbuf = buffer(t02, datatype=patch_ty, name="pbuf", address=PATCH_BUF)
            wbuf = buffer(t02, datatype=wt_ty, name="wbuf", address=WT_BUF)
            lock(t02, lock_id=0, init=0, sym_name="wt_done")
            lock(t02, lock_id=1, init=0, sym_name="telem_done")
            lk_pc = lock(t02, lock_id=2, init=0, sym_name="patch_cons")
            lk_pp = lock(t02, lock_id=3, init=ROUNDS, sym_name="patch_prod")
            lk_msrc = lock(t01, lock_id=0, init=ROUNDS, sym_name="mem_p")
            lk_dummy = lock(t01, lock_id=1, init=0, sym_name="mem_dummy")
            flow(t01, WireBundle.DMA, 0, t02, WireBundle.DMA, 0)  # patches
            flow(t01, WireBundle.DMA, 5, t02, WireBundle.DMA, 1)  # wt
            flow(t02, WireBundle.DMA, 1, t00, WireBundle.DMA, 0)  # telem
            shim_dma_allocation("telem_alloc", t00, DMAChannelDir.S2MM, 0)

            @memtile_dma(t01)
            def mt(block):
                dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(lk_msrc, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(psrc, offset=0, len=ROUNDS * PATCH_W, bd_id=0)
                    use_lock(lk_dummy, LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    EndOp()
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 5, dest=block[4], chain=block[5],
                              repeat_count=ROUNDS - 1)
                with block[4]:
                    dma_bd(wsrc, offset=0, len=MEM_N, bd_id=44)
                    next_bd(block[5])
                with block[5]:
                    EndOp()

            # core S2MM0: 4 patch BDs lock-paced ring
            @mem_op(t02)
            def cm(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(lk_pp, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pbuf, offset=0, len=PATCH_W, bd_id=0)
                    use_lock(lk_pc, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()

            @core(t02)
            def cb():
                kernel(pbuf, wbuf, telem, WT_BUF // 4)

            @runtime_sequence(telem_ty)
            def seq(C):
                npu_maskwrite32(address=mpc.CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2)
                t = shim_dma_single_bd_task("telem_alloc", C, sizes=[1, 1, 1, TELEM_N], issue_token=True)
                dma_start_task(t)
                dma_await_task(t)
                dma_free_task(t)

        assert ctx.module.operation.verify()
        wd = HERE / "build_wt_bisect"
        wd.mkdir(parents=True, exist_ok=True)
        os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
        compile_mlir_module(mlir_module=ctx.module, insts_path=str(wd / "insts.bin"),
                            xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    h = DefaultNPURuntime.load(NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE"))
    out = iron.zeros(TELEM_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [out])
    t = [int(x) for x in np.array(out.numpy())]
    want = sum(100 + r * PATCH_W + (r % 64) for r in range(ROUNDS)) + ROUNDS * sum((s % 64) + 1 for s in range(N_SLOT))
    print(f"bisect stage1: {t[0]} cyc chk={t[1]} want={want} {'PASS' if t[1]==want else 'FAIL'}")


if __name__ == "__main__":
    main()
