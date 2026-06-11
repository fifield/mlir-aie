#!/usr/bin/env python3
"""Bare-core validation of the CHAIN's arm kernel (chain_wt_arm).

Identical topology to test_wt_replay_col_hw (proven working with the mpc
helper): memtile MM2S0 ungated repeat streams the slot stream, NWORK cores
arm S2MM ch1 per slot — but here arming via chain_wt_arm (BD15, lock 12,
buffer at WT_BUF_ADDR, constant slot length). Isolates the chain's arm
primitive from the iron context.
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

import aie.iron as iron  # noqa: E402
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

from kernels.rn3_chain_pythoc import (  # noqa: E402
    chain_wt_arm, WT_BD, WT_LOCK, WT_BUF_ADDR, WT_BDBASE, WT_S2MM1Q,
)

SLOT_W = 2320
N_SLOT = 4
TPR = 4
NWORK = 4
MEM_N = SLOT_W * N_SLOT
TELEM_N = 8


@aie_kernel
def arm_bench(in_buf: ptr[i32, True], telem: ptr[i32, True], in_words: i32) -> void:
    t0: i32 = read_tm(TIMER_LOW)
    chk: i32 = 0
    r: i32 = 0
    while r < TPR * N_SLOT:
        program_dma_and_start(1, DMA_S2MM_1_START_QUEUE, in_words, WT_SLOT_I32, 0)
        spin_lock_ge(LOCK0_VALUE, r + 1)
        chk = chk + in_buf[r % 64]
        r = r + 1
    t1: i32 = read_tm(TIMER_LOW)
    telem[0] = t1 - t0
    telem[1] = chk
    program_dma_and_start(2, DMA_MM2S_1_START_QUEUE, TELEM_ADDR_WORDS, 8, 1)
    spin_lock_ge(LOCK1_VALUE, 1)


def main():
    src = np.tile(np.arange(1, SLOT_W + 1, dtype=np.int32), N_SLOT)
    telem_ty = np.ndarray[(TELEM_N,), np.dtype[np.int32]]
    src_ty = np.ndarray[(MEM_N,), np.dtype[np.int32]]
    in_ty = np.ndarray[(SLOT_W,), np.dtype[np.int32]]
    g = mpc._globals(MEM_N)
    g.update(TPR=TPR, N_SLOT=N_SLOT, SLOT_W=SLOT_W, WT_BD=WT_BD, WT_LOCK=WT_LOCK, WT_BUF_ADDR=WT_BUF_ADDR,
             WT_SLOT_I32=SLOT_W, WT_BDBASE=WT_BDBASE,
             WT_S2MM1Q=WT_S2MM1Q,
             LOCK12_VALUE=0x0001F000 + 12 * 16)
    kernel = PythocKernel(arm_bench, [in_ty, telem_ty, np.int32],
                          target_arch="aie2p", extra_globals=g,
                          helpers=[mpc.program_dma_and_start, mpc.spin_lock_ge])

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def dev():
            kernel.resolve()
            t00 = tile(0, 0)
            t01 = tile(0, 1)
            cores = [tile(0, 2 + i) for i in range(NWORK)]
            mem_src = buffer(t01, datatype=src_ty, name="mem_src",
                             address=mpc.MEM_SRC_ADDR, initial_value=src)
            telems, in_bufs = [], []
            for i, ct in enumerate(cores):
                telems.append(buffer(ct, datatype=telem_ty, name=f"telem{i}", address=mpc.CORE_TELEM_ADDR))
                in_bufs.append(buffer(ct, datatype=in_ty, name=f"in{i}", address=0x2000))
                lock(ct, lock_id=0, init=0, sym_name=f"s2mm_done{i}")
                lock(ct, lock_id=1, init=0, sym_name=f"mm2s_done{i}")
                lock(ct, lock_id=12, init=0, sym_name=f"wt_done{i}")
                flow(t01, WireBundle.DMA, 0, ct, WireBundle.DMA, 1)
            flow(cores[0], WireBundle.DMA, 1, t00, WireBundle.DMA, 0)
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

            def mk(ct, ib, tl):
                @core(ct)
                def cb():
                    kernel(ib, tl, 0x2000 // 4)
            for i, ct in enumerate(cores):
                mk(ct, in_bufs[i], telems[i])

            @runtime_sequence(telem_ty)
            def seq(C):
                for i in range(NWORK):
                    npu_maskwrite32(address=mpc.CORE_PROCESSOR_BUS, value=1, mask=1, column=0, row=2 + i)
                t = shim_dma_single_bd_task("telem_alloc", C, sizes=[1, 1, 1, TELEM_N], issue_token=True)
                dma_start_task(t)
                dma_await_task(t)
                dma_free_task(t)

        assert ctx.module.operation.verify()
        wd = HERE / "build_wt_arm_micro"
        wd.mkdir(parents=True, exist_ok=True)
        os.environ["PYTHOC_DEBUG_DIR"] = str(wd / "pythoc_objects")
        compile_mlir_module(mlir_module=ctx.module, insts_path=str(wd / "insts.bin"),
                            xclbin_path=str(wd / "final.xclbin"), work_dir=str(wd))
    h = DefaultNPURuntime.load(NPUKernel(str(wd / "final.xclbin"), str(wd / "insts.bin"), kernel_name="MLIR_AIE"))
    out = iron.zeros(TELEM_N, dtype=np.int32)
    DefaultNPURuntime.run(h, [out])
    t = [int(x) for x in np.array(out.numpy())]
    want = sum((r % 64) + 1 for r in range(TPR * N_SLOT))
    print(f"chain_wt_arm bare: {t[0]} cyc, chk={t[1]} want={want} "
          f"{'PASS' if t[1] == want else 'FAIL'}")


if __name__ == "__main__":
    main()
