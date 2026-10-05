# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Stream-B MMAC microbenchmark (npu3): A via object fifo, B via raw flow
# into the core stream port, C via object fifo.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

KT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
A_W = KT * 64 * 64 // 4
B_W = KT * 4096 // 4
C_W = 64 * 64 + 16

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu3)
    def device_body():
        a_ty = np.ndarray[(A_W,), np.dtype[np.int32]]
        b_ty = np.ndarray[(B_W,), np.dtype[np.int32]]
        c_ty = np.ndarray[(C_W,), np.dtype[np.int32]]
        gemm_tile = external_func(
            "gemm_tile", inputs=[a_ty, c_ty], link_with="project/kernel.o"
        )
        gemm_tile_last = external_func(
            "gemm_tile_last", inputs=[a_ty, c_ty], link_with="project/kernel.o"
        )
        gemm_stage_b = external_func(
            "gemm_stage_b", inputs=[], link_with="project/kernel.o"
        )
        shim = tile(0, 0)
        shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, 0])
        ct = tile(0, 2)
        packetflow(
            0,
            shim,
            WireBundle.Control32,
            0,
            [{"dest": shim, "port": WireBundle.Controller32, "channel": 0}],
        )
        inA = object_fifo("inA", shim, ct, 1, a_ty)
        outC = object_fifo("outC", ct, shim, 2, c_ty)
        # Load B once into the memtile and replay it on core stream port 1.
        mt = tile(0, 1)
        flow(shim, WireBundle.DMA, 1, mt, WireBundle.DMA, 0)
        flow(mt, WireBundle.DMA, 0, ct, WireBundle.Core, 1)
        shim_dma_allocation("inB", shim, DMAChannelDir.MM2S, 1)
        bufB = buffer(mt, b_ty, name="bufB")
        lk_empty = lock(mt, lock_id=0, init=1, sym_name="lk_empty")
        lk_full = lock(mt, lock_id=1, init=0, sym_name="lk_full")

        @memtile_dma(mt)
        def mt_dma(block):
            dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(lk_empty, LockAction.AcquireGreaterEqual, 1)
                dma_bd(bufB)
                use_lock(lk_full, LockAction.Release, 1)
                next_bd(block[1])
            with block[2]:
                dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(lk_full, LockAction.AcquireGreaterEqual, 1)
                dma_bd(bufB)
                use_lock(lk_full, LockAction.Release, 1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        @core(ct, stack_size=0x1000)
        def _():
            for _ in range_(sys.maxsize):
                a = inA.acquire(ObjectFifoPort.Consume, 1)
                gemm_stage_b()
                for _ in range_(CALLS - 1):
                    c = outC.acquire(ObjectFifoPort.Produce, 1)
                    gemm_tile(a, c)
                    outC.release(ObjectFifoPort.Produce, 1)
                c = outC.acquire(ObjectFifoPort.Produce, 1)
                gemm_tile_last(a, c)
                outC.release(ObjectFifoPort.Produce, 1)
                inA.release(ObjectFifoPort.Consume, 1)

        @runtime_sequence(a_ty, b_ty, c_ty)
        def sequence(A, B, C):
            ta = shim_dma_single_bd_task(inA, A, sizes=[1, 1, 1, A_W], issue_token=True)
            tb = shim_dma_single_bd_task(
                "inB",
                B,
                sizes=[1, 1, 1, B_W],
                issue_token=True,
            )
            tc = shim_dma_single_bd_task(
                outC,
                C,
                sizes=[1, CALLS, 1, C_W],
                strides=[0, C_W, 0, 1],
                issue_token=True,
            )
            dma_start_task(ta, tb, tc)
            dma_await_task(ta, tb, tc)

    print(ctx.module)
