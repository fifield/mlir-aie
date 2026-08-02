#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# Hardware (npu3 / medusa) port of the matmul_fp8_sparse compute example.
#
# C[4x64] += A[4x128] * B_sparse[128x64] with fp8 inputs and fp32 output,
# using 4:2 structured sparsity: every quartet of 4 logical K positions
# keeps 2, so B is stored as a 64x64 fp8 matrix while A stays dense with
# 128 logical values per row.
#
# The kernel needs three inputs (A, B and the 192-word sparsity register
# image) but an AIE4 compute tile has only two input DMA channels, so A and
# the mask are streamed as one contiguous AM buffer and split again by the
# matmul_hw wrapper in kernel.cc.
#
#   DDR --(shim MM2S ch0)--> inAM --> core(0, 2) --> outC --(S2MM)--> DDR
#   DDR --(shim MM2S ch1)--> inB  -->
#

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Sizes in 32-bit words, matching the buffers in aie4.mlir.
A_WORDS = 128  # 4 x 128 dense fp8 = 512 bytes
MASK_WORDS = 192  # 6-Kbit sparsity register image = 768 bytes
AM_WORDS = A_WORDS + MASK_WORDS  # streamed as one buffer
B_WORDS = 1024  # 64 x 64 stored fp8 = 4096 bytes (32 x v128 staging loads)
C_WORDS = 256  # 4 x 64 fp32

COL = 0
COMPUTE_ROW = 2


def build(dev, iters_per_call=1):
    @device(dev)
    def device_body():
        am_ty = np.ndarray[(AM_WORDS,), np.dtype[np.int32]]
        b_ty = np.ndarray[(B_WORDS,), np.dtype[np.int32]]
        c_ty = np.ndarray[(C_WORDS,), np.dtype[np.int32]]

        matmul_hw = external_func(
            "matmul_hw",
            inputs=[am_ty, b_ty, c_ty],
            link_with="project/kernel.o",
        )

        shim = tile(COL, 0)
        shim.attributes["controller_id"] = AttrBuilder.get("PacketInfoAttr")([0, COL])
        compute = tile(COL, COMPUTE_ROW)

        # CERT control packet return path (required on npu3).
        packetflow(
            COL,
            shim,
            WireBundle.Control32,
            0,
            [{"dest": shim, "port": WireBundle.Controller32, "channel": 0}],
        )

        inAM = object_fifo("inAM", shim, compute, 2, am_ty)
        inB = object_fifo("inB", shim, compute, 2, b_ty)
        outC = object_fifo("outC", compute, shim, 2, c_ty)

        @core(compute)
        def _():
            for _ in range_(sys.maxsize):
                elem_am = inAM.acquire(ObjectFifoPort.Consume, 1)
                elem_b = inB.acquire(ObjectFifoPort.Consume, 1)
                elem_c = outC.acquire(ObjectFifoPort.Produce, 1)
                matmul_hw(elem_am, elem_b, elem_c)
                inAM.release(ObjectFifoPort.Consume, 1)
                inB.release(ObjectFifoPort.Consume, 1)
                outC.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(am_ty, b_ty, c_ty)
        def sequence(AM, B, C):
            for _ in range(iters_per_call):
                ta = shim_dma_single_bd_task(
                    inAM, AM, sizes=[1, 1, 1, AM_WORDS], issue_token=True
                )
                tb = shim_dma_single_bd_task(
                    inB, B, sizes=[1, 1, 1, B_WORDS], issue_token=True
                )
                tc = shim_dma_single_bd_task(
                    outC, C, sizes=[1, 1, 1, C_WORDS], issue_token=True
                )
                dma_start_task(ta, tb, tc)
                dma_await_task(ta, tb, tc)


parser = argparse.ArgumentParser(description="matmul_fp8_sparse for npu3")
parser.add_argument("-d", "--dev", default="npu3", help="AIE device (npu3 only)")
parser.add_argument(
    "-i", "--iters", type=int, default=1, help="iterations per kernel call"
)
args = parser.parse_args()

if args.dev != "npu3":
    raise ValueError(f"This example targets npu3; got {args.dev}")

with mlir_mod_ctx() as ctx:
    build(AIEDevice.npu3, iters_per_call=args.iters)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res, file=sys.stderr)
        sys.exit(1)
