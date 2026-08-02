#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# Hardware (npu3 / medusa) port of the matmul_mxfp4 compute example.
#
# The sim variant (aie4.mlir + test_sim.cpp) pokes the core's local
# memory directly.  This placed design instead moves the MX-packed A and
# B operands and the fp32 C result over real shim DMAs into/out of a
# single compute tile via object FIFOs, so the same kernel.cc runs
# unmodified on the NPU under XRT.
#
#   DDR --(shim MM2S ch0)--> inA  --> core(0, 2) --> outC --(S2MM)--> DDR
#   DDR --(shim MM2S ch1)--> inB  -->
#
# A is 544 bytes of MXFP4 (4 K_outer * M/2=2 vectors * 68 bytes),
# B is 8704 bytes (4 K_outer * 32 vectors * 68 bytes),
# C is 4x64 fp32 = 256 32-bit words.
#

import argparse
import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Must match the constants in kernel.cc / aie4.mlir.
A_BYTES = 544
B_BYTES = 8704
C_WORDS = 256

COL = 0
COMPUTE_ROW = 2


def build(dev, iters_per_call=1):
    @device(dev)
    def device_body():
        a_ty = np.ndarray[(A_BYTES,), np.dtype[np.int8]]
        b_ty = np.ndarray[(B_BYTES,), np.dtype[np.int8]]
        c_ty = np.ndarray[(C_WORDS,), np.dtype[np.int32]]

        matmul = external_func(
            "matmul",
            inputs=[a_ty, b_ty, c_ty],
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

        inA = object_fifo("inA", shim, compute, 2, a_ty)
        inB = object_fifo("inB", shim, compute, 2, b_ty)
        outC = object_fifo("outC", compute, shim, 2, c_ty)

        @core(compute)
        def _():
            for _ in range_(sys.maxsize):
                elem_a = inA.acquire(ObjectFifoPort.Consume, 1)
                elem_b = inB.acquire(ObjectFifoPort.Consume, 1)
                elem_c = outC.acquire(ObjectFifoPort.Produce, 1)
                matmul(elem_a, elem_b, elem_c)
                inA.release(ObjectFifoPort.Consume, 1)
                inB.release(ObjectFifoPort.Consume, 1)
                outC.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(a_ty, b_ty, c_ty)
        def sequence(A, B, C):
            for _ in range(iters_per_call):
                ta = shim_dma_single_bd_task(
                    inA, A, sizes=[1, 1, 1, A_BYTES], issue_token=True
                )
                tb = shim_dma_single_bd_task(
                    inB, B, sizes=[1, 1, 1, B_BYTES], issue_token=True
                )
                tc = shim_dma_single_bd_task(
                    outC, C, sizes=[1, 1, 1, C_WORDS], issue_token=True
                )
                dma_start_task(ta, tb, tc)
                dma_await_task(ta, tb, tc)


parser = argparse.ArgumentParser(description="matmul_mxfp4 for npu3")
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
